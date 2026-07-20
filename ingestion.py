from pathlib import Path
from typing import Generator, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

from settings import settings #type:ignore
from logger import get_logger #type:ignore

logger = get_logger("ingestion")

# Per-process counter of how many FFmpeg fallback conversions have happened
# in THIS process. Safe as plain module state (not shared across processes)
# because each ProcessPoolExecutor worker gets its own fresh copy on fork/
# spawn -- there's no cross-process synchronization needed or intended here.
_conversion_count = [0]

class IngestionError(Exception):
    pass

class CorruptedVideoError(IngestionError):
    pass


def validate_video(video_path: Path) -> bool:
    """Quick sanity check: can we open it and read at least one frame?"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def _ensure_opencv_readable(video_path: Path) -> Path:
    """
    Try to open the video directly with OpenCV first. If that fails
    (unsupported codec/container -- common with cheaper CCTV/DVR exports),
    fall back to re-encoding it via FFmpeg into a standard H.264/mp4 file,
    then return the path to THAT file instead.

    Called automatically at the top of sample_frames(), so every entry
    point into ingestion (single-video CLI, batch queue, orchestrator
    later) gets this fallback for free -- no separate opt-in step needed.

    Returns a tuple (path_to_open, was_converted). was_converted is False
    when the original path is already OpenCV-readable, True when a temp
    converted copy was created -- callers use this flag to know whether
    they're responsible for deleting the temp file afterward.
    """
    if validate_video(video_path):
        return video_path, False  # OpenCV can already read it directly -- no conversion needed #type:ignore

    logger.warning(
        f"OpenCV cannot open {video_path.name} directly -- "
        f"attempting FFmpeg fallback conversion"
    )

    if shutil.which("ffmpeg") is None:
        raise IngestionError(
            f"Cannot open {video_path.name} with OpenCV, and FFmpeg is not "
            f"installed to attempt a fallback conversion. Install ffmpeg "
            f"(e.g. 'apt install ffmpeg') or pre-convert the file manually."
        )

    # Hash the full resolved path (not just the filename stem) so two
    # videos with the same name in different folders -- e.g. cam1/room101.mp4
    # and cam2/room101.mp4 -- never collide on the same temp file when
    # converted concurrently by different worker processes.
    tmp_dir = Path(tempfile.gettempdir())
    path_hash = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:10]
    converted_path = tmp_dir / f"{video_path.stem}_{path_hash}_converted.mp4"

    cmd = [
        "ffmpeg", "-y",              # -y: overwrite temp file if it already exists
        "-i", str(video_path),
        "-c:v", "libx264",           # standard, widely-supported codec
        "-pix_fmt", "yuv420p",       # ensures broad player/decoder compatibility
        "-an",                        # drop audio -- irrelevant for this pipeline,
                                       # smaller/faster output
        str(converted_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800  # 30 min safety cap
        )
    except subprocess.TimeoutExpired:
        raise IngestionError(f"FFmpeg conversion timed out for {video_path.name}")

    if result.returncode != 0:
        raise IngestionError(
            f"FFmpeg conversion failed for {video_path.name}: "
            f"{result.stderr[-500:]}"  # last 500 chars -- usually the actual error
        )

    if not validate_video(converted_path):
        raise IngestionError(
            f"FFmpeg conversion produced a file that OpenCV still cannot open: "
            f"{converted_path}"
        )

    logger.info(f"FFmpeg fallback succeeded -- using converted file: {converted_path}")
    _conversion_count[0] += 1
    return converted_path, True #type:ignore


def _compute_downscale_target(
    src_w: int, src_h: int, max_w: int, max_h: int
) -> Tuple[int, int]:
    """
    Compute a downscale-only target size that preserves aspect ratio.

    Only shrinks -- never enlarges. If the source already fits within
    (max_w, max_h) on both dimensions, returns the source size unchanged.
    Otherwise scales down uniformly (same factor on both axes) so the
    frame fits within the max bounds without stretching/warping it.
    """
    if src_w <= max_w and src_h <= max_h:
        return src_w, src_h  # already small enough -- leave untouched

    scale = min(max_w / src_w, max_h / src_h)  # uniform factor, preserves aspect ratio
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    return new_w, new_h


def _resize_frame(
    frame: np.ndarray, max_w: int, max_h: int
) -> np.ndarray:
    """
    Downscale a frame only if it exceeds (max_w, max_h), preserving aspect
    ratio (no warping) and without cropping (no content is cut off -- the
    whole original field of view is kept, just at a smaller pixel size).

    Uses INTER_AREA interpolation, which is the standard OpenCV choice for
    shrinking images: it averages pixel blocks into each output pixel,
    giving a clean, well-anti-aliased result rather than the blurring or
    aliasing artifacts you get from naively using INTER_LINEAR/INTER_CUBIC
    (those are designed for enlarging, not shrinking).
    """
    src_h, src_w = frame.shape[:2]
    new_w, new_h = _compute_downscale_target(src_w, src_h, max_w, max_h)

    if (new_w, new_h) == (src_w, src_h):
        return frame  # already within bounds -- no resize needed, avoids
                       # any unnecessary quality loss or wasted compute

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_frames(
    video_path: Path,
    target_fps: int = None,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Yield (timestamp_seconds, frame) tuples sampled at ~target_fps.

    MERGED VERSION -- combines the best of all implementations so far:

    - Frame skipping: SEQUENTIAL cap.read() + discard (measured faster and
      more reliable than repeated cv2.CAP_PROP_POS_FRAMES seeking on
      compressed video, which must decode from the nearest keyframe on
      every seek).

    - Timestamps: plain float seconds (CAP_PROP_POS_MSEC / 1000.0) --
      sortable, subtractable, no datetime anchor-date workaround needed.

    - EOF vs. corruption handling: checked against total_frames so a
      genuine end-of-video isn't mistaken for corruption.

    - FFmpeg fallback: automatic, via _ensure_opencv_readable() at the top
      of this function -- any caller (single-video CLI, batch queue,
      future orchestrator) gets this for free, no separate opt-in call.

    - Resizing: CONDITIONAL downscale-only, aspect-ratio-preserving
      (_resize_frame). Never warps non-16:9 sources, never upscales
      sources already smaller than the target resolution.
    """
    target_fps = target_fps or settings.TARGET_FPS
    video_path = Path(video_path)

    if not video_path.exists():
        raise IngestionError(f"Video not found: {video_path}")

    video_path, was_converted = _ensure_opencv_readable(video_path)  # FFmpeg fallback if #type:ignore

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IngestionError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            logger.warning("Source FPS unreadable, assuming 25 fps")
            orig_fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(orig_fps / target_fps))

        logger.info(
            f"Opening {video_path.name}: source_fps={orig_fps:.1f}, "
            f"total_frames={total_frames}, step={step} (-> ~{target_fps} fps sampled) "
            f"[sequential read+skip mode]"
        )

        frame_idx = 0
        consecutive_failures = 0
        yielded = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                # Could be genuine end-of-video, or a corrupt frame. Distinguish
                # by checking whether we're near the known total frame count.
                if total_frames and frame_idx < total_frames - step:
                    consecutive_failures += 1
                    logger.debug(f"Failed read at frame {frame_idx}, continuing")
                    if consecutive_failures > 10:
                        raise CorruptedVideoError(
                            f"Too many consecutive failed reads near frame {frame_idx}"
                        )
                    frame_idx += 1
                    continue
                break

            # Only keep every `step`-th frame -- sequential read + discard
            # instead of seeking.
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            consecutive_failures = 0
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = _resize_frame(frame, settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT)

            yield (timestamp_sec, frame)
            yielded += 1
            frame_idx += 1

        logger.info(f"Done sampling {video_path.name}: yielded {yielded} frames")

    finally:
        if cap is not None:
            cap.release()
        # Clean up the FFmpeg-converted temp copy regardless of how we exit
        # this generator (normal completion, exception, or early abandonment
        # by the caller) -- otherwise every non-OpenCV-native video leaves a
        # permanent orphaned file in /tmp.
        if was_converted:
            try:
                video_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned up temp converted file: {video_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {video_path}: {e}")


def sample_frames_chunked(
    video_path: Path,
    target_fps: int = None, #type:ignore
    chunk_size: int = None, #type:ignore
) -> Generator[List[Tuple[float, np.ndarray]], None, None]:
    """
    Same as sample_frames but batches frames into fixed-size chunks.
    Downstream stages (motion detection, etc.) can process chunk-by-chunk
    so memory stays bounded regardless of video length.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE_FRAMES
    chunk: List[Tuple[float, np.ndarray]] = []

    for ts, frame in sample_frames(video_path, target_fps):
        chunk.append((ts, frame))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


# --------------------------------------------------------------------------
# TICKET-02: Batch / parallel job queue -- process multiple videos/cameras
# concurrently. Needed since deployment has to scale across multiple exam
# halls/centres, not just one video at a time.
# --------------------------------------------------------------------------

@dataclass
class IngestionJobResult:
    video_path: str
    success: bool
    total_frames: int = 0
    total_chunks: int = 0
    processing_seconds: float = 0.0
    used_ffmpeg_conversion: bool = False
    error: Optional[str] = None


def _ingestion_worker(video_path: str, target_fps: int = None, chunk_size: int = None) -> IngestionJobResult:
    """
    Top-level function (required for ProcessPoolExecutor -- bound methods /
    closures aren't picklable across process boundaries). Per-video unit of
    work dispatched to worker processes.

    FFmpeg fallback conversion (if needed) happens transparently inside
    sample_frames_chunked -> sample_frames, via _ensure_opencv_readable.
    That function is the ONLY place that calls validate_video() now --
    this worker no longer pre-checks separately. That earlier pre-check
    was pure waste: it opened, read one frame, and closed the file, then
    _ensure_opencv_readable immediately did the exact same thing again a
    moment later. We get the same information (was a conversion needed)
    by reading the module-level counter that _ensure_opencv_readable
    increments, with no duplicate I/O.

    Currently exercises ingestion only (frame sampling + chunking); once
    the orchestrator exists, this is where the full pipeline call
    (motion -> roi -> object_detect -> segmentation -> ranking) plugs in
    for true end-to-end parallel processing across exam halls/cameras.
    """
    start = time.time()
    video_path = str(video_path)
    try:
        total_frames = 0
        total_chunks = 0
        conversions_before = _conversion_count[0]

        for chunk in sample_frames_chunked(video_path, target_fps, chunk_size):
            total_frames += len(chunk)
            total_chunks += 1

        will_convert = _conversion_count[0] > conversions_before

        return IngestionJobResult(
            video_path=video_path,
            success=True,
            total_frames=total_frames,
            total_chunks=total_chunks,
            processing_seconds=round(time.time() - start, 2),
            used_ffmpeg_conversion=will_convert,
        )
    except IngestionError as e:
        return IngestionJobResult(
            video_path=video_path,
            success=False,
            processing_seconds=round(time.time() - start, 2),
            error=str(e),
        )


class BatchIngestionQueue:
    """
    Simple multiprocessing job queue for processing multiple videos/cameras
    in parallel -- needed since deployment has to scale across multiple
    exam halls/centres, not just one video at a time.

    Uses ProcessPoolExecutor (stdlib, no extra infra) rather than
    Celery+Redis. For a hackathon deliverable this is the right tradeoff:
    zero extra services to run/demo. If this needs to scale beyond a
    single machine in a real deployment (multiple exam centres, distributed
    workers, job persistence/retry across restarts), swap this class for a
    Celery+Redis-backed queue -- the worker function (`_ingestion_worker`)
    stays the same either way, only the dispatch mechanism changes.
    """

    def __init__(self, max_workers: Optional[int] = None):
        # Default to CPU count; video processing is CPU-bound (decode +
        # OpenCV ops), so more workers than cores won't help and can
        # thrash memory on long videos.
        self.max_workers = max_workers

    def run(
        self,
        video_paths: List[str],
        target_fps: int = None, #type:ignore
        chunk_size: int = None, #type:ignore
        on_result: Optional[Callable[[IngestionJobResult], None]] = None,
    ) -> List[IngestionJobResult]:
        results: List[IngestionJobResult] = []
        logger.info(
            f"Dispatching {len(video_paths)} video(s) to batch queue "
            f"(max_workers={self.max_workers or 'auto/cpu_count'})"
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_ingestion_worker, vp, target_fps, chunk_size): vp
                for vp in video_paths
            }
            for future in as_completed(futures):
                vp = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = IngestionJobResult(video_path=str(vp), success=False, error=str(e))

                status = "OK" if result.success else f"FAILED ({result.error})"
                logger.info(
                    f"[{Path(result.video_path).name}] {status} -- "
                    f"{result.total_frames} frames, {result.processing_seconds}s"
                )
                results.append(result)
                if on_result:
                    on_result(result)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TICKET-01/02 standalone test (ingestion + batch queue)")
    parser.add_argument("--video", help="Single video (sequential test)")
    parser.add_argument("--videos", nargs="+", help="Multiple videos (parallel batch test)")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    args = parser.parse_args()

    if args.videos:
        queue = BatchIngestionQueue(max_workers=args.workers)
        t0 = time.time()
        results = queue.run(args.videos)
        elapsed = time.time() - t0

        ok = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {ok}/{len(results)} succeeded in {elapsed:.2f}s wall-clock")

    elif args.video:
        if not validate_video(args.video):
            logger.warning(
                "Video failed direct OpenCV validation -- sample_frames() "
                "will attempt an FFmpeg fallback conversion automatically"
            )

        total = 0
        for chunk in sample_frames_chunked(args.video):
            total += len(chunk)
            first_ts = chunk[0][0]
            last_ts = chunk[-1][0]
            logger.info(f"Chunk: {len(chunk)} frames, t={first_ts:.1f}s -> {last_ts:.1f}s")

        logger.info(f"TOTAL sampled frames: {total}")
    else:
        parser.error("Provide either --video (single) or --videos (batch)")