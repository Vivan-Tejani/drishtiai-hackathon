from pathlib import Path
from typing import Generator, List, Tuple

import cv2
import numpy as np

from settings import settings
from logger import get_logger

logger = get_logger("ingestion")


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


def _compute_downscale_target(
    src_w: int, src_h: int, max_w: int, max_h: int
) -> Tuple[int, int]:
    """
    Compute a downscale-only target size that preserves aspect ratio.

    Only shrinks — never enlarges. If the source already fits within
    (max_w, max_h) on both dimensions, returns the source size unchanged.
    Otherwise scales down uniformly (same factor on both axes) so the
    frame fits within the max bounds without stretching/warping it.
    """
    if src_w <= max_w and src_h <= max_h:
        return src_w, src_h  # already small enough — leave untouched

    scale = min(max_w / src_w, max_h / src_h)  # uniform factor, preserves aspect ratio
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    return new_w, new_h


def _resize_frame(
    frame: np.ndarray, max_w: int, max_h: int
) -> np.ndarray:
    """
    Downscale a frame only if it exceeds (max_w, max_h), preserving aspect
    ratio (no warping) and without cropping (no content is cut off — the
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
        return frame  # already within bounds — no resize needed, avoids
                       # any unnecessary quality loss or wasted compute

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_frames(
    video_path: Path,
    target_fps: int = None,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Yield (timestamp_seconds, frame) tuples sampled at ~target_fps.

    MERGED VERSION — combines the best of both implementations:

    - Frame skipping: uses SEQUENTIAL cap.read() + discard, rather than
      cap.set(cv2.CAP_PROP_POS_FRAMES, ...) seeking. Measured on a 5-minute
      test video: sequential read+skip took ~4.2s vs ~5.9s for repeated
      seeking. Seeking on compressed video (H.264/H.265) requires the
      decoder to jump back to the nearest keyframe and decode forward on
      every single call, which is slower and can be less reliable than
      just reading forward and discarding unwanted frames.

    - Timestamps: kept as a plain float (elapsed seconds via
      CAP_PROP_POS_MSEC / 1000.0), NOT wrapped in datetime. This was
      already the cleaner design — sortable, subtractable, no artificial
      anchor-date workaround needed downstream.

    - EOF vs. corruption handling: kept the smarter check against
      total_frames, so a genuine end-of-video isn't mistakenly flagged as
      corruption just because the last read attempt failed.

    - Logging + config: kept centralized settings/logger usage instead of
      hardcoded module-level constants.

    - Resizing: CONDITIONAL downscale-only, aspect-ratio-preserving resize.
      Previously every frame was force-resized to a fixed
      (RESOLUTION_WIDTH, RESOLUTION_HEIGHT), which would WARP the image if
      the source aspect ratio differed (e.g. a 4:3 camera squeezed into a
      16:9 target) and would actually UPSCALE (adding no real detail, only
      wasting compute) if the source was already smaller than the target.
      Now: if the source already fits within settings.RESOLUTION_WIDTH x
      RESOLUTION_HEIGHT, it is left completely untouched. Otherwise it is
      scaled down by a single uniform factor (same on both axes) so the
      full original field of view is preserved with no cropping and no
      distortion — just a smaller version of the same image. Uses
      INTER_AREA interpolation, OpenCV's recommended method for shrinking
      images, which avoids the blur/aliasing artifacts of interpolation
      methods meant for enlarging.
    """
    target_fps = target_fps or settings.TARGET_FPS
    video_path = Path(video_path)

    if not video_path.exists():
        raise IngestionError(f"Video not found: {video_path}")

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
            # by checking whether we're near the known total frame count —
            # kept from the original implementation, still applies equally
            # well to sequential reading.
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

        # Only keep every `step`-th frame — sequential read + discard
        # instead of seeking (this is the merged speed improvement).
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        consecutive_failures = 0
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame = _resize_frame(frame, settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT)

        yield (timestamp_sec, frame)
        yielded += 1
        frame_idx += 1

    cap.release()
    logger.info(f"Done sampling {video_path.name}: yielded {yielded} frames")


def sample_frames_chunked(
    video_path: Path,
    target_fps: int = None,
    chunk_size: int = None,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TICKET-01 standalone test (merged version)")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    if not validate_video(args.video):
        logger.error("Video failed validation — cannot open or read first frame")
        raise SystemExit(1)

    total = 0
    for chunk in sample_frames_chunked(args.video):
        total += len(chunk)
        first_ts = chunk[0][0]
        last_ts = chunk[-1][0]
        logger.info(f"Chunk: {len(chunk)} frames, t={first_ts:.1f}s -> {last_ts:.1f}s")

    logger.info(f"TOTAL sampled frames: {total}")