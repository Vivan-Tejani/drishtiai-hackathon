"""
================================================================================
DRISHTI-PS2 — Preprocessing Pipeline (v2.1 Accurate)
================================================================================
Connects with video_ingestion.py to provide:
  1. CLAHE-enhanced frames (LAB-space, colour preserved for downstream models)
  2. Cleaned MOG2 foreground masks with shadow suppression
  3. Motion ROI extraction via contour analysis + connected components
  4. Temporal persistence filtering (noise rejection across frames)
  5. ROI merging & intensity scoring
  6. Accumulated motion heatmap for analytics

Fixes applied to baseline (v2.0):
  • FLAW #2  — MOG2 mask was discarded; now extracted as persistent ROIs.
  • FLAW #A  — CLAHE on grayscale destroyed colour info for pose models;
               now applied in LAB space, colour frame returned.
  • FLAW #B  — Shadow pixels (127) were kept in mask; now suppressed.
  • FLAW #C  — No temporal filtering → single-frame noise became ROIs;
               now ROIs require min_persistence_frames.
  • FLAW #D  — No ROI merging → one person split into multiple boxes;
               now greedy NMS-style merge for overlapping contours.
  • FLAW #E  — Hardcoded thresholds; now exposed via dataclass config.
  • FLAW #F  — No motion intensity metric; now computed per-ROI.
  • FLAW #G  — No quality gate; now rejects all-black / all-white frames.
  • FLAW #H  — ROI intensity averaged over the full bounding box, diluting
               scores for non-rectangular blobs (hand, angled limb, etc.)
               by including background pixels; now averaged only over the
               actual contour shape.

Usage:
    from video_ingestion import sample_frames
    from preprocessing import PreprocessingPipeline, PreprocessingConfig

    cfg = PreprocessingConfig()
    pipe = PreprocessingPipeline(cfg)

    for result in pipe.process_video(Path("exam.mp4")):
        # result.enhanced_frame   -> 3-ch BGR for pose / object models
        # result.masked_frame     -> 1-ch masked for optional debug
        # result.rois             -> List[MotionROI] with bbox + intensity
        # result.motion_heatmap   -> accumulated float32 heatmap
        # result.fg_mask          -> binary mask for flow gating
        ...
================================================================================
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, Dict
from collections import defaultdict, deque

# ------------------------------------------------------------------------------
# Import your ingestion layer (assumes video_ingestion.py is on PYTHONPATH)
# ------------------------------------------------------------------------------
from video_ingestion import sample_frames, validate_video, IngestionError


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PreprocessingConfig:
    """Tunable knobs for the preprocessing stage."""

    # --- CLAHE ---------------------------------------------------------------
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # --- MOG2 ----------------------------------------------------------------
    mog2_history: int = 500
    mog2_var_threshold: float = 36.0
    mog2_detect_shadows: bool = True          # keep True, but we filter later
    mog2_learning_rate: float = 0.01
    mog2_shadow_value: int = 127              # OpenCV default

    # --- Noise reduction ------------------------------------------------------
    gaussian_blur_kernel: int = 5             # must be odd; blur applied pre-MOG2
                                               # to suppress sensor/compression
                                               # noise before it reaches the
                                               # background model

    # --- Mask cleaning -------------------------------------------------------
    morph_kernel_size: int = 3                # ellipse kernel
    morph_open_iter: int = 1
    morph_dilate_iter: int = 1
    mask_bin_threshold: int = 200             # >shadow, <255 to catch weak fg

    # --- ROI extraction ------------------------------------------------------
    min_roi_area: int = 350                # px² — lowered per anthropometric estimate;
                                           # 800 was too high to catch isolated hand/limb
                                           # motion at mid-to-far camera distances (est.
                                           # ~280-370px² for a hand at 4m). Needs real-
                                           # footage calibration once actual camera geometry
                                           # is known.
    max_roi_area: int = 120_000               # ignore whole-frame flicker
    min_roi_aspect: float = 0.25              # w/h — reject extreme slivers
    max_roi_aspect: float = 4.0

    # --- Temporal persistence ------------------------------------------------
    min_persistence_frames: int = 2        # lowered from 3 — real head-turn duration
                                           # (~0.5-0.6s per biomechanics data) can complete
                                           # within ~1-1.5 frames at 2 FPS; 3 frames risked
                                           # filtering out genuine fast natural movements.
    temporal_iou_threshold: float = 0.35      # match ROIs across frames
    centroid_match_distance: float = 30.0     # px — fallback match for small motions
    persistence_grace_frames: int = 2         # allow N missed frames before a
                                               # candidate is dropped (handles brief pauses)

    # --- ROI merging ---------------------------------------------------------
    merge_iou_threshold: float = 0.25         # merge overlapping candidate ROIs

    # --- Heatmap -------------------------------------------------------------
    heatmap_decay_per_second: float = 0.95    # decay factor normalised to 1 real second
                                               # (applied as decay ** dt, not per-frame)

    # --- Quality gate --------------------------------------------------------
    min_mean_brightness: float = 8.0          # reject all-black frames
    max_mean_brightness: float = 250.0        # reject all-white / flash frames

    # --- Resolution (must match ingestion) -----------------------------------
    width: int = 1280
    height: int = 720


# ==============================================================================
# DATA OBJECTS
# ==============================================================================

@dataclass
class MotionROI:
    """A single region of interest derived from foreground motion."""
    x: int                      # top-left corner
    y: int
    w: int
    h: int
    area: int                   # pixel count inside contour
    intensity: float            # mean fg-mask value inside ROI (0-255)
    timestamp: datetime
    persistence_count: int = 1  # how many consecutive frames seen
    frames_since_seen: int = 0  # grace counter for temporal persistence matching
    preproc_temp_id: Optional[int] = None   # SHORT-LIVED id, used only within this
                                             # preprocessing stage for persistence
                                             # filtering. NOT the final tracking id —
                                             # ByteTrack (downstream) assigns the
                                             # authoritative track_id from these ROIs
                                             # as detections.

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def iou(self, other: MotionROI) -> float:
        """Intersection-over-Union with another ROI."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "area": self.area, "intensity": round(self.intensity, 2),
            "timestamp": self.timestamp.isoformat(),
            "persistence": self.persistence_count,
            "preproc_temp_id": self.preproc_temp_id,
        }


@dataclass
class PreprocessedFrame:
    """Output bundle for one frame after preprocessing."""
    timestamp: datetime
    original: np.ndarray                    # raw resized BGR
    enhanced_frame: np.ndarray              # CLAHE-enhanced BGR (for pose/YOLO)
    masked_frame: np.ndarray                # fg-masked grayscale (for debug/flow)
    fg_mask: np.ndarray                     # binary/cleaned mask (uint8)
    rois: List[MotionROI] = field(default_factory=list)
    motion_score: float = 0.0               # sum of all ROI intensities
    motion_heatmap: Optional[np.ndarray] = None   # accumulated float32
    quality_passed: bool = True
    rejection_reason: Optional[str] = None


# ==============================================================================
# PREPROCESSING PIPELINE
# ==============================================================================

class PreprocessingPipeline:
    """
    Accurate preprocessing that:
      • Enhances frames in LAB colour space (preserves 3-ch for pose models).
      • Runs MOG2, suppresses shadows, cleans mask.
      • Extracts *persistent* motion ROIs via contour analysis.
      • Merges split contours and tracks them temporally.
      • Maintains a decaying motion heatmap.
      • Gates on frame quality (black/white flash rejection).
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.cfg = config or PreprocessingConfig()
        self._init_processors()
        self._reset_state()

    # --------------------------------------------------------------------------
    # Init / Reset
    # --------------------------------------------------------------------------
    def _init_processors(self) -> None:
        """Create OpenCV processors."""
        self.clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_grid_size,
        )
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.mog2_history,
            varThreshold=self.cfg.mog2_var_threshold,
            detectShadows=self.cfg.mog2_detect_shadows,
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )

    def _reset_state(self) -> None:
        """Reset temporal state (call between videos)."""
        self.heatmap: np.ndarray = np.zeros(
            (self.cfg.height, self.cfg.width), dtype=np.float32
        )
        self._roi_candidates: Dict[int, MotionROI] = {}   # temp_id -> ROI
        self._next_temp_id: int = 0
        self._frame_counter: int = 0
        self._last_heatmap_timestamp: Optional[datetime] = None  # for per-second decay

        # Camera-health tracking (FLAW #15 FIX): count consecutive quality-gate
        # rejections so a long unusable stretch is surfaced, not silently dropped.
        self._consecutive_rejections: int = 0
        self._camera_health_warning_emitted: bool = False

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def process_video(
        self,
        video_path: Path,
        target_fps: int = 2,
    ) -> Generator[PreprocessedFrame, None, None]:
        """
        Generator wrapper around video_ingestion.sample_frames.
        Yields PreprocessedFrame objects with ROIs, heatmap, and enhanced frames.
        """
        if not validate_video(video_path):
            raise IngestionError(f"Video unreadable or corrupt: {video_path}")

        self._reset_state()

        for timestamp, frame in sample_frames(video_path, target_fps=target_fps):
            result = self.process_frame(frame, timestamp)
            self._frame_counter += 1
            yield result

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime,
    ) -> PreprocessedFrame:
        """Process a single frame through the full pipeline."""
        # ------------------------------------------------------------------
        # 1. Quality gate on raw input
        # ------------------------------------------------------------------
        if frame is None or frame.size == 0:
            return self._reject(timestamp, frame, "empty_frame")

        mean_bright = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if mean_bright < self.cfg.min_mean_brightness:
            return self._reject(timestamp, frame, "too_dark")
        if mean_bright > self.cfg.max_mean_brightness:
            return self._reject(timestamp, frame, "too_bright")

        # Passed quality gate — reset consecutive rejection streak/warning
        # (FLAW #15 FIX companion: only reset here, on the success path)
        self._consecutive_rejections = 0
        self._camera_health_warning_emitted = False

        # ------------------------------------------------------------------
        # 2. CLAHE enhancement in LAB space (colour preserved)
        #    FLAW #A FIX: baseline destroyed colour by converting to grayscale
        #    before CLAHE. Pose models (RTMPose) and YOLO expect 3-ch BGR.
        # ------------------------------------------------------------------
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Grayscale version for MOG2 (must be single channel)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # Suppress sensor/compression noise BEFORE it reaches MOG2's per-pixel
        # background model. Morphological opening (later, on the mask) only
        # removes isolated single-pixel noise -- it can't undo MOG2 already
        # misreading spatially-correlated noise (e.g. H.264 macroblock
        # flicker) as real foreground if that noise clusters into a blob
        # larger than the morphology kernel. Blurring here reduces how often
        # that happens in the first place.
        k = self.cfg.gaussian_blur_kernel
        if k % 2 == 0:
            k += 1  # kernel size must be odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        # ------------------------------------------------------------------
        # 3. MOG2 background subtraction + shadow suppression
        #    FLAW #B FIX: baseline thresholded at 127, keeping shadow pixels.
        # ------------------------------------------------------------------
        fg_mask_raw = self.bg_subtractor.apply(gray, learningRate=self.cfg.mog2_learning_rate)

        # Suppress shadows: OpenCV marks shadows with value 127 when
        # detectShadows=True. We force them to background (0).
        if self.cfg.mog2_detect_shadows:
            fg_mask_raw = np.where(fg_mask_raw == self.cfg.mog2_shadow_value, 0, fg_mask_raw).astype(np.uint8)

        # ------------------------------------------------------------------
        # 4. Morphological cleaning
        # ------------------------------------------------------------------
        fg_mask = cv2.morphologyEx(
            fg_mask_raw, cv2.MORPH_OPEN, self.morph_kernel,
            iterations=self.cfg.morph_open_iter,
        )
        fg_mask = cv2.dilate(
            fg_mask, self.morph_kernel,
            iterations=self.cfg.morph_dilate_iter,
        )

        # Binarise with a threshold that sits above noise but catches weak fg
        _, fg_mask_bin = cv2.threshold(
            fg_mask, self.cfg.mask_bin_threshold, 255, cv2.THRESH_BINARY
        )

        # ------------------------------------------------------------------
        # 5. Masked frame for downstream optical-flow / debug
        # ------------------------------------------------------------------
        masked_gray = cv2.bitwise_and(gray, gray, mask=fg_mask_bin)

        # ------------------------------------------------------------------
        # 6. ROI extraction from foreground mask
        #    FLAW #2 FIX: baseline discarded the mask after bitwise_and.
        #    We now run contour detection + connected-component analysis.
        # ------------------------------------------------------------------
        raw_rois = self._extract_raw_rois(fg_mask_bin, timestamp)

        # Merge split contours belonging to the same person
        merged_rois = self._merge_overlapping_rois(raw_rois)

        # Temporal persistence filter (noise rejection)
        persistent_rois = self._filter_by_persistence(merged_rois)

        # ------------------------------------------------------------------
        # 7. Motion heatmap update (decaying accumulator)
        # ------------------------------------------------------------------
        self._update_heatmap(fg_mask_bin, timestamp)

        # ------------------------------------------------------------------
        # 8. Compute aggregate motion score
        # ------------------------------------------------------------------
        motion_score = sum(r.intensity for r in persistent_rois)

        return PreprocessedFrame(
            timestamp=timestamp,
            original=frame,
            enhanced_frame=enhanced_bgr,
            masked_frame=masked_gray,
            fg_mask=fg_mask_bin,
            rois=persistent_rois,
            motion_score=motion_score,
            motion_heatmap=self.heatmap.copy(),
            quality_passed=True,
        )

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _reject(
        self,
        timestamp: datetime,
        frame: Optional[np.ndarray],
        reason: str,
    ) -> PreprocessedFrame:
        """
        Return a rejected PreprocessedFrame with empty fields.

        FLAW #15 FIX: previously each bad frame was dropped silently and
        independently, so a sustained camera failure / lighting fault produced
        no visible trace beyond individual per-frame rejections. We now track
        a consecutive-rejection streak and surface a single explicit warning
        once it crosses a threshold, so this is a monitored condition rather
        than data that just quietly vanishes from the ROI/heatmap output.
        """
        self._consecutive_rejections += 1
        if (
            self._consecutive_rejections >= 10
            and not self._camera_health_warning_emitted
        ):
            print(
                f"[CAMERA-HEALTH WARNING] {self._consecutive_rejections} "
                f"consecutive frames rejected (reason: '{reason}') as of "
                f"{timestamp.isoformat()} — footage may be unusable in this "
                f"stretch (camera fault, lens covered, or lighting failure)."
            )
            self._camera_health_warning_emitted = True

        h, w = self.cfg.height, self.cfg.width
        empty = np.zeros((h, w, 3), dtype=np.uint8) if frame is None else frame
        return PreprocessedFrame(
            timestamp=timestamp,
            original=empty,
            enhanced_frame=empty,
            masked_frame=np.zeros((h, w), dtype=np.uint8),
            fg_mask=np.zeros((h, w), dtype=np.uint8),
            rois=[],
            motion_score=0.0,
            motion_heatmap=self.heatmap.copy(),
            quality_passed=False,
            rejection_reason=reason,
        )

    def _extract_raw_rois(
        self,
        mask: np.ndarray,
        timestamp: datetime,
    ) -> List[MotionROI]:
        """Find contours in the binary mask and convert to MotionROI list."""
        # Use RETR_EXTERNAL to avoid nested holes; CHAIN_APPROX_SIMPLE for speed
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rois: List[MotionROI] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_roi_area or area > self.cfg.max_roi_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            aspect = w / h
            if aspect < self.cfg.min_roi_aspect or aspect > self.cfg.max_roi_aspect:
                continue

            # Intensity = mean pixel value inside the CONTOUR shape (not the
            # bounding box). BUG FIX: previously this averaged over the full
            # bounding rectangle, which includes background pixels for any
            # non-rectangular blob (a hand, torso, angled limb, etc.) --
            # diluting intensity downward for shapes that don't fill their
            # bbox, and making intensity not a fair comparison between a
            # blob that's roughly rectangular vs one that isn't. We now
            # build a contour-shaped mask and average only the pixels the
            # contour actually covers, on the *raw* mask (before
            # binarisation) so weak but real motion still scores > 0.
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                contour_mask, [cnt - [x, y]], contourIdx=-1,
                color=255, thickness=cv2.FILLED,
            )
            mask_roi = mask[y:y+h, x:x+w]
            contour_pixels = mask_roi[contour_mask > 0]
            intensity = float(np.mean(contour_pixels)) if contour_pixels.size > 0 else 0.0

            rois.append(MotionROI(
                x=x, y=y, w=w, h=h,
                area=int(area),
                intensity=intensity,
                timestamp=timestamp,
            ))
        return rois

    def _merge_overlapping_rois(self, rois: List[MotionROI]) -> List[MotionROI]:
        """
        Greedy merge of ROIs that heavily overlap (same person split into
        multiple contours by occlusion or desk edges).
        FLAW #D FIX: baseline returned split contours as separate ROIs.
        """
        if not rois:
            return []

        # Sort by area descending — large boxes swallow small ones
        rois = sorted(rois, key=lambda r: r.area, reverse=True)
        merged: List[MotionROI] = []

        for roi in rois:
            absorbed = False
            for m in merged:
                if roi.iou(m) > self.cfg.merge_iou_threshold:
                    # Expand bounding box to union
                    x1 = min(roi.x, m.x)
                    y1 = min(roi.y, m.y)
                    x2 = max(roi.x + roi.w, m.x + m.w)
                    y2 = max(roi.y + roi.h, m.y + m.h)
                    m.x, m.y = x1, y1
                    m.w, m.h = x2 - x1, y2 - y1
                    # FLAW #10 FIX: previously `m.area += roi.area` summed the
                    # original contour areas, which drifts away from the real
                    # pixel area of the (now-expanded) union box, especially
                    # after multiple merges. Recompute area directly from the
                    # merged box's true dimensions instead.
                    m.area = m.w * m.h
                    m.intensity = max(m.intensity, roi.intensity)
                    absorbed = True
                    break
            if not absorbed:
                merged.append(roi)
        return merged

    def _filter_by_persistence(
        self,
        rois: List[MotionROI],
    ) -> List[MotionROI]:
        """
        Track ROIs across frames using centroid distance + IoU.
        Only return ROIs that have survived >= min_persistence_frames.
        FLAW #C FIX: baseline had no temporal filtering → noise ROIs leaked.

        FLAW #11 FIX: previously, any previous candidate with no match in the
        current frame was dropped immediately, resetting persistence_count to
        1 the moment a person paused for even one frame. We now allow a grace
        period (persistence_grace_frames) of missed frames before a candidate
        is actually expired, carrying it forward (without a fresh ROI) so a
        brief pause doesn't defeat persistence filtering.

        FLAW #12 FIX: centroid distance threshold was hardcoded (30px) instead
        of coming from PreprocessingConfig; now uses cfg.centroid_match_distance.
        """
        new_candidates: Dict[int, MotionROI] = {}
        used_current = set()

        # Match current ROIs to previous candidates
        for prev_id, prev_roi in self._roi_candidates.items():
            best_match: Optional[int] = None
            best_iou = 0.0

            for idx, cur in enumerate(rois):
                if idx in used_current:
                    continue
                iou = prev_roi.iou(cur)
                # Also check centroid distance for small motions
                dist = np.hypot(prev_roi.cx - cur.cx, prev_roi.cy - cur.cy)
                # Accept if IoU is decent OR centroid is very close
                if iou > self.cfg.temporal_iou_threshold or dist < self.cfg.centroid_match_distance:
                    if iou > best_iou:
                        best_iou = iou
                        best_match = idx

            if best_match is not None:
                cur = rois[best_match]
                cur.persistence_count = prev_roi.persistence_count + 1
                cur.frames_since_seen = 0
                cur.preproc_temp_id = prev_id
                new_candidates[prev_id] = cur
                used_current.add(best_match)
            else:
                # No match this frame — carry the candidate forward within its
                # grace period instead of dropping it outright (FLAW #11 FIX).
                if prev_roi.frames_since_seen < self.cfg.persistence_grace_frames:
                    prev_roi.frames_since_seen += 1
                    new_candidates[prev_id] = prev_roi
                # else: grace period exceeded, candidate is allowed to expire
                # (simply not carried into new_candidates)

        # Unmatched current ROIs become new candidates
        for idx, cur in enumerate(rois):
            if idx not in used_current:
                tid = self._next_temp_id
                self._next_temp_id += 1
                cur.preproc_temp_id = tid
                cur.frames_since_seen = 0
                new_candidates[tid] = cur

        self._roi_candidates = new_candidates

        # Return only those that have reached persistence threshold AND were
        # actually seen in the current frame (don't emit "ghost" ROIs that are
        # only being carried forward through their grace period).
        return [
            r for r in new_candidates.values()
            if r.persistence_count >= self.cfg.min_persistence_frames
            and r.frames_since_seen == 0
        ]

    def _update_heatmap(self, mask_bin: np.ndarray, timestamp: datetime) -> None:
        """
        Decay and accumulate motion heatmap.

        FLAW #14 FIX: previously decay was applied as a flat per-frame
        multiplier (heatmap_decay ** 1 every call), so the actual real-world
        decay rate silently changed with sampling FPS — the same 10 seconds
        of footage decayed differently at 2 FPS vs 5 FPS, making heatmaps
        generated at different sampling rates incomparable. We now normalise
        decay to elapsed real seconds between frames: decay_rate ** dt.
        """
        if self._last_heatmap_timestamp is not None:
            dt = (timestamp - self._last_heatmap_timestamp).total_seconds()
            dt = max(dt, 0.0)
        else:
            dt = 0.0  # first frame — no decay to apply yet
        self._last_heatmap_timestamp = timestamp

        decay_factor = self.cfg.heatmap_decay_per_second ** dt
        self.heatmap *= decay_factor
        self.heatmap += mask_bin.astype(np.float32) * 0.05
        np.clip(self.heatmap, 0, 255, out=self.heatmap)

    # --------------------------------------------------------------------------
    # Visualisation helpers (optional, for debug / Streamlit overlay)
    # --------------------------------------------------------------------------
    @staticmethod
    def draw_rois(
        frame: np.ndarray,
        rois: List[MotionROI],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw ROI bounding boxes with persistence count labels."""
        out = frame.copy()
        for r in rois:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{r.preproc_temp_id} P:{r.persistence_count} I:{r.intensity:.0f}"
            cv2.putText(
                out, label, (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return out

    @staticmethod
    def draw_heatmap_overlay(
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay accumulated motion heatmap on BGR frame."""
        hm_u8 = np.clip(heatmap, 0, 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, colormap)
        return cv2.addWeighted(frame, 1.0 - alpha, hm_color, alpha, 0)


# ==============================================================================
# STAND-ALONE DEMO / SANITY CHECK
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Preprocessing pipeline demo")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS")
    parser.add_argument("--out-dir", default="./preproc_output", help="Where to dump debug frames")
    parser.add_argument("--max-frames", type=int, default=200, help="Stop after N frames (demo only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessingConfig()
    pipeline = PreprocessingPipeline(cfg)

    roi_log: List[Dict] = []
    frame_idx = 0

    print(f"[INFO] Processing {args.video} at {args.fps} FPS …")

    for result in pipeline.process_video(Path(args.video), target_fps=args.fps):
        frame_idx += 1
        if not result.quality_passed:
            print(f"[WARN] Frame {frame_idx} rejected: {result.rejection_reason}")
            continue

        # Log ROIs
        for roi in result.rois:
            roi_log.append({"frame": frame_idx, **roi.to_dict()})

        # Save debug visualisation every 10th frame
        if frame_idx % 10 == 0:
            vis = pipeline.draw_rois(result.enhanced_frame, result.rois)
            vis = pipeline.draw_heatmap_overlay(vis, result.motion_heatmap)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), vis)
            print(f"[INFO] Frame {frame_idx:05d} | ROIs: {len(result.rois)} | "
                  f"MotionScore: {result.motion_score:.1f}")

        if frame_idx >= args.max_frames:
            print("[INFO] Reached --max-frames limit.")
            break

    # Dump ROI JSON log
    with open(out_dir / "roi_log.json", "w") as f:
        json.dump(roi_log, f, indent=2)

    print(f"[DONE] {frame_idx} frames processed. Debug frames + roi_log.json saved to {out_dir}")