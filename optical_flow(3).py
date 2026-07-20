"""
================================================================================
DRISHTI-PS2 — Optical Flow Refinement Module
================================================================================
Sits downstream of preprocessing.py (MOG2 + ROI extraction). Consumes each
PreprocessedFrame and adds two things MOG2 alone cannot provide:

  1. Real motion intensity/direction per ROI (dense Farneback flow), replacing
     the flat "moved or didn't" signal from the binary fg_mask with a graded
     magnitude — used to enrich ROI.intensity and the heatmap.
  2. Camera-vibration rejection — distinguishing frame-wide uniform motion
     (camera shake) from spatially-concentrated real activity, per the PS's
     own "Camera Vibration" risk factor, which MOG2 cannot tell apart on its
     own (a vibrating camera produces foreground blobs just like a moving
     person would).

Design constraint: OPTIMISED, not exhaustive.
  - Dense flow (Farneback) is only computed on the minimal region needed:
    the union of all ROI boxes (padded) for the "local" signal, plus a cheap
    coarse full-frame pass (downscaled) purely to get a global vibration
    baseline — not full-resolution flow on the whole frame.
  - Flow is skipped entirely on frames with zero ROIs (nothing to refine).
  - This does NOT replace MOG2 and does NOT do its own background
    subtraction — it only refines ROIs that preprocessing.py already found.

Usage:
    from preprocessing import PreprocessingPipeline, PreprocessingConfig
    from optical_flow import OpticalFlowRefiner, OpticalFlowConfig

    pre_pipe = PreprocessingPipeline(PreprocessingConfig())
    flow_refiner = OpticalFlowRefiner(OpticalFlowConfig())

    prev_gray = None
    for result in pre_pipe.process_video(Path("exam.mp4")):
        refined = flow_refiner.refine(result, prev_gray)
        prev_gray = flow_refiner.last_gray
        # refined.rois            -> same ROIs, intensity enriched with flow magnitude
        # refined.vibration_flagged -> bool, True if this frame looks like camera shake
        # refined.flow_heatmap_delta -> per-pixel magnitude update, ready to add to heatmap
================================================================================
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from preprocessing import PreprocessedFrame, MotionROI


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class OpticalFlowConfig:
    """Tunable knobs for the optical flow refinement stage."""

    # --- Farneback parameters (local, ROI-gated pass) -------------------------
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 2          # kept low for speed (optimised, not max accuracy)
    farneback_winsize: int = 15
    farneback_iterations: int = 2      # kept low for speed
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.1

    # --- ROI gating ------------------------------------------------------------
    roi_padding_px: int = 15          # extra margin around each ROI when cropping
                                       # for flow, so motion isn't clipped at the box edge

    # --- Global vibration check (cheap, downscaled full-frame pass) -----------
    vibration_check_downscale: float = 0.25   # run coarse flow at 1/4 resolution
    vibration_uniformity_threshold: float = 0.6
    # if this fraction (or more) of the *coarse* frame shows motion roughly
    # equal in magnitude/direction, treat it as camera shake rather than
    # localised human activity
    vibration_min_magnitude: float = 0.3      # ignore near-zero flow as "no vibration"

    # --- Intensity blending -----------------------------------------------------
    flow_intensity_weight: float = 0.6        # how much flow magnitude contributes
                                               # vs. the original MOG2-based intensity
                                               # (1.0 = fully replace, 0.0 = ignore flow)


# ==============================================================================
# OUTPUT OBJECT
# ==============================================================================

@dataclass
class FlowRefinedFrame:
    """PreprocessedFrame enriched with optical-flow information."""
    source: PreprocessedFrame
    rois: List[MotionROI] = field(default_factory=list)   # same ROIs, intensity updated
    vibration_flagged: bool = False
    vibration_score: float = 0.0             # 0-1, higher = more uniform/frame-wide
    flow_heatmap_delta: Optional[np.ndarray] = None  # per-pixel magnitude, add to heatmap
    flow_computed: bool = False              # False if skipped (no ROIs / first frame)


# ==============================================================================
# OPTICAL FLOW REFINER
# ==============================================================================

class OpticalFlowRefiner:
    """
    Motion-gated dense optical flow refinement on top of MOG2 + ROI output.

    Does NOT run every pixel of every frame through Farneback — only:
      (a) a small, padded crop around the union of current ROIs (local signal)
      (b) a heavily downscaled full-frame pass (global vibration baseline)
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None):
        self.cfg = config or OpticalFlowConfig()
        self.last_gray: Optional[np.ndarray] = None  # full-res gray, for next call's flow

    def refine(
        self,
        frame: PreprocessedFrame,
        prev_gray: Optional[np.ndarray] = None,
    ) -> FlowRefinedFrame:
        """
        Refine one PreprocessedFrame using optical flow.

        `prev_gray` should be the `.last_gray` returned from the previous call
        (or None on the very first frame of a video).
        """
        gray = cv2.cvtColor(frame.enhanced_frame, cv2.COLOR_BGR2GRAY)

        # Nothing to refine: no ROIs, or this is the first frame (no prev to diff against)
        if not frame.quality_passed or not frame.rois or prev_gray is None:
            self.last_gray = gray
            return FlowRefinedFrame(
                source=frame,
                rois=frame.rois,
                vibration_flagged=False,
                vibration_score=0.0,
                flow_heatmap_delta=None,
                flow_computed=False,
            )

        # ------------------------------------------------------------------
        # 1. Cheap global vibration check (downscaled full-frame flow)
        # ------------------------------------------------------------------
        vibration_score, is_vibrating = self._check_vibration(prev_gray, gray)

        # ------------------------------------------------------------------
        # 2. Local flow, gated to the union of ROI regions only
        # ------------------------------------------------------------------
        refined_rois, flow_delta = self._refine_rois(frame.rois, prev_gray, gray)

        # If the frame looks like camera vibration, suppress the flow-based
        # intensity boost (fall back to original MOG2 intensity) rather than
        # letting vibration inflate scores — this is the actual vibration
        # rejection behaviour, applied at the point of use.
        if is_vibrating:
            refined_rois = frame.rois  # discard flow enrichment for this frame
            flow_delta = None

        self.last_gray = gray

        return FlowRefinedFrame(
            source=frame,
            rois=refined_rois,
            vibration_flagged=is_vibrating,
            vibration_score=vibration_score,
            flow_heatmap_delta=flow_delta,
            flow_computed=True,
        )

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _check_vibration(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Cheap, heavily-downscaled full-frame flow pass, used only to check
        whether motion is spread uniformly across the whole frame (vibration)
        rather than concentrated in specific regions (real activity).
        """
        scale = self.cfg.vibration_check_downscale
        small_prev = cv2.resize(prev_gray, None, fx=scale, fy=scale)
        small_cur = cv2.resize(gray, None, fx=scale, fy=scale)

        flow = cv2.calcOpticalFlowFarneback( 
            small_prev, small_cur, None,
            pyr_scale=self.cfg.farneback_pyr_scale,
            levels=1,                              # coarse pass — 1 level is enough
            winsize=self.cfg.farneback_winsize,
            iterations=1,                           # coarse pass — 1 iteration is enough
            poly_n=self.cfg.farneback_poly_n,
            poly_sigma=self.cfg.farneback_poly_sigma,
            flags=0,
        ) #type:ignore
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        moving_mask = magnitude > self.cfg.vibration_min_magnitude
        moving_fraction = float(np.mean(moving_mask))

        # A vibrating camera moves *almost every* pixel by a similar small
        # amount; real localised activity only moves a small fraction of the
        # frame. Use the fraction of "moving" pixels as the uniformity proxy.
        vibration_score = moving_fraction
        is_vibrating = moving_fraction >= self.cfg.vibration_uniformity_threshold

        return vibration_score, is_vibrating

    def _refine_rois(
        self,
        rois: List[MotionROI],
        prev_gray: np.ndarray,
        gray: np.ndarray,
    ) -> Tuple[List[MotionROI], Optional[np.ndarray]]:
        """
        Compute Farneback flow only inside the padded union of ROI boxes,
        then update each ROI's intensity with real flow magnitude and build
        a sparse per-pixel delta to merge into the heatmap.
        """
        h, w = gray.shape[:2]
        pad = self.cfg.roi_padding_px

        # Union bounding box of all ROIs (+padding), clipped to frame bounds —
        # this keeps the flow computation to the smallest rectangle that
        # covers everything currently flagged, instead of the full frame.
        x1 = max(0, min(r.x for r in rois) - pad)
        y1 = max(0, min(r.y for r in rois) - pad)
        x2 = min(w, max(r.x + r.w for r in rois) + pad)
        y2 = min(h, max(r.y + r.h for r in rois) + pad)

        if x2 <= x1 or y2 <= y1:
            return rois, None

        prev_crop = prev_gray[y1:y2, x1:x2]
        cur_crop = gray[y1:y2, x1:x2]

        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, cur_crop, None,
            pyr_scale=self.cfg.farneback_pyr_scale,
            levels=self.cfg.farneback_levels,
            winsize=self.cfg.farneback_winsize,
            iterations=self.cfg.farneback_iterations,
            poly_n=self.cfg.farneback_poly_n,
            poly_sigma=self.cfg.farneback_poly_sigma,
            flags=0,
        ) #type:ignore
        magnitude_crop = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Build a full-frame-sized delta with zeros outside the cropped region,
        # so it can be added directly onto the existing heatmap array.
        flow_delta = np.zeros((h, w), dtype=np.float32)
        flow_delta[y1:y2, x1:x2] = magnitude_crop

        # Update each ROI's intensity using the mean flow magnitude within its
        # own box (offset into the crop's coordinate space).
        refined: List[MotionROI] = []
        for r in rois:
            rx1, ry1 = r.x - x1, r.y - y1
            rx2, ry2 = rx1 + r.w, ry1 + r.h
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2 = min(magnitude_crop.shape[1], rx2)
            ry2 = min(magnitude_crop.shape[0], ry2)

            if rx2 > rx1 and ry2 > ry1:
                flow_mag = float(np.mean(magnitude_crop[ry1:ry2, rx1:rx2]))
                # Blend original MOG2-based intensity with flow magnitude,
                # scaled to a comparable 0-255-ish range for consistency with
                # the existing intensity field.
                flow_intensity_scaled = min(flow_mag * 40.0, 255.0)
                w_flow = self.cfg.flow_intensity_weight
                r.intensity = (1 - w_flow) * r.intensity + w_flow * flow_intensity_scaled

            refined.append(r)

        return refined, flow_delta