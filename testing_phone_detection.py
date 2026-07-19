# ==============================================================================
# Standalone YOLO Phone Detection Test — Kaggle Notebook Version (single GPU)
# ==============================================================================
# Purpose: confirm whether a pretrained YOLO model can detect a mobile phone
# in your test footage (lighting, angle, distance) — independent of the
# ROI/motion pipeline. Raw capability check only, not an integration test.
#
# SETUP (run these once at the top of your Kaggle notebook):
#   1. Notebook Settings (right sidebar) -> Accelerator -> GPU (T4 x2 or P100)
#      (this script only uses ONE gpu, but either accelerator option is fine)
#   2. !pip install ultralytics
#   3. Upload your test video via "Add Data" or place it in /kaggle/working/
# ==============================================================================

get_ipython().system('pip install -q ultralytics')

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# 1. Confirm GPU is actually available and being used
# ------------------------------------------------------------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
    DEVICE = 0          # first GPU only (single-GPU by design, see discussion)
else:
    print("WARNING: No GPU detected — falling back to CPU, will be much slower.")
    DEVICE = "cpu"


# ------------------------------------------------------------------------------
# 2. Core detection function
# ------------------------------------------------------------------------------
def test_phone_detection(
    video_path: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    sample_fps: float = 2.0,
    save_frames: bool = True,
    output_dir: str = "/kaggle/working/yolo_test_output",
):
    """
    Run standalone phone detection on a video, sampled at `sample_fps`.
    Prints every detection with timestamp + confidence + bbox, and optionally
    saves annotated frames for visual inspection.
    """
    print(f"Loading model: {model_name}  (device={DEVICE})")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(orig_fps / sample_fps)))

    out_dir = Path(output_dir)
    if save_frames:
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    checked_count = 0
    phone_detections = 0
    detection_log = []  # collected for a returned summary (handy in a notebook)

    print(f"Sampling every {step} frames (~{sample_fps} FPS) at conf>={conf_threshold}")
    print("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            checked_count += 1
            results = model.predict(frame, verbose=False, conf=conf_threshold, device=DEVICE)

            found_phone_this_frame = False
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    label = model.names.get(cls_id, str(cls_id))
                    conf = float(box.conf.item())

                    if label == "cell phone":
                        found_phone_this_frame = True
                        phone_detections += 1
                        t_sec = frame_idx / orig_fps
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        entry = {
                            "time_sec": round(t_sec, 2),
                            "confidence": round(conf, 3),
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        }
                        detection_log.append(entry)
                        print(f"  t={t_sec:6.2f}s  PHONE detected  conf={conf:.2f}  "
                              f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

                        if save_frames:
                            annotated = r.plot()
                            out_path = out_dir / f"phone_t{t_sec:.2f}s_conf{conf:.2f}.jpg"
                            cv2.imwrite(str(out_path), annotated)

            if not found_phone_this_frame and checked_count % 20 == 0:
                t_sec = frame_idx / orig_fps
                print(f"  ... checked up to t={t_sec:.1f}s, no phone yet")

        frame_idx += 1

    cap.release()

    print("-" * 70)
    print(f"Frames checked: {checked_count}")
    print(f"Total phone detections: {phone_detections}")
    if phone_detections == 0:
        print("WARNING: No phone detected anywhere in this video at this confidence "
              "threshold. Try lowering conf_threshold, or check the phone is "
              "actually visible clearly enough (lighting/angle/distance/occlusion).")
    if save_frames:
        print(f"Annotated frames saved to: {out_dir.resolve()}")

    return {
        "frames_checked": checked_count,
        "total_detections": phone_detections,
        "detections": detection_log,
    }


# ------------------------------------------------------------------------------
# 3. Run it — EDIT THIS PATH to point at your uploaded video
# ------------------------------------------------------------------------------
# Example (uncomment and edit):
#
# results = test_phone_detection(
#     video_path="/kaggle/input/your-dataset-name/cheating_test.mp4",
#     model_name="yolov8n.pt",     # lightest/fastest; try "yolov8s.pt" for more accuracy
#     conf_threshold=0.35,
#     sample_fps=2.0,
#     save_frames=True,
# )
#
# print(results["total_detections"], "phones found across", results["frames_checked"], "frames")
