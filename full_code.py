#!/usr/bin/env python3
"""
================================================================================
DRISHTI-PS2 — Version 2.0 (Single-File Consolidated Script)
================================================================================
Purpose: Complete offline video segmentation and ROI detection for exam forensics.
Author: DRISHTI-AI Consortium
License: DEXIT Confidential (Hackathon Use)

Yeh script ek hi file mein saare modules ko merge karti hai:
- Configuration (Pydantic + YAML + Env)
- Custom Exceptions
- Pydantic Schemas (FeatureVector, EventLog)
- Logging (JSON/Console)
- Database (File-based fallback)
- Video Ingestion (OpenCV, 2 FPS sampling)
- Preprocessing (CLAHE + MOG2)
- Feature Extractors (Pose: RTMPose, Flow: RAFT, Object: YOLOv9)
- Tracking (ByteTrack wrapper)
- Analytics (MAD Baseline, Z-Score, Decaying Confidence State Machine)
- Segmentation (Event Boundary Detection)
- Main Pipeline (Orchestrator)
- CLI Entry Point

Har module ke Hinglish comments explain karte hain ki "kyon" aur "kaise".

Usage:
    python drishti_ps2_v2.py --video path/to/video.mp4 --manifest path/to/manifest.json

Dependencies:
    pip install opencv-python-headless numpy torch onnxruntime onnx python-json-logger
================================================================================
"""

# ========================================================================
# SECTION 1: STANDARD LIBRARY IMPORTS
# ========================================================================
import os
import sys
import json
import yaml
import time
import math
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Generator, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

# ========================================================================
# SECTION 2: CONFIGURATION (Pydantic-style, but lightweight)
# ========================================================================
"""
Why: Ek central settings object jo saari thresholds, paths, aur FPS control karta hai.
     Agar humein FPS change karna hai toh code edit nahi karna, bas yahan change karo.
"""

@dataclass
class Settings:
    # Environment
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    
    # Video Processing
    FPS: int = 2                      # 2 frames per second
    RESOLUTION_WIDTH: int = 1280
    RESOLUTION_HEIGHT: int = 720
    
    # Analytics
    ZSCORE_THRESHOLD: float = 3.5     # Combined Z-score threshold
    BASELINE_WINDOW_SECONDS: int = 600 # 10 minutes rolling window
    COUPLING_MULTIPLIER: float = 1.5   # Hand-eye coupling boost
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = Path("./data")
    MODEL_DIR: Path = Path("./models")
    OUTPUT_DIR: Path = Path("./data/output")
    
    # Models (filenames)
    POSE_MODEL: str = "rtmpose.onnx"
    FLOW_MODEL: str = "raft.th"
    YOLO_MODEL: str = "yolov9.onnx"
    
    # Database (file-based for hackathon)
    DB_FILE: str = "data/features.jsonl"

settings = Settings()

# Create directories
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ========================================================================
# SECTION 3: CUSTOM EXCEPTIONS
# ========================================================================
"""
Why: Semantic exceptions help debug fast. Agar 'IngestionError' aata hai toh 
     developer ko turant pata chalega ki video file corrupt hai.
"""

class DrishtiError(Exception):
    """Base exception for all DRISHTI errors."""
    pass

class ConfigurationError(DrishtiError):
    pass

class IngestionError(DrishtiError):
    pass

class CorruptedVideoError(IngestionError):
    pass

class InferenceError(DrishtiError):
    pass

class ModelLoadError(InferenceError):
    pass

class TrackingError(DrishtiError):
    pass

class DatabaseError(DrishtiError):
    pass


# ========================================================================
# SECTION 4: LOGGING (Structured + Colored Console)
# ========================================================================
"""
Why: Production mein JSON logs machine-parseable hote hain. Dev mein colorized 
     console se debugging aasan hoti hai.
"""

def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if settings.LOG_LEVEL == "DEBUG" else logging.INFO)
    
    # Console handler with colors
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "\033[36m%(asctime)s\033[0m | \033[33m%(levelname)-8s\033[0m | \033[35m%(name)s\033[0m | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger("drishti")


# ========================================================================
# SECTION 5: SCHEMAS (Data Transfer Objects)
# ========================================================================
"""
Why: Type-safe dictionaries. Har module ko pata hai ki FeatureVector mein 
     'head_yaw' float hai, 'wrist_in_private_zone' bool hai. Isse runtime 
     errors kam hote hain.
"""

@dataclass
class FeatureVector:
    """Single frame's kinematic data for one student."""
    timestamp: datetime
    tracking_id: int
    student_id: Optional[int] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    shoulder_angle: Optional[float] = None
    wrist_left_x: Optional[float] = None
    wrist_left_y: Optional[float] = None
    wrist_right_x: Optional[float] = None
    wrist_right_y: Optional[float] = None
    wrist_in_private_zone: bool = False
    writing_velocity: Optional[float] = None
    object_class: Optional[str] = None
    object_confidence: Optional[float] = None
    motion_magnitude: Optional[float] = None
    gaze_vector_x: Optional[float] = None
    gaze_vector_y: Optional[float] = None
    confidence_score: float = 1.0
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['timestamp'] = self.timestamp.isoformat()
        return d

@dataclass
class BaselineProfile:
    """Median + MAD baseline for a student."""
    student_id: int
    last_updated: datetime
    head_yaw_mean: float = 0.0      # Actually median
    head_yaw_std: float = 1.0       # Actually MAD
    wrist_zone_mean: float = 0.0
    wrist_zone_std: float = 1.0
    writing_velocity_mean: float = 0.0
    writing_velocity_std: float = 1.0
    motion_magnitude_mean: float = 0.0
    motion_magnitude_std: float = 1.0
    sample_count: int = 0

@dataclass
class EventLog:
    """Fired when anomaly sequence detected."""
    student_id: int
    trigger_sequence: str
    peak_anomaly_score: float
    head_z_score: float
    hand_z_score: float
    velocity_z_score: float
    start_timestamp: datetime
    end_timestamp: datetime
    detected_object: Optional[str] = None
    object_confidence: Optional[float] = None
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['start_timestamp'] = self.start_timestamp.isoformat()
        d['end_timestamp'] = self.end_timestamp.isoformat()
        return d


# ========================================================================
# SECTION 6: DATABASE (File-based JSONL for Hackathon)
# ========================================================================
"""
Why: TimescaleDB production mein use hota hai, but hackathon ke liye file-based 
     JSONL (JSON Lines) store kaafi hai. Har feature vector ek line mein store.
"""

def write_feature_to_db(feature: FeatureVector):
    """Append feature vector to JSONL file."""
    try:
        with open(settings.DB_FILE, 'a') as f:
            f.write(json.dumps(feature.to_dict()) + '\n')
    except Exception as e:
        logger.error(f"Failed to write feature: {e}")

def write_event_to_db(event: EventLog):
    """Append event to JSONL file."""
    try:
        with open(settings.DATA_DIR / 'events.jsonl', 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    except Exception as e:
        logger.error(f"Failed to write event: {e}")


# ========================================================================
# SECTION 7: INGESTION — Video Loader (2 FPS Sampler)
# ========================================================================
"""
Why: Raw video 24-30 FPS hoti hai. Hum sirf 2 FPS sample karte hain kyunki:
     1. GPU load kam hota hai.
     2. "Slow creep" (gradual rotation) capture ho jati hai.
     3. 6-hour video ~43,000 frames mein convert hoti hai.
"""

def validate_video(video_path: Path) -> bool:
    """Check if video is readable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def sample_frames(video_path: Path, target_fps: int = settings.FPS) -> Generator[Tuple[datetime, np.ndarray], None, None]:
    """
    Yield frames at constant time intervals (2 FPS) with accurate timestamps.
    VFR (Variable Frame Rate) ko handle karne ke liye CAP_PROP_POS_MSEC use karte hain.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IngestionError(f"Cannot open video: {video_path}")
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 24.0
    step = max(1, int(round(orig_fps / target_fps)))
    time_increment = 1.0 / target_fps
    
    frame_idx = 0
    elapsed = 0.0
    consecutive_failures = 0
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get actual timestamp from PTS
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec < 0:
            msec = elapsed * 1000.0
        timestamp = datetime.fromtimestamp(msec / 1000.0)
        
        # Resize to target resolution
        if frame is not None:
            frame = cv2.resize(frame, (settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT))
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures > 10:
                raise CorruptedVideoError(f"Too many failed frames at index {frame_idx}")
        
        yield (timestamp, frame)
        
        frame_idx += step
        elapsed += time_increment
    
    cap.release()
    logger.info(f"Finished sampling video: {video_path.name}")


# ========================================================================
# SECTION 8: PREPROCESSOR (CLAHE + MOG2 Background Subtraction)
# ========================================================================
"""
Why: MOG2 se static background (walls, desks) remove ho jata hai, sirf students 
     remain karte hain. CLAHE se low-light areas (under desks) visible ho jaate hain.
"""

class VideoPreprocessor:
    def __init__(self, learning_rate: float = 0.01):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )
        self.learning_rate = learning_rate
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        logger.info("Preprocessor initialized.")
    
    def process(self, frame: np.ndarray, apply_bg_mask: bool = True) -> np.ndarray:
        if frame is None:
            return None
        # Grayscale for CLAHE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(enhanced, learningRate=self.learning_rate)
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        if apply_bg_mask:
            _, mask_bin = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            return cv2.bitwise_and(enhanced, enhanced, mask=mask_bin)
        return enhanced


# ========================================================================
# SECTION 9: FEATURE EXTRACTORS (Placeholder — actual models load lazily)
# ========================================================================
"""
Why: Hackathon mein actual ONNX models load karna heavy hai. Isliye hum placeholder
     extractors use karte hain jo synthetic features generate karte hain.
     Real deployment mein inhe actual model inference se replace karna hai.
"""

class PoseExtractor:
    """Placeholder for RTMPose."""
    def __init__(self, model_path):
        self.model_path = model_path
        logger.info(f"PoseExtractor initialized (placeholder) with {model_path}")
    
    def extract(self, frame):
        # Simulate keypoints: random positions with confidence
        h, w = frame.shape[:2]
        # Return dummy keypoints for 1 person
        kp = np.random.rand(17, 3) * np.array([w, h, 1]) + np.array([0, 0, 0.5])
        kp[:, 2] = np.clip(kp[:, 2], 0.4, 1.0)  # confidence
        return {'keypoints': kp[np.newaxis, ...], 'num_people': 1}

class FlowExtractor:
    """Placeholder for RAFT."""
    def __init__(self, model_path):
        self.model_path = model_path
        logger.info(f"FlowExtractor initialized (placeholder) with {model_path}")
    
    def extract(self, frame, prev_frame=None):
        if prev_frame is None:
            return {'mean_magnitude': 0.0, 'flow_valid': False}
        return {'mean_magnitude': np.random.rand() * 10, 'flow_valid': True}

class ObjectExtractor:
    """Placeholder for YOLOv9."""
    def __init__(self, model_path):
        self.model_path = model_path
        logger.info(f"ObjectExtractor initialized (placeholder) with {model_path}")
    
    def extract(self, frame):
        # Simulate no detections
        return {'detections': [], 'num_detections': 0}


# ========================================================================
# SECTION 10: TRACKING — ByteTrack Wrapper (Placeholder)
# ========================================================================
"""
Why: ByteTrack IoU-based tracker. Identity swap ko avoid karne ke liye post-hoc 
     corrector bhi hai (placeholder).
"""

class ByteTracker:
    def __init__(self):
        self.frame_id = 0
        self.tracks = {}  # track_id -> bbox
        self.next_id = 1
        logger.info("ByteTracker initialized (placeholder).")
    
    def update(self, detections):
        self.frame_id += 1
        results = []
        for det in detections:
            # Assign new track ID or reuse existing (simplified)
            # For hackathon, we just assign a new ID per detection
            track_id = self.next_id
            self.next_id += 1
            results.append({
                'track_id': track_id,
                'bbox': det['bbox'],
                'score': det.get('score', 0.9)
            })
        return results


# ========================================================================
# SECTION 11: ANALYTICS — Baseline (MAD), Z-Score, Decaying Confidence
# ========================================================================

class BaselineEngine:
    """
    Rolling Median + MAD using circular buffer.
    Robust to heavy-tailed human motion.
    """
    def __init__(self, window_seconds: int = 600, buffer_size: int = 100):
        self.window_seconds = window_seconds
        self.buffer_size = buffer_size
        self.buffers = defaultdict(lambda: {
            'head_yaw': deque(maxlen=buffer_size),
            'wrist_zone': deque(maxlen=buffer_size),
            'writing_velocity': deque(maxlen=buffer_size),
            'motion_magnitude': deque(maxlen=buffer_size),
        })
        self.timestamps = defaultdict(deque)
        logger.info(f"BaselineEngine initialized with MAD, buffer={buffer_size}")
    
    def update(self, student_id: int, features: Dict[str, Any]):
        timestamp = features.get('timestamp', datetime.now())
        # Expire old data
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        while self.timestamps[student_id] and self.timestamps[student_id][0] < cutoff:
            self.timestamps[student_id].popleft()
        # Append new
        for feat in ['head_yaw', 'wrist_zone', 'writing_velocity', 'motion_magnitude']:
            val = features.get(feat)
            if val is not None:
                self.buffers[student_id][feat].append(val)
        self.timestamps[student_id].append(timestamp)
        while len(self.timestamps[student_id]) > self.buffer_size:
            self.timestamps[student_id].popleft()
    
    def get_baseline(self, student_id: int) -> Optional[BaselineProfile]:
        if student_id not in self.buffers:
            return None
        buffers = self.buffers[student_id]
        min_samples = 10
        def get_median_mad(buf):
            if len(buf) < min_samples:
                return None, None
            arr = np.array(buf)
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            return median, max(mad, 0.01)
        
        h_med, h_mad = get_median_mad(buffers['head_yaw'])
        if h_med is None:
            return None
        z_med, z_mad = get_median_mad(buffers['wrist_zone'])
        v_med, v_mad = get_median_mad(buffers['writing_velocity'])
        m_med, m_mad = get_median_mad(buffers['motion_magnitude'])
        
        return BaselineProfile(
            student_id=student_id,
            last_updated=datetime.now(),
            head_yaw_mean=h_med, head_yaw_std=h_mad,
            wrist_zone_mean=z_med or 0.0, wrist_zone_std=z_mad or 1.0,
            writing_velocity_mean=v_med or 0.0, writing_velocity_std=v_mad or 1.0,
            motion_magnitude_mean=m_med or 0.0, motion_magnitude_std=m_mad or 1.0,
            sample_count=len(buffers['head_yaw'])
        )
    
    def reset(self, student_id: int):
        if student_id in self.buffers:
            del self.buffers[student_id]
        if student_id in self.timestamps:
            del self.timestamps[student_id]


class ZScoreEngine:
    """
    Computes robust Z-scores using Median and MAD.
    Handles left-handed normalization.
    """
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
    
    def compute_scores(self, features: Dict[str, Any], baseline: BaselineProfile,
                       handedness: str = 'right') -> Dict[str, float]:
        head_val = features.get('head_yaw', 0.0)
        head_z = abs((head_val - baseline.head_yaw_mean) / baseline.head_yaw_std)
        
        zone_val = 1.0 if features.get('wrist_in_private_zone', False) else 0.0
        hand_z = max(0.0, (zone_val - baseline.wrist_zone_mean) / baseline.wrist_zone_std)
        
        vel_val = features.get('writing_velocity', 0.0)
        velocity_z = max(0.0, (vel_val - baseline.writing_velocity_mean) / baseline.writing_velocity_std)
        
        motion_val = features.get('motion_magnitude', 0.0)
        motion_z = max(0.0, (motion_val - baseline.motion_magnitude_mean) / baseline.motion_magnitude_std)
        
        # Left-handed: reduce torso weight
        shoulder_weight = 0.5 if handedness == 'left' else 1.0
        
        combined_z = (0.3 * head_z) + (0.4 * hand_z) + (0.2 * velocity_z) + (0.1 * motion_z * shoulder_weight)
        return {'head_z': head_z, 'hand_z': hand_z, 'velocity_z': velocity_z,
                'motion_z': motion_z, 'combined_z': combined_z}
    
    def apply_coupling(self, gaze_vec, wrist_pos, z_scores, is_private):
        """Apply hand-eye coupling multiplier."""
        coupled = z_scores['combined_z']
        mult = 1.0
        if is_private and z_scores['head_z'] > 2.0 and z_scores['hand_z'] > 2.0:
            mult = settings.COUPLING_MULTIPLIER
        z_scores['coupling_multiplier'] = mult
        z_scores['coupled_score'] = coupled * mult
        return z_scores


class DecayingConfidenceEngine:
    """
    Replaces rigid FSM with decaying confidence score.
    Captures variable timing and reduces false positives.
    """
    def __init__(self, decay_rate: float = 0.05, threshold: float = 5.0):
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.scores = defaultdict(float)
        self.last_update = defaultdict(datetime.now)
        logger.info(f"DecayingConfidenceEngine: decay={decay_rate}, thresh={threshold}")
    
    def process(self, student_id: int, head_z: float, hand_z: float, velocity_z: float,
                is_private: bool, timestamp: datetime) -> Optional[EventLog]:
        dt = (timestamp - self.last_update[student_id]).total_seconds()
        if dt > 0:
            self.scores[student_id] *= np.exp(-self.decay_rate * dt)
        
        # Contributions
        if head_z > 3.0:
            self.scores[student_id] += 0.5 * head_z
        if is_private and hand_z > 2.0:
            self.scores[student_id] += 0.8 * hand_z
        if velocity_z > 2.5:
            self.scores[student_id] += 0.4 * velocity_z
        
        self.scores[student_id] = min(self.scores[student_id], 20.0)
        self.last_update[student_id] = timestamp
        
        if self.scores[student_id] > self.threshold:
            event = EventLog(
                student_id=student_id,
                trigger_sequence="DECAYING_CONFIDENCE",
                peak_anomaly_score=self.scores[student_id],
                head_z_score=head_z,
                hand_z_score=hand_z,
                velocity_z_score=velocity_z,
                start_timestamp=timestamp - timedelta(seconds=30),
                end_timestamp=timestamp
            )
            self.scores[student_id] = 0.0
            return event
        return None


# ========================================================================
# SECTION 12: SEGMENTATION — Event Boundary Detection
# ========================================================================

class EventBoundaryDetector:
    def __init__(self, pre_buffer: int = 15, post_buffer: int = 10):
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
    
    def detect_boundaries(self, event: EventLog, history: List[Dict]) -> Tuple[datetime, datetime]:
        trigger = event.start_timestamp
        start = trigger - timedelta(seconds=self.pre_buffer)
        end = trigger + timedelta(seconds=self.post_buffer)
        # If history has private zone entries, refine start
        if history:
            earliest_private = trigger
            for f in history:
                ts = f.get('timestamp')
                if ts and f.get('wrist_in_private_zone', False) and ts < earliest_private:
                    earliest_private = ts
            if earliest_private < trigger:
                start = earliest_private - timedelta(seconds=2)
        return start, end


# ========================================================================
# SECTION 13: MAIN PROCESSING PIPELINE (The Orchestrator)
# ========================================================================

class ProcessingPipeline:
    def __init__(self):
        self.preprocessor = VideoPreprocessor()
        # Lazy-load extractors (placeholder)
        self.pose_extractor = PoseExtractor(str(settings.MODEL_DIR / settings.POSE_MODEL))
        self.flow_extractor = FlowExtractor(str(settings.MODEL_DIR / settings.FLOW_MODEL))
        self.object_extractor = ObjectExtractor(str(settings.MODEL_DIR / settings.YOLO_MODEL))
        self.tracker = ByteTracker()
        self.baseline_engine = BaselineEngine()
        self.zscore_engine = ZScoreEngine()
        self.state_machine = DecayingConfidenceEngine()
        self.boundary_detector = EventBoundaryDetector()
        
        # State
        self.prev_frame = None
        self.prev_flow_frame = None
        self.student_tracks = {}
        self.events = []
        self.history_buffer = defaultdict(list)
        self.prev_wrist_positions = {}
        self.writing_velocity_cache = {}
    
    def process_video(self, video_path: Path, manifest: Dict[str, Any] = None) -> List[EventLog]:
        manifest = manifest or {}
        calibration_start = manifest.get('calibration_start', 600)
        calibration_end = manifest.get('calibration_end', 1200)
        handedness_map = manifest.get('handedness', {})
        self.student_tracks = manifest.get('student_map', {})
        
        self.events = []
        self.history_buffer.clear()
        self.prev_frame = None
        self.prev_flow_frame = None
        self.prev_wrist_positions.clear()
        self.writing_velocity_cache.clear()
        self.baseline_engine = BaselineEngine()
        self.state_machine = DecayingConfidenceEngine()
        
        frame_count = 0
        start_time = datetime.now()
        logger.info(f"🚀 Processing {video_path.name} with calibration {calibration_start}-{calibration_end}s")
        
        for timestamp, frame in sample_frames(video_path):
            frame_count += 1
            elapsed = (timestamp - start_time).total_seconds()
            
            # Preprocess
            processed = self.preprocessor.process(frame, apply_bg_mask=True)
            
            # Extract features (placeholder)
            pose_data = self.pose_extractor.extract(processed)
            flow_data = self.flow_extractor.extract(processed, self.prev_flow_frame)
            object_data = self.object_extractor.extract(processed)
            
            # Build detections from pose
            detections = self._build_detections(pose_data)
            tracks = self.tracker.update(detections)
            
            for track in tracks:
                track_id = track['track_id']
                student_id = self.student_tracks.get(track_id, track_id)
                handedness = handedness_map.get(str(track_id), 'right')
                
                # Extract per-student features
                features = self._extract_features(track, pose_data, flow_data, object_data, timestamp)
                if features is None:
                    continue
                
                # Update baseline only during calibration window
                if calibration_start <= elapsed <= calibration_end:
                    self.baseline_engine.update(student_id, features)
                
                baseline = self.baseline_engine.get_baseline(student_id)
                if baseline is not None:
                    z_scores = self.zscore_engine.compute_scores(features, baseline, handedness)
                    z_scores = self.zscore_engine.apply_coupling(
                        (features.get('gaze_vector_x'), features.get('gaze_vector_y')),
                        (features.get('wrist_left_x'), features.get('wrist_left_y')),
                        z_scores,
                        features.get('wrist_in_private_zone', False)
                    )
                    event = self.state_machine.process(
                        student_id,
                        z_scores['head_z'], z_scores['hand_z'], z_scores['velocity_z'],
                        features.get('wrist_in_private_zone', False),
                        timestamp
                    )
                    if event:
                        if object_data['detections']:
                            event.detected_object = object_data['detections'][0]['class']
                            event.object_confidence = object_data['detections'][0]['confidence']
                        # Refine boundaries
                        hist = [f for _, f in self.history_buffer[student_id]]
                        start, end = self.boundary_detector.detect_boundaries(event, hist)
                        event.start_timestamp = start
                        event.end_timestamp = end
                        self.events.append(event)
                        write_event_to_db(event)
                        logger.warning(f"🚨 EVENT: Student {student_id} at {timestamp}")
                
                # Store in history
                self.history_buffer[student_id].append((timestamp, features))
                # Trim history
                cutoff = timestamp - timedelta(seconds=settings.BASELINE_WINDOW_SECONDS + 30)
                self.history_buffer[student_id] = [(t, f) for t, f in self.history_buffer[student_id] if t > cutoff]
                
                # Write feature to DB
                write_feature_to_db(features)
            
            self.prev_frame = processed
            self.prev_flow_frame = processed
            
            if frame_count % 500 == 0:
                logger.info(f"Progress: {frame_count} frames, Events: {len(self.events)}")
        
        logger.info(f"✅ Done. Total frames: {frame_count}, Events: {len(self.events)}")
        return self.events
    
    def _build_detections(self, pose_data):
        """Convert pose keypoints to detections."""
        dets = []
        kps = pose_data.get('keypoints', np.array([]))
        for i in range(pose_data.get('num_people', 0)):
            kp = kps[i]
            valid = kp[:, 2] > 0.5
            if not any(valid):
                continue
            xs = kp[valid, 0]
            ys = kp[valid, 1]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(settings.RESOLUTION_WIDTH, x2 + margin)
            y2 = min(settings.RESOLUTION_HEIGHT, y2 + margin)
            dets.append({'bbox': (x1, y1, x2, y2), 'score': 0.9})
        return dets
    
    def _extract_features(self, track, pose_data, flow_data, object_data, timestamp):
        """Extract per-student feature vector from pose data."""
        kps = pose_data.get('keypoints', np.array([]))
        if kps.size == 0:
            return None
        # Use first person
        kp = kps[0]
        h, w = settings.RESOLUTION_HEIGHT, settings.RESOLUTION_WIDTH
        
        # Head yaw (simplified)
        head_yaw = 0.0
        if kp[0][2] > 0.3:  # nose
            head_yaw = (kp[0][0] - w/2) / w * 90
        
        # Wrist positions
        wrist_lx = kp[9][0]/w if kp[9][2] > 0.3 else None
        wrist_ly = kp[9][1]/h if kp[9][2] > 0.3 else None
        wrist_rx = kp[10][0]/w if kp[10][2] > 0.3 else None
        wrist_ry = kp[10][1]/h if kp[10][2] > 0.3 else None
        
        # Private zone: hip line ~0.6 of height
        hip_y = 0.6
        wrist_private = False
        if wrist_ly is not None and wrist_ly > hip_y:
            wrist_private = True
        if wrist_ry is not None and wrist_ry > hip_y:
            wrist_private = True
        
        # Writing velocity (simplified)
        velocity = 0.0
        track_id = track['track_id']
        prev = self.prev_wrist_positions.get(track_id)
        if prev is not None:
            px, py, pts = prev
            cx = wrist_rx or wrist_lx or 0.0
            cy = wrist_ry or wrist_ly or 0.0
            dt = (timestamp - pts).total_seconds()
            if dt > 0:
                dist = np.sqrt(((cx - px)*w)**2 + ((cy - py)*h)**2)
                velocity = dist / dt
                # Smooth
                if track_id in self.writing_velocity_cache:
                    velocity = 0.7 * self.writing_velocity_cache[track_id] + 0.3 * velocity
                self.writing_velocity_cache[track_id] = velocity
        self.prev_wrist_positions[track_id] = (wrist_rx or wrist_lx or 0.0, wrist_ry or wrist_ly or 0.0, timestamp)
        
        return {
            'timestamp': timestamp,
            'tracking_id': track['track_id'],
            'student_id': None,
            'head_yaw': head_yaw,
            'head_pitch': None,
            'shoulder_angle': 0.0,
            'wrist_left_x': wrist_lx,
            'wrist_left_y': wrist_ly,
            'wrist_right_x': wrist_rx,
            'wrist_right_y': wrist_ry,
            'wrist_in_private_zone': wrist_private,
            'writing_velocity': velocity,
            'object_class': None,
            'object_confidence': None,
            'motion_magnitude': flow_data.get('mean_magnitude', 0.0),
            'gaze_vector_x': 0.0,
            'gaze_vector_y': 0.0,
            'confidence_score': 1.0
        }


# ========================================================================
# SECTION 14: CLI ENTRY POINT
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="DRISHTI-PS2 V2 — Offline Exam Forensics")
    parser.add_argument("--video", required=True, help="Path to video file (.mp4)")
    parser.add_argument("--manifest", help="Path to manifest JSON (optional)")
    parser.add_argument("--output", default="./data/output/", help="Output directory")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    manifest = {}
    if args.manifest and Path(args.manifest).exists():
        with open(args.manifest) as f:
            manifest = json.load(f)
    
    # Run pipeline
    pipeline = ProcessingPipeline()
    events = pipeline.process_video(video_path, manifest)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DRISHTI-PS2 V2 — Processing Summary")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Events Detected: {len(events)}")
    for i, ev in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(f"  Student: {ev.student_id}")
        print(f"  Score: {ev.peak_anomaly_score:.2f}")
        print(f"  Sequence: {ev.trigger_sequence}")
        print(f"  Time: {ev.start_timestamp.strftime('%H:%M:%S')} - {ev.end_timestamp.strftime('%H:%M:%S')}")
        if ev.detected_object:
            print(f"  Object: {ev.detected_object} (conf: {ev.object_confidence:.2f})")
    print("\n✅ Output saved to:", settings.OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()