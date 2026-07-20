from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ---- Environment ----
    LOG_LEVEL: str = "INFO"

    # ---- Video Ingestion (Objective 1) ----
    TARGET_FPS: int = 2                 # process N frames/sec, not every frame
    RESOLUTION_WIDTH: int = 1280
    RESOLUTION_HEIGHT: int = 720
    CHUNK_SIZE_FRAMES: int = 500        # frames per chunk, keeps memory bounded

    # ---- Motion Detection (Objective 2) -- filled in next ticket ----
    MOG2_HISTORY: int = 500
    MOG2_VAR_THRESHOLD: float = 36.0
    MOG2_DETECT_SHADOWS: bool = True
    GAUSSIAN_BLUR_KERNEL: int = 5       # noise/compression artifact removal

    # ---- ROI Extraction (Objective 3) -- filled in later ticket ----
    ROI_MIN_AREA: int = 800             # pixels; filters tiny noise blobs
    ROI_PERSISTENCE_FRAMES: int = 3     # must persist this many frames to count
    ROI_MERGE_DISTANCE: int = 40        # px distance to merge nearby blobs

    # ---- Event Segmentation (Objective 5) -- filled in later ticket ----
    # NOTE: thresholds are against MotionResult.active_pixel_fraction
    # (fraction of frame pixels showing meaningful motion), NOT the raw
    # whole-frame mean -- the mean gets diluted to near-zero by static
    # background and is unusable as a threshold target directly.
    EVENT_START_THRESHOLD: float = 0.005
    EVENT_END_THRESHOLD: float = 0.002
    EVENT_PAD_SECONDS: float = 3.0
    EVENT_MERGE_GAP_SECONDS: float = 5.0
    EVENT_MIN_DURATION_SECONDS: float = 1.0
    SAVE_FULL_CLIPS: bool = False          # False = timestamps only (saves storage)

    # ---- Paths ----
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent)
    DATA_DIR: Path = field(default_factory=lambda: Path("./data"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("./data/output"))

    def ensure_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()