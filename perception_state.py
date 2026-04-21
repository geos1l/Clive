from dataclasses import dataclass, field
from threading import Lock

@dataclass 
class PerceptionState:
    lock: Lock = field(default_factory=Lock)

    #MediaPipe signals (updated every frame)
    face_present: bool = False
    gaze_on_clive: bool = False
    head_nod: float = 0.0       # pitch in degrees, positive = nodding down
    head_tilt: float = 0.0      # roll in degrees
    wave_detected: bool = False

    # DeepFace signal (updated every 2-3 sec)
    emotion: str = "neutral"

    # Derived by state machine (read by motion layer)
    look_away_count: int = 0    # tracks repeated look-aways for SHY state