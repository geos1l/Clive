from dataclasses import dataclass, field
from threading import Lock

@dataclass 
class PerceptionState:
    lock: Lock = field(default_factory=Lock)

    #MediaPipe signals (updated every frame)
    face_present: bool = False
    gaze_on_clive: bool = False
    head_nod: float = 0.0       # pitch in degrees, positive = nodding down
    head_turn: float = 0.0      # yaw in degrees, positive = turned right
    wave_detected: bool = False
    latest_frame: object = None

    # DeepFace signal (updated every 2-3 sec)
    emotion: str = "neutral"

    # Current emotional state (written by state machine)
    emotional_state: str = "idle"
    