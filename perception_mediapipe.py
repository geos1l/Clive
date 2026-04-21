import cv2
import mediapipe as mp
import numpy as np
from perception_state import PerceptionState

def _estimate_gaze(landmarks, frame_width: int) -> bool:
    """Check if iris is roughly centered in the eye, meaning they're looking at us"""
    # Right eye: inner corner = 133, outer corner = 33, iris center = 468
    # Left eye: inner corner = 362, outer corner = 263, iris center = 473
    def iris_ratio(inner_idx, outer_idx, iris_idx):
        inner_x = landmarks[inner_idx].x
        outer_x = landmarks[outer_idx].x
        iris_x = landmarks[iris_idx].x
        eye_width = abs(outer_x - inner_x)
        if eye_width < 1e-6:
            return 0.5
        return abs(iris_x - outer_x) / eye_width
    
    right_ratio = iris_ratio(133, 33, 468)
    left_ratio = iris_ratio(362, 263, 473)
    avg = (right_ratio + left_ratio) / 2

    # Centered iris is about 0.4-0.6 ratio. Outside of that = looking away
    return 0.35 < avg < 0.65

def _estimate_head_pose(landmarks, frame_h: int, frame_w: int):
    """Compute head pitch (nod) and roll (tilt) using solvePnP from OpenCV"""
    # 6 key face points for pose estimation
    # nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
    face_2d = []
    face_3d = []
    indices = [1, 152, 33, 263, 61, 291]

    for idx in indices: # where is idx coming from
        lm = landmarks[idx]
        x, y = lm.x * frame_w, lm.y * frame_h
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z * 3000]) # what does this do

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix - approximate, assumes camera is at image center 
    # ^^ what does that mean practically, and is this realistic 
    focal_length = frame_w
    cam_matrix = np.array([
        [focal_length, 0, frame_w / 2],
        [0, focal_length, frame_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    _, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vec) # what is this _ stuff how is that a variable whats happening here
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch = angles[0] # nod: positive = looking down
    roll = angles[2] # tilt: positive = tilting right 
    return pitch, roll

def _detect_wave(hand_landmarks) -> bool:
    """Wave = hand is high in frame with fingers extended"""
    wrist_y = hand_landmarks[0].y # 0.0 = top of frame, 1.0 = bottom

    # Can play around/change this later, for now leave as top 60%  
    # Hand must be in upper 60% of frame
    if wrist_y > 0.6:
        return False
    
    # Check if fingers are extended: fingertip y < knuckle y (y=0 is top)
    # Index=8/6, Middle=12/10, Ring=16/14, Pinky=20/18
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    extended = sum(
        1 for t, k in zip(tips, knuckles)
        if hand_landmarks[t].y < hand_landmarks[k].y
    )

    # Can add thumb later but have to compare x coords and left vs right hand matters

    return extended >= 3 # at least 3 fingers up
    # Can lowk change this to be 4 lets start with 3 tho


def run_mediapipe(state: PerceptionState, camera_index: int = 0):
    """Main loop; runs on its own thread, writes to shared state"""
    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # enables iris landmarks (468-477)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,    # add 2 later
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = mp_face.process(rgb)
        hand_results = mp_hands.process(rgb)

        # Extracts signals
        face_present = face_results.multi_face_landmarks is not None
        gaze = False
        nod = 0.0
        tilt = 0.0
        wave = False

        if face_present:
            landmarks = face_results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            gaze = _estimate_gaze(landmarks, w)
            nod, tilt = _estimate_head_pose(landmarks, h, w)

        if hand_results.multi_hand_landmarks:
            hand_lm = hand_results.multi_hand_landmarks[0].landmark
            wave = _detect_wave(hand_lm)

        # Draw skeleton on camera 
        annotated = frame.copy()
        if face_present:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                face_results.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
            )
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated,
                    hand_lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )
        cv2.putText(annotated, f"gaze={gaze} nod={nod:.1f} wave={wave}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("MediaPipe Debug", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write to shared state (lock!)
        with state.lock: 
            state.face_present = face_present
            state.gaze_on_clive = gaze
            state.head_nod = nod
            state.head_tilt = tilt
            state.wave_detected = wave

    cap.release()

