import cv2
import mediapipe as mp
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

def _estimate_head_pose(landmarks):
    """Estimate nod and turn using 2D landmark ratios — distance-invariant, no camera matrix needed."""
    # Nod (pitch): compare nose-to-chin vs nose-to-forehead vertical distances
    # When you nod down, nose gets closer to chin and farther from forehead
    # Landmark 1 = nose tip, 152 = chin, 10 = forehead
    nose_y = landmarks[1].y
    chin_y = landmarks[152].y
    forehead_y = landmarks[10].y

    nose_to_chin = chin_y - nose_y
    nose_to_forehead = nose_y - forehead_y

    face_height = chin_y - forehead_y
    if face_height < 1e-6:
        return 0.0, 0.0

    # Ratio: 0.5 = nose is halfway between forehead and chin (neutral)
    # Higher = nodding down, lower = looking up
    nod_ratio = nose_to_chin / face_height
    nod = (nod_ratio - 0.5) * 100  # scale to roughly degrees-ish

    # Turn (yaw): compare nose-to-left-ear vs nose-to-right-ear horizontal distances
    # When you turn right, nose gets closer to right ear and farther from left
    # Landmark 1 = nose tip, 234 = right ear, 454 = left ear
    nose_x = landmarks[1].x
    left_ear_x = landmarks[454].x
    right_ear_x = landmarks[234].x

    face_width = left_ear_x - right_ear_x
    if abs(face_width) < 1e-6:
        return nod, 0.0

    # Ratio: 0.5 = nose centered between ears (looking straight)
    # Higher = turned left, lower = turned right
    nose_offset = nose_x - right_ear_x
    turn_ratio = nose_offset / face_width
    turn = (turn_ratio - 0.5) * -100  # negative so right turn = positive

    return nod, turn

def _detect_wave(hand_landmarks, handedness: str) -> bool:
    """Wave = hand is high in frame with fingers extended"""
    wrist_y = hand_landmarks[0].y # 0.0 = top of frame, 1.0 = bottom

    # Hand must be in upper 80% of frame
    if wrist_y > 0.9:
        return False
    
    # Check if fingers are extended: fingertip y < knuckle y (y=0 is top)
    # Index=8/6, Middle=12/10, Ring=16/14, Pinky=20/18
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    extended = sum(
        1 for t, k in zip(tips, knuckles)
        if hand_landmarks[t].y < hand_landmarks[k].y
    )

    # Thumb: compare x not y. Direction depend on hand
    thumb_tip_x = hand_landmarks[4].x
    thumb_knuckle_x = hand_landmarks[2].x
    if handedness == "Left": # mirrored, so left is actually right hand
        thumb_extended = thumb_tip_x > thumb_knuckle_x
    else:
        thumb_extended = thumb_tip_x < thumb_knuckle_x
    
    if thumb_extended:
        extended += 1

    return extended >= 4 # at least 4 fingers up


def run_mediapipe(state: PerceptionState, camera_index: int = 0):
    """Main loop; runs on its own thread, writes to shared state"""
    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # enables iris landmarks (468-477)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(camera_index)

    smooth_nod = 0.0
    smooth_turn = 0.0

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
        turn = 0.0
        wave = False

        if face_present:
            landmarks = face_results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            gaze = _estimate_gaze(landmarks, w)
            nod, turn = _estimate_head_pose(landmarks)
            alpha = 0.3 # lower = smoother, higher = more responsive
            smooth_nod = alpha * nod + (1 - alpha) * smooth_nod
            smooth_turn = alpha * turn + (1 - alpha) * smooth_turn
            nod = smooth_nod
            turn = smooth_turn

        if hand_results.multi_hand_landmarks:
            wave = any(
                _detect_wave(
                    hand_lm.landmark,
                    hand_results.multi_handedness[i].classification[0].label,
                    )
                for i, hand_lm in enumerate(hand_results.multi_hand_landmarks)
            ) 

        # Draw skeleton on camera 
        annotated = frame.copy()
        if face_present:
              lms = face_results.multi_face_landmarks[0].landmark
              h, w = frame.shape[:2]

              # Draw only the landmarks we use
              # Iris: 468 (right), 473 (left)
              # Eye corners: 33, 133 (right eye), 263, 362 (left eye)
              # Head pose: 1 (nose), 152 (chin), 10 (forehead), 234 (right ear), 454 (left ear)
              for idx in [468, 473, 33, 133, 263, 362, 1, 152, 10, 234, 454]:
                  x = int(lms[idx].x * w)
                  y = int(lms[idx].y * h)
                  cv2.circle(annotated, (x, y), 3, (255, 255, 255), -1)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated,
                    hand_lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )
        cv2.putText(annotated, f"gaze={gaze} nod={nod:.1f} turn={turn:.1f} wave={wave}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        with state.lock:
            current_emotion = state.emotion
            current_state = state.emotional_state
        cv2.putText(annotated, f"emotion={current_emotion}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated, f"STATE={current_state}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("MediaPipe Debug", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write to shared state (lock!)
        with state.lock: 
            state.face_present = face_present
            state.gaze_on_clive = gaze
            state.head_nod = nod
            state.head_turn = turn
            state.wave_detected = wave
            state.latest_frame = frame.copy()

    cap.release()

