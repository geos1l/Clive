import cv2  
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        indices = [1, 152, 33, 263, 61, 291]

        face_2d = []
        face_3d = []
        for idx in indices:
            lm = landmarks[idx]
            x, y = lm.x * w, lm.y * h
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z * 3000])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        _, rvec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        roll = angles[2]

        if abs(roll) > 90:
            pitch = -pitch
            roll = roll - 180 if roll > 0 else roll + 180

        z_vals = [f"{landmarks[i].z:.4f}" for i in indices]
        print(f"pitch={pitch:7.1f}  roll={roll:7.1f}  yaw={angles[1]:7.1f}  |  z={z_vals}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()