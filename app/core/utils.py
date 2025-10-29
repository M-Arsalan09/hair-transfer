import numpy as np
import cv2
import mediapipe as mp
import numpy as np
    
def detect_and_align_face(input_path,
                          save_detected,
                          save_aligned):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Image not found: {input_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("❌ No face detected.")
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]

        # --- Get eye landmarks
        left_eye_outer = face_landmarks.landmark[33]
        right_eye_outer = face_landmarks.landmark[263]
        left_eye = np.array([left_eye_outer.x * w, left_eye_outer.y * h])
        right_eye = np.array([right_eye_outer.x * w, right_eye_outer.y * h])

        # --- Compute rotation angle
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))
        print(f"Detected face tilt angle: {angle:.2f}°")

        # Skip alignment if face is already straight
        if abs(angle) < 2.0:
            print("ℹ️ Face already aligned, skipping rotation.")
            cv2.imwrite(save_detected, image)
            return image, image, None

        # --- Draw detected face
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imwrite(save_detected, annotated)
        print(f"✅ Saved detected face image at: {save_detected}")

        # --- Align face (rotation)
        center_point = np.mean([left_eye, right_eye], axis=0)
        center = (float(center_point[0]), float(center_point[1]))
        pad = int(max(w, h) * 0.25)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        padded_center = (center[0] + pad, center[1] + pad)
        rot_mat_padded = cv2.getRotationMatrix2D(padded_center, angle, 1.0)

        aligned = cv2.warpAffine(
            padded,
            rot_mat_padded,
            (padded.shape[1], padded.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        aligned_cropped = aligned[pad:pad + h, pad:pad + w]
        cv2.imwrite(save_aligned, aligned_cropped)
        print(f"✅ Saved aligned face image at: {save_aligned}")

        return annotated, aligned_cropped, {
            "aligned": aligned,
            "padded_center": padded_center,
            "angle": angle,
            "pad": pad,
            "padded": padded,
            "w": w,
            "h": h
        }


def realign_face(aligned_image, meta):
    """
    Realign the aligned face image back to its original orientation.
    """
    if meta is None:
        print("ℹ️ No alignment was done, skipping realignment.")
        return aligned_image
    aligned = meta["aligned"]
    padded_center = meta["padded_center"]
    angle = meta["angle"]
    pad = meta["pad"]
    padded = meta["padded"]
    h = meta["h"]
    w = meta["w"]

    # Pad again to prevent border artifacts
    pad2 = int(max(w, h) * 0.25)
    aligned_padded = cv2.copyMakeBorder(aligned_image, pad2, pad2, pad2, pad2, cv2.BORDER_REFLECT)
    inverse_rot_mat_padded = cv2.getRotationMatrix2D(padded_center, -angle, 1.0)

    realigned = cv2.warpAffine(aligned, inverse_rot_mat_padded,
                                   (padded.shape[1], padded.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

    realigned_cropped = realigned[pad:pad + h, pad:pad + w]

    return realigned_cropped
