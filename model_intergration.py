import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib


# Load model & label encoder
model = tf.keras.models.load_model("bim_hand_sign_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = joblib.load("label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            # Prediction
            prediction = model.predict(landmarks, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            label = label_encoder.inverse_transform([class_id])[0]

            # Display result
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
