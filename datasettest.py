import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

base_path = r"C:\Users\hiine\Desktop\FYP\FYP_dataset\Dataset"
categories = ["Alphabets", "Numbers", "SingleWords"]

data = []
labels = []
category_labels = []

for category in categories:
    category_path = os.path.join(base_path, category)

    for gesture in os.listdir(category_path):
        gesture_path = os.path.join(category_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for img_name in os.listdir(gesture_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(gesture_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                data.append(landmarks)
                labels.append(gesture)        # A, B, 1, Hello
                category_labels.append(category)  # Alphabets, Numbers, SingleWords

hands.close()

# Create landmark column names
columns = []
for i in range(21):
    columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

df = pd.DataFrame(data, columns=columns)
df["label"] = labels
df["category"] = category_labels

df.to_csv("hand_landmarks_all.csv", index=False)
print("✅ CSV created successfully: hand_landmarks_all.csv")
