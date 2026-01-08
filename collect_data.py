import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

label = input("Enter the label name (example: A, Hello, Yes, No): ")
file_path = f"dataset/{label}.csv"

# Open CSV file
csv_file = open(file_path, "w", newline="")
csv_writer = csv.writer(csv_file)

# Write landmark header
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
csv_writer.writerow(header)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                csv_writer.writerow(row)

                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"Dataset saved → {file_path}")
