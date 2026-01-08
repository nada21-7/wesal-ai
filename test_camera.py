import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

# ----------- Mediapipe -----------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ----------- Arabic Text Function -----------
def put_arabic_text(img, text, position, font_size=32):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, bidi_text, font=font, fill=(0, 255, 0))

    return np.array(img_pil)

# ----------- Camera -----------
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS
            )

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # ----------- Arabic Text -----------
        frame = put_arabic_text(
            frame,
            "اختبار الكاميرا – مشروع لغة الإشارة",
            (20, 20),
            32
        )

        cv2.imshow("اختبار الكاميرا", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
