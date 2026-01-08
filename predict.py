import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------- Arabic Text Function --------
def put_arabic_text(img, text, position, font_size=40):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", font_size)

    draw.text(position, bidi_text, font=font, fill=(0, 255, 0))
    return np.array(img_pil)

# تحويل إنجليزي → عربي
english_to_arabic = {
    'A': 'أ','B': 'ب','C': 'ت','D': 'ث','E': 'ج','F': 'ح',
    'G': 'خ','H': 'د','I': 'ذ','J': 'ر','K': 'ز','L': 'س',
    'M': 'ش','N': 'ص','O': 'ض','P': 'ط','Q': 'ظ','R': 'ع',
    'S': 'غ','T': 'ف','U': 'ق','V': 'ك','W': 'ل','X': 'م',
    'Y': 'ن','Z': 'ي'
}

# تحميل الموديل
with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                data = np.array(data).reshape(1, -1)
                prediction = model.predict(data)[0]
                arabic_letter = english_to_arabic.get(prediction, prediction)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                text = f"الإشارة: {arabic_letter}"
                frame = put_arabic_text(frame, text, (20, 20), 40)

        cv2.imshow("التعرف على الإشارة", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
