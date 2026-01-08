from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import base64
import mediapipe as mp
import os
import csv
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_PATH = "model/sign_model.pkl"
DATASET_DIR = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

english_to_arabic = {
    'A': 'أ','B': 'ب','C': 'ت','D': 'ث','E': 'ج','F': 'ح',
    'G': 'خ','H': 'د','I': 'ذ','J': 'ر','K': 'ز','L': 'س',
    'M': 'ش','N': 'ص','O': 'ض','P': 'ط','Q': 'ظ','R': 'ع',
    'S': 'غ','T': 'ف','U': 'ق','V': 'ك','W': 'ل','X': 'م',
    'Y': 'ن','Z': 'ي'
}

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    return None

model = load_model()

def process_base64_image(base64_string):
    try:
        encoded_data = base64_string.split(',')[1] if ',' in base64_string else base64_string
        img_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Decoding Error: {e}")
        return None

@app.route('/frame', methods=['POST'])
def predict_frame():
    global model
    try:
        data = request.json
        frame_base64 = data.get('frame')
        
        if not frame_base64:
            print("Warning: No image frame received in request")
            return jsonify({"error": "لم يتم إرسال الصورة"}), 400

        frame = process_base64_image(frame_base64)
        if frame is None:
            print("Error: Image decoding failed")
            return jsonify({"error": "فشل في معالجة بيانات الصورة"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            print("Success: Hand landmarks detected")
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y])
            
            if model is None:
                print("Error: Prediction attempted but model is not loaded")
                return jsonify({"error": "الموديل غير موجود، يرجى التدريب أولاً"}), 400

            prediction = model.predict(np.array(landmarks).reshape(1, -1))[0]
            arabic_text = english_to_arabic.get(prediction, prediction)
            
            print(f"Prediction result: {prediction} -> {arabic_text}")
            
            return jsonify({
                "translatedText": arabic_text, 
                "label": prediction,
                "confidence": 0.98
            })
        
        print("Info: No hand landmarks detected in the frame")
        return jsonify({"translatedText": "لم يتم اكتشاف إشارة", "confidence": 0})

    except Exception as e:
        print(f"Internal Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test_user_action():
    return predict_frame()

@app.route('/train', methods=['POST'])
def train_model():
    global model
    try:
        X, y = [], []
        if not os.path.exists(DATASET_DIR):
             print(f"Error: Dataset directory {DATASET_DIR} not found")
             return jsonify({"error": "مجلد البيانات غير موجود"}), 400

        for file in os.listdir(DATASET_DIR):
            if file.endswith(".csv"):
                label = file.replace(".csv", "")
                with open(os.path.join(DATASET_DIR, file), "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        X.append(list(map(float, row)))
                        y.append(label)

        if len(X) == 0:
            print("Error: Training failed because dataset is empty")
            return jsonify({"error": "لا توجد بيانات كافية للتدريب"}), 400

        print(f"Start training on {len(X)} samples...")
        new_model = RandomForestClassifier(n_estimators=100)
        new_model.fit(np.array(X), np.array(y))

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        
        model = new_model
        print("Success: Model trained and saved successfully")
        return jsonify({"status": "success", "message": "تم تحديث الموديل بنجاح بعد التدريب"})
    except Exception as e:
        print(f"Training Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("AI Server is running on port 5100")
    app.run(port=5100, debug=True)