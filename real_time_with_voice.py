import cv2
import mediapipe as mp
import pickle
import pyttsx3
import time

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Init TTS
engine = pyttsx3.init()
last_spoken = ""
last_time = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            prediction = model.predict([data])[0]

            # Show text
            cv2.putText(img, prediction, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            # Speak (avoid repeating too fast)
            current_time = time.time()
            if prediction != last_spoken or (current_time - last_time) > 2:
                engine.say(prediction)
                engine.runAndWait()
                last_spoken = prediction
                last_time = current_time

    cv2.imshow("Sign2Speak", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()