from collections import Counter, deque

import cv2
import mediapipe as mp
import pickle


LANDMARK_BUFFER_SIZE = 10
PREDICTION_BUFFER_SIZE = 5
TEXT_COLOR = (0, 255, 0)
TEXT_SHADOW_COLOR = (0, 0, 0)


# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
model_classes = list(model.classes_)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def extract_landmark_data(hand_landmarks):
    """Flatten 21 hand landmarks into [x1, y1, x2, y2, ...]."""
    data = []
    for landmark in hand_landmarks.landmark:
        data.append(landmark.x)
        data.append(landmark.y)
    return data


def average_landmarks(landmark_buffer):
    """Average each feature across buffered frames."""
    feature_count = len(landmark_buffer[0])
    return [
        sum(frame[feature_index] for frame in landmark_buffer) / len(landmark_buffer)
        for feature_index in range(feature_count)
    ]


def get_stable_prediction(prediction_buffer):
    """Return the most frequent prediction in the recent history."""
    return Counter(prediction_buffer).most_common(1)[0][0]


def predict_from_buffer(landmark_buffer):
    """Predict class probabilities from averaged buffered landmarks only."""
    averaged_data = average_landmarks(landmark_buffer)
    probabilities = model.predict_proba([averaged_data])[0]
    predicted_index = probabilities.argmax()
    return model_classes[predicted_index], probabilities


def format_gesture_name(raw_name):
    """Convert model class labels like 'thumbs_up' to 'Thumbs Up'."""
    return raw_name.replace("_", " ").title()


def draw_prediction_ui(frame, prediction, confidence):
    """Draw prediction and confidence with better readability."""
    gesture_text = f"Detected: {format_gesture_name(prediction)}"
    confidence_text = f"Confidence: {confidence * 100:.1f}%"

    # Shadow for readability
    cv2.putText(
        frame,
        gesture_text,
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_SHADOW_COLOR,
        5,
    )
    cv2.putText(
        frame,
        confidence_text,
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_SHADOW_COLOR,
        4,
    )

    # Foreground text
    cv2.putText(
        frame,
        gesture_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        2,
    )
    cv2.putText(
        frame,
        confidence_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
    )


landmark_buffer = deque(maxlen=LANDMARK_BUFFER_SIZE)
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Keep existing behavior: draw and predict using first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_data = extract_landmark_data(hand_landmarks)
        landmark_buffer.append(frame_data)

        if len(landmark_buffer) == LANDMARK_BUFFER_SIZE:
            current_prediction, probabilities = predict_from_buffer(landmark_buffer)
            confidence = probabilities[model_classes.index(current_prediction)]

            prediction_buffer.append(current_prediction)
            stable_prediction = get_stable_prediction(prediction_buffer)

            stable_index = model_classes.index(stable_prediction)
            stable_confidence = probabilities[stable_index]

            draw_prediction_ui(img, stable_prediction, stable_confidence)

    cv2.imshow("Sign2Speak", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
