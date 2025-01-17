import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
actions = np.array(['hello','thanks','iloveyou'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                         mp_draw.DrawingSpec(color=(108,171,221), thickness=1, circle_radius=1),
                         mp_draw.DrawingSpec(color=(71,225,12), thickness=1, circle_radius=1))
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                         mp_draw.DrawingSpec(color=(133,171,200), thickness=2, circle_radius=3),
                         mp_draw.DrawingSpec(color=(71,200,50), thickness=2, circle_radius=2))
    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                         mp_draw.DrawingSpec(color=(130,121,111), thickness=2, circle_radius=1),
                         mp_draw.DrawingSpec(color=(221,15,122), thickness=2, circle_radius=3))
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                         mp_draw.DrawingSpec(color=(108,141,241), thickness=2, circle_radius=2),
                         mp_draw.DrawingSpec(color=(111,25,122), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def test_model():
    model = load_model('action_recognition_model.h5')
    cap = cv2.VideoCapture(1)
    sequence = []
    threshold = 0.6

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                prediction_confidence = res[np.argmax(res)]
                
                if prediction_confidence > threshold:
                    cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
                    cv2.putText(image, predicted_action, (3,30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()

