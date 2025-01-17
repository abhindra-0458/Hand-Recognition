import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DATA_PATH = os.path.join('MP_DATA')
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array(['hello','thanks','iloveyou'])
label_map = {label:num for num, label in enumerate(actions)}
no_sequence = 30
sequence_length = 30

cap = cv2.VideoCapture(1)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

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

def collect_training_data():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequence):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Skipping frame due to capture failure.")
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    if frame_num == 0:
                        frame_width = image.shape[1]
                        font_scale = frame_width / 1000
                        cv2.putText(image, 'Starting Collection', (120,150), 
                                  cv2.FONT_HERSHEY_COMPLEX, font_scale, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                  (120,200), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0,0,255), 4, cv2.LINE_AA)
                        cv2.waitKey(3000)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                  (120,200), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0,0,255), 4, cv2.LINE_AA)
                        cv2.waitKey(80)

                    cv2.imshow('OpenCV Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

def train_model():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequence):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=200, callbacks=[tb_callback])
    model.save('action_recognition_model.h5')

if __name__ == "__main__":
    collect_training_data()
    cap.release()
    cv2.destroyAllWindows()
    train_model()