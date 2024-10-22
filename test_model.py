
import cv2
from keras.models import load_model
import joblib
import numpy as np
from utils import get_face_landmarks, IMPORTANT_LANDMARKS
import mediapipe as mp
from ultralytics import YOLO  # Import YOLO

# Load YOLO model for face detection
yolo_model = YOLO('./yolov8n-face.pt')

# Load model, scaler, and PCA
model = load_model('model_ferv4_82.keras')
scaler = joblib.load('scaler_ferv4_82.pkl')
pca = joblib.load('pca_fer4_82.pkl')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

labels = ['angry', 'happy', 'neutral', 'surprise']

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()

    results = yolo_model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy 
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            padding = 10
            x1 = max(5, x1 - padding)
            y1 = max(5, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            face_roi = frame[y1:y2, x1:x2]
 
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize the grayscale face to 48x48
            # gray_face_roi_resized = cv2.resize(gray_face_roi, (48, 48))
            gray_face_roi_resized =  gray_face_roi
            
            # Hiển thị khuôn mặt đã resize về 48x48
            cv2.imshow('Resized Grayscale Cropped Face', gray_face_roi_resized)
            cv2.waitKey(1)
            
            face_landmarks_distances = get_face_landmarks(gray_face_roi_resized, face_mesh, draw=False)
            
            if face_landmarks_distances:

                face_landmarks_distances = np.array(face_landmarks_distances).reshape(1, -1)

                scaled_distances = scaler.transform(face_landmarks_distances)

                pca_features = pca.transform(scaled_distances)

                prediction = model.predict(pca_features)
                print(f"Prediction probabilities: {prediction}")
                
                predicted_label = labels[np.argmax(prediction)]

                print(f"Prediction: {predicted_label}")

                cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
    scale_percent = 150
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('frame', frame_resized)
    cv2.waitKey(25)
    
cap.release()
cv2.destroyAllWindows()
