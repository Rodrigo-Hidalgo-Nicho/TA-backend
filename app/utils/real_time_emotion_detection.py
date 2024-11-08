# real_time_emotion_detection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
from app.utils.mobilenet_emotion_model import MobileNetEmotionModel

def start_real_time_emotion_detection():
    model = MobileNetEmotionModel()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion_prediction = model.predict_emotion(frame)
        label = f"{emotion_prediction[1]}: {emotion_prediction[2] * 100:.2f}%"

        # Muestra el resultado en el frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_real_time_emotion_detection()
