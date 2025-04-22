import cv2
import numpy as np
import json
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from cvzone.HandTrackingModule import HandDetector
from collections import deque
import time


class MarathiSignLanguageApp:
    def __init__(self):
        # Load model and class indices
        self.model, self.inv_map = self.load_model_and_indices()
        self.predictions_queue = deque(maxlen=5)  # Store last 5 predictions for smoothing

        # Initialize hand detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Initialize Pi Camera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        self.picam2.start()
        time.sleep(2)  # Camera warm-up

        # Camera settings (tuned for lighting)
        self.picam2.set_controls({
            "AwbEnable": False,
            "ColourGains": (1.5, 1.0),  # Adjust red/blue balance
            "Brightness": 0.5,
            "Contrast": 1.2,
            "Saturation": 1.5
        })

        # Settings
        self.roi_size = 224  # Model input size
        self.is_running = True
        self.sentence = ""  # Accumulate predicted letters
        self.last_letter = None  # Track last letter to avoid rapid repeats
        self.last_letter_time = 0  # Timestamp for last letter addition

    def load_model_and_indices(self, model_path='msl_model.tflite', indices_path='class_indices.json'):
        # Load class indices
        with open(indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        inv_map = {int(v): k for k, v in class_indices.items()}  # Ensure keys are integers

        # Load TFLite model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, inv_map

    def preprocess_frame(self, frame):
        # Resize to match model input size
        frame = cv2.resize(frame, (224, 224))

        # Convert to float32, keep pixel values in [0-255] as expected by EfficientNet
        img_array = frame.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_sign(self, frame):
        # Preprocess frame
        img_array = self.preprocess_frame(frame)

        # Run inference
        self.model.set_tensor(self.model.get_input_details()[0]['index'], img_array)
        self.model.invoke()
        predictions = self.model.get_tensor(self.model.get_output_details()[0]['index'])[0]

        # Store prediction
        self.predictions_queue.append(predictions)

        # Smooth predictions by averaging
        avg_predictions = np.mean(self.predictions_queue, axis=0)
        predicted_class = np.argmax(avg_predictions)
        predicted_letter = self.inv_map[predicted_class]
        confidence = avg_predictions[predicted_class]
        return predicted_letter, confidence

    def run(self):
        print("[INFO] Starting Marathi Sign Language detection...")

        while self.is_running:
            # Capture frame
            frame = self.picam2.capture_array()
            frame = cv2.flip(frame, 1)  # Mirror horizontally

            # Detect hands
            hands, _ = self.detector.findHands(frame, draw=False)
            roi = None

            if hands:
                # Extract dynamic ROI based on hand bounding box
                x, y, w, h = hands[0]['bbox']
                # Add padding to capture full hand
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                roi = frame[y1:y2, x1:x2]

                # Draw bounding box around hand
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Fallback to static ROI if no hand detected
                roi = frame[self.roi_y:self.roi_y + self.roi_size, self.roi_x:self.roi_x + self.roi_size]
                cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_size, self.roi_y + self.roi_size),
                              (0, 255, 0), 2)

            # Process ROI if valid
            if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                # Predict sign
                predicted_letter, confidence = self.predict_sign(roi)

                # Update sentence if confidence is high and letter is stable
                current_time = time.time()
                if (confidence > 0.7 and
                        predicted_letter != self.last_letter and
                        current_time - self.last_letter_time > 0.5):  # Avoid rapid repeats
                    self.sentence += predicted_letter
                    self.last_letter = predicted_letter
                    self.last_letter_time = current_time

                # Draw prediction on frame
                cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (self.roi_x, self.roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display sentence
            cv2.putText(frame, f"Sentence: {self.sentence}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display frame
            cv2.imshow("Marathi Sign Language Detection - PiCam", frame)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                self.is_running = False

        # Cleanup
        cv2.destroyAllWindows()
        self.picam2.close()


def main():
    app = MarathiSignLanguageApp()
    app.run()


if __name__ == '__main__':
    main()
