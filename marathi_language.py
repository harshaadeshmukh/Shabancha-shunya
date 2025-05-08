import cv2
import numpy as np
import json
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from cvzone.HandTrackingModule import HandDetector
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image

class MarathiSignLanguageApp:
    def __init__(self):
        # Load model and class indices
        self.model, self.inv_map = self.load_model_and_indices()
        self.predictions_queue = deque(maxlen=5)

        # Initialize hand detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Initialize Pi Camera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        self.picam2.start()
        time.sleep(2)  # Camera warm-up

        # Camera settings
        self.picam2.set_controls({
            "AwbEnable": False,
            "ColourGains": (1.5, 1.0),
            "Brightness": 0.5,
            "Contrast": 1.2,
            "Saturation": 1.5
        })

        # Display and ROI settings
        self.roi_size = 224
        self.roi_x = (640 - self.roi_size) // 2
        self.roi_y = (480 - self.roi_size) // 2
        self.is_running = True
        self.sentence = ""
        self.last_letter = None
        self.last_letter_time = 0

        # Load Marathi font
        self.font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
        self.font = ImageFont.truetype(self.font_path, 32)

    def load_model_and_indices(self, model_path='msl_model.tflite', indices_path='class_indices.json'):
        with open(indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        inv_map = {int(v): k for k, v in class_indices.items()}
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, inv_map

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        img_array = frame.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_sign(self, frame):
        img_array = self.preprocess_frame(frame)
        self.model.set_tensor(self.model.get_input_details()[0]['index'], img_array)
        self.model.invoke()
        predictions = self.model.get_tensor(self.model.get_output_details()[0]['index'])[0]
        self.predictions_queue.append(predictions)
        avg_predictions = np.mean(self.predictions_queue, axis=0)
        predicted_class = np.argmax(avg_predictions)
        predicted_letter = self.inv_map[predicted_class]
        confidence = avg_predictions[predicted_class]
        return predicted_letter, confidence

    def draw_marathi_text(self, frame, text, position, font_size=32, color=(0, 255, 0)):
        image_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(self.font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return np.array(image_pil)

    def run(self):
        print("[INFO] Starting Marathi Sign Language detection...")

        while self.is_running:
            frame = self.picam2.capture_array()
            frame = cv2.flip(frame, 1)
            hands, _ = self.detector.findHands(frame, draw=False)
            roi = None

            if hands:
                x, y, w, h = hands[0]['bbox']
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                roi = frame[self.roi_y:self.roi_y + self.roi_size, self.roi_x:self.roi_y + self.roi_size]
                cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_size, self.roi_y + self.roi_size), (0, 255, 0), 2)

            if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                predicted_letter, confidence = self.predict_sign(roi)
                current_time = time.time()
                if (confidence > 0.7 and predicted_letter != self.last_letter and current_time - self.last_letter_time > 0.5):
                    self.sentence += predicted_letter
                    self.last_letter = predicted_letter
                    self.last_letter_time = current_time

                # ✅ Use PIL to draw prediction
                text = f"{predicted_letter} ({confidence:.2f})"
                frame = self.draw_marathi_text(frame, text, (self.roi_x, self.roi_y - 40))

            # ✅ Draw full sentence with Marathi font
            frame = self.draw_marathi_text(frame, f"Sentence: {self.sentence}", (50, 50))

            cv2.imshow("Marathi Sign Language Detection - PiCam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                self.is_running = False

        cv2.destroyAllWindows()
        self.picam2.close()


def main():
    app = MarathiSignLanguageApp()
    app.run()


if __name__ == '__main__':
    main()
