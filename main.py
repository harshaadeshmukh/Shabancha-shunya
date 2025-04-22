import cv2
import numpy as np
import tensorflow as tf
import json
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from collections import deque
import threading


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi Sign Language Detection")
        self.root.geometry("800x600")

        # Load model and class indices
        self.model, self.inv_map = self.load_model_and_indices()
        self.predictions_queue = deque(maxlen=5)  # Store last 5 predictions for smoothing

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.quit()

        # GUI elements
        self.video_label = Label(self.root)
        self.video_label.pack(pady=10)

        self.result_label = Label(self.root, text="Predicted Letter: None", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.start_button = Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = Button(self.root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.quit_button = Button(self.root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=5)

        # Detection control flags
        self.is_detecting = False
        self.roi_size = 224
        self.roi_x, self.roi_y = 100, 100

    def load_model_and_indices(self, model_path='marathi_sign_language_model.h5', indices_path='class_indices.json'):
        # Load class indices
        with open(indices_path, 'r') as f:
            class_indices = json.load(f)
        inv_map = {v: k for k, v in class_indices.items()}
        # Load model
        model = tf.keras.models.load_model(model_path)
        return model, inv_map

    def preprocess_frame(self, frame):
        # Resize to match model input size
        frame = cv2.resize(frame, (224, 224))
        # Convert to array and preprocess for EfficientNet
        img_array = tf.keras.preprocessing.image.img_to_array(frame)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_sign(self, frame):
        # Preprocess frame
        img_array = self.preprocess_frame(frame)
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        self.predictions_queue.append(predictions[0])  # Store prediction

        # Smooth predictions by averaging
        avg_predictions = np.mean(self.predictions_queue, axis=0)
        predicted_class = np.argmax(avg_predictions)
        predicted_letter = self.inv_map[predicted_class]
        confidence = avg_predictions[predicted_class]
        return predicted_letter, confidence

    def update_frame(self):
        if not self.is_detecting:
            return

        ret, frame = self.cap.read()
        if ret:
            # Flip frame for intuitive display
            frame = cv2.flip(frame, 1)

            # Extract ROI
            roi = frame[self.roi_y:self.roi_y + self.roi_size, self.roi_x:self.roi_x + self.roi_size]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                return

            # Predict sign
            predicted_letter, confidence = self.predict_sign(roi)

            # Update result label
            self.result_label.config(text=f"Predicted Letter: {predicted_letter} ({confidence:.2f})")

            # Draw ROI and prediction on frame
            cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x + self.roi_size, self.roi_y + self.roi_size),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (self.roi_x, self.roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)  # Resize for display
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Schedule next frame update
        if self.is_detecting:
            self.root.after(10, self.update_frame)

    def start_detection(self):
        if not self.is_detecting:
            self.is_detecting = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            # Start updating frames
            self.update_frame()

    def stop_detection(self):
        if self.is_detecting:
            self.is_detecting = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.video_label.config(image='')  # Clear video display
            self.result_label.config(text="Predicted Letter: None")

    def quit_app(self):
        self.is_detecting = False
        self.cap.release()
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
