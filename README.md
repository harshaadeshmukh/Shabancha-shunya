# Shabdancha Shunya: Marathi Sign Language Recognition System

**Shabdancha Shunya** is a real-time sign language recognition system tailored for the Marathi language. It leverages deep learning and computer vision to recognize hand gestures corresponding to Marathi characters. The system is trained using a custom dataset and deployed on a Raspberry Pi, making it affordable and portable.

## 🧠 Project Highlights

- **Language Focus**: Marathi Sign Language  
- **Model**: EfficientNetB0 (optimized with TensorFlow Lite)  
- **Dataset**: 43 gestures (Marathi letters), 1000 images per gesture  
- **Technologies**: TensorFlow, OpenCV, Python, TFLite  
- **Hardware**: Raspberry Pi with a 5-inch HDMI display and camera module  
- **Real-Time**: Predicts and displays gesture output live on Raspberry Pi

## 📁 Dataset

- 43 Marathi letters
- 1000 images per letter (captured using OpenCV)
- Custom-created for this project
- Image preprocessing: resizing, normalization, background removal

## 🚀 Workflow

1. **Data Collection**  
   Captured gesture images using OpenCV in real-time.

2. **Preprocessing**  
   Converted images to grayscale, applied filters, resized to model input dimensions.

3. **Model Training**  
   Trained EfficientNetB0 on PC using TensorFlow.

4. **Model Conversion**  
   Converted trained model to TensorFlow Lite (.tflite) format.

5. **Deployment**  
   Deployed TFLite model to Raspberry Pi with custom Python script for real-time recognition.

6. **Output Display**  
   Recognized sign is displayed as corresponding Marathi letter on the screen.

## 🖥️ System Architecture

[Gesture Input] → [Camera Module] → [Preprocessing] → [TFLite Model Inference] → [Marathi Letter Output]



## 🛠️ Requirements

To run this project, install the following Python packages:

```bash
opencv-python>=4.5.5  
numpy>=1.21.0  
tensorflow>=2.10.0  
Pillow>=9.0.0  
picamera2>=0.3.0  
tflite-runtime>=2.10.0  
cvzone>=1.5.0

