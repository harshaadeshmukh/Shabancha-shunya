# Shabdancha Shunya 🤟✨

**Shabdancha Shunya** is a pioneering project that empowers deaf and mute individuals by recognizing hand gestures from Marathi Sign Language in real time. Using a custom dataset, the EfficientNetB0 model, and TensorFlow Lite on a Raspberry Pi, this system provides a low-cost, offline, and portable solution for seamless communication in Marathi-speaking communities. 🚀

## About the Project 🌟

Shabdancha Shunya tackles the gap in technological support for regional sign languages like Marathi Sign Language, used by millions in India. Our system employs advanced computer vision and deep learning to translate hand gestures into text, facilitating effective communication for those with hearing and speech impairments. Deployed on a Raspberry Pi, it operates offline, making it ideal for schools, homes, and rural areas. 💬

The project achieved an impressive **96% accuracy** in recognizing 43 Marathi signs (vowels and consonants). Our mission is to promote inclusivity, preserve regional languages, and make assistive technology accessible and affordable. 🙌

## Features 🎉

- Real-time recognition of 43 Marathi Sign Language gestures 🤲
- 96% classification accuracy with EfficientNetB0 📊
- Lightweight TensorFlow Lite model deployed on Raspberry Pi 🥧
- Custom dataset with 1,000+ images per gesture 📸
- Offline functionality for privacy and accessibility 💸
- 5-inch HDMI touchscreen for instant feedback 🖥️

## Technical Details 🛠️

- **Dataset**: Custom dataset of 43 Marathi signs, each with ~1,000 images, augmented with rotation, flipping, and brightness adjustments. 🖼️
- **Model**: EfficientNetB0 CNN, optimized for efficiency and accuracy (96%). 🧠
- **Framework**: TensorFlow for training, TensorFlow Lite for edge deployment. ⚙️
- **Hardware**: Raspberry Pi 4 with a camera module and 5-inch HDMI touchscreen. 🔌
- **Software**: OpenCV for image preprocessing, MediaPipe for hand detection. 🖥️
- **Use Case**: Real-time gesture-to-text translation for deaf and mute users. 🌍

## Getting Started 🚀

Set up Shabdancha Shunya on a Raspberry Pi or local machine to start recognizing Marathi Sign Language gestures.

### Prerequisites 📋

- Python 3.8+ 🐍
- Raspberry Pi 4 (for deployment) 🍓
- Pi Camera or USB webcam 📷
- 5-inch HDMI touchscreen (optional for display) 🖥️
- System dependencies (Raspberry Pi):
  ```bash
  sudo apt-get update
  sudo apt-get install libatlas-base-dev libopenjp2-7 libtiff5 libjpeg-dev zlib1g-dev python3-tk
  ```

### Installation ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/harshaadeshmukh/Shabancha-shunya.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Shabancha-shunya
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   opencv-python>=4.5.5
   numpy>=1.21.0
   tensorflow>=2.10.0
   Pillow>=9.0.0
   picamera2>=0.3.0
   tflite-runtime>=2.10.0
   cvzone>=1.5.0
   ```

4. (For Raspberry Pi) Configure the TensorFlow Lite model:
   - Ensure the `msl_model.tflite` file is in the project directory. 📂
   - Connect and configure the camera module. 🔧

5. (Optional) For model training, ensure access to the custom dataset and TensorFlow environment. 🖥️

## Usage 🎮

To run the gesture recognition system on a Raspberry Pi:

1. Connect the camera module and (optionally) the HDMI touchscreen. 📹
2. Execute the main script:
   ```bash
   python main.py
   ```
3. Perform Marathi Sign Language gestures in front of the camera. The system will display recognized text on the screen. 🖐️

Example command with custom model and camera:
```bash
python main.py --model msl_model.tflite --camera /dev/video0
```

For training or modifying the model, refer to scripts in the `training/` directory and ensure the dataset is available. 📚

## System Workflow 🔄

1. **Data Collection**: Captured ~1,000 images per gesture for 43 Marathi signs. 📸
2. **Preprocessing**: Resized, normalized, and augmented images using OpenCV. 🖼️
3. **Feature Extraction**: Used MediaPipe for hand landmark detection. ✋
4. **Model Training**: Trained EfficientNetB0 to 96% accuracy with TensorFlow. 🧠
5. **Model Conversion**: Converted to TensorFlow Lite (`msl_model.tflite`) for edge deployment. ⚙️
6. **Real-Time Inference**: Processes live video on Raspberry Pi, displaying text output. 🖥️

## Results 📊

- **Accuracy**: 96% on a 43-class dataset with diverse conditions.
- **Performance**: Smooth real-time recognition on Raspberry Pi with minimal latency.
- **Comparison**: Outperforms models like ResNet (65.96%) and Inception (83.42%) for Marathi gestures, comparable to ensemble Gustafson (98.66%).

## Future Scope 🔮

- Recognize dynamic gestures for full sentences. 📝
- Support additional Indian sign languages (e.g., Tamil, Telugu). 🌐
- Add text-to-speech for spoken output in Marathi or English. 🗣️
- Deploy on mobile devices or wearables for portability. 📱
- Integrate with smart assistants for enhanced functionality. 🤖

## Contributing 🤝

We welcome contributions to enhance Shabdancha Shunya! To contribute:

1. Fork the repository. 🍴
2. Create a new branch (`git checkout -b feature/your-feature`). 🌿
3. Commit your changes (`git commit -m 'Add your feature'`). 💾
4. Push to the branch (`git push origin feature/your-feature`). 🚀
5. Open a Pull Request. 📬

Please follow PEP 8 standards and include tests or documentation updates. ✅

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 🗳️

## Contact 📧

For questions or feedback, reach out to:

- Harshad Deshmukh - GitHub: [@harshaadeshmukh](https://github.com/harshaadeshmukh), Portfolio: [Harshad Deshmukh](https://harshaadeshmukh.github.io/Portfolio/) 😊
- Project Link: [https://github.com/harshaadeshmukh/Shabancha-shunya](https://github.com/harshaadeshmukh/Shabancha-shunya) 🌐

Thank you for exploring Shabdancha Shunya! Join us in making communication accessible for all. 💪✨
