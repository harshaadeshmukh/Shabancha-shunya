# Shabdancha Shunya 🤟✨

**Shabdancha Shunya** is an innovative project aimed at bridging the communication gap for individuals who are deaf or mute by recognizing hand gestures from Marathi sign language. Using a custom dataset of Marathi hand signs and a highly efficient machine learning model, this system enables real-time gesture recognition on low-power devices like the Raspberry Pi. 🚀

## About the Project 🌟

Shabdancha Shunya leverages computer vision and machine learning to recognize hand gestures used in Marathi sign language, enabling seamless communication for individuals who are deaf or mute. By creating a custom dataset of hand signs and deploying an optimized model on a Raspberry Pi, the project offers a practical, affordable, and real-time solution tailored to a regional language. 💬

The primary goal is to empower users by providing an accessible tool that facilitates communication using Marathi sign language, running efficiently on low-power hardware. 🙌

## Features 🎉

- Real-time recognition of Marathi sign language hand gestures 🤲
- High accuracy (\~96%) using the EfficientNetB0 model 📊
- Lightweight deployment on Raspberry Pi with TensorFlow Lite 🥧
- Custom dataset tailored to Marathi sign language 📸
- Low-cost and portable solution for accessibility 💸

## Technical Details 🛠️

- **Dataset**: Custom collection of images capturing various hand gestures used in Marathi sign language. 🖼️
- **Model**: EfficientNetB0, chosen for its balance of performance and efficiency, achieving \~96% accuracy. 🧠
- **Framework**: TensorFlow for model training, converted to TensorFlow Lite for deployment. ⚙️
- **Hardware**: Raspberry Pi for real-time, low-power gesture recognition. 🔌
- **Use Case**: Real-time translation of hand gestures to facilitate communication for deaf and mute individuals. 🌍

## Getting Started 🚀

Follow these instructions to set up and run the project locally or on a Raspberry Pi.

### Prerequisites 📋

- Python 3.8+ 🐍
- TensorFlow 2.x (for training) and TensorFlow Lite (for deployment) 🧩
- Raspberry Pi (for deployment, e.g., Raspberry Pi 4) 🍓
- Camera module compatible with Raspberry Pi (e.g., Pi Camera) 📷
- pip for Python package management 📦
- Git for cloning the repository 🌐

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

4. (For Raspberry Pi) Set up the TensorFlow Lite model:

   - Copy the `.tflite` model file to the Raspberry Pi. 📂
   - Ensure the camera module is connected and configured. 🔧

5. (Optional) For training the model, ensure access to the custom dataset and TensorFlow environment. 🖥️

## Usage 🎮

To run the gesture recognition system on a Raspberry Pi:

1. Connect the camera module to the Raspberry Pi. 📹

2. Execute the main script:

   ```bash
   python run_gesture_recognition.py
   ```

3. The system will process live video feed and output recognized Marathi sign language gestures. 🖐️

For training or modifying the model, refer to the scripts in the `training/` directory and ensure the custom dataset is available. 📚

Example usage on Raspberry Pi:

```bash
python run_gesture_recognition.py --model model.tflite --camera /dev/video0
```

## Contributing 🤝

We welcome contributions to enhance Shabdancha Shunya! To contribute:

1. Fork the repository. 🍴
2. Create a new branch (`git checkout -b feature/your-feature`). 🌿
3. Commit your changes (`git commit -m 'Add your feature'`). 💾
4. Push to the branch (`git push origin feature/your-feature`). 🚀
5. Open a Pull Request. 📬

Please ensure your code adheres to PEP 8 standards and includes relevant tests or documentation updates. ✅

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details. 🗳️

## Contact 📧

For questions or feedback, reach out to:

- Harshad Deshmukh - GitHub: @harshaadeshmukh 😊
- Project Link: https://github.com/harshaadeshmukh/Shabancha-shunya 🌐

Thank you for exploring Shabdancha Shunya! Let's work together to make communication more accessible. 💪✨
