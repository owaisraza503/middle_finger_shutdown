# 🖕 Middle Finger Shutdown Detector

This Python project uses your webcam to detect a **middle finger gesture** and automatically shuts down your Ubuntu system if the gesture is held for a specified duration (default: 3 seconds).

## 🧠 How It Works

- Uses [MediaPipe](https://google.github.io/mediapipe/) for real-time hand landmark detection.
- Recognizes if only the **middle finger is raised**.
- Waits for a continuous hold of 3 seconds before triggering the shutdown command.
- Built for fun, experimentation, and showcasing computer vision + system commands.

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
