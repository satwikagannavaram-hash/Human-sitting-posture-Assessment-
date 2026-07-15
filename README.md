# 🪑 Intelligent Sitting Posture Assessment System using MediaPipe and Random Forest

A real-time computer vision application that monitors a user's sitting posture through a webcam and provides instant posture assessment using **MediaPipe Pose**, **OpenCV**, and a **Random Forest** classifier. The system is designed to encourage healthy sitting habits by detecting incorrect posture and providing immediate visual feedback.

---

## 📌 Overview

Poor sitting posture is one of the leading causes of neck pain, back pain, and musculoskeletal disorders. This project leverages computer vision and machine learning to automatically assess a person's sitting posture in real time.

The system extracts human body landmarks using **MediaPipe Pose**, engineers posture-related features, and classifies posture as **Correct** or **Incorrect** using a trained **Random Forest** model.

---

## ✨ Features

- 📷 Real-time webcam-based posture monitoring
- 🤖 Automatic body landmark detection using MediaPipe Pose
- 🧠 Machine learning-based posture classification
- ⚡ Instant visual feedback for posture correction
- 📊 High-performance Random Forest classifier
- 💻 Lightweight and easy to run on a standard laptop

---

## 🛠 Tech Stack

### Programming Language
- Python

### Computer Vision
- OpenCV
- MediaPipe Pose

### Machine Learning
- Scikit-learn
- Random Forest Classifier

### Data Processing
- NumPy
- Pandas

### Visualization
- OpenCV

---

## 📂 Project Structure

```text
Intelligent-Sitting-Posture-Assessment-System/
│
├── dataset/
├── models/
├── app/
├── images/
├── requirements.txt
├── README.md
└── run.py
```

---

## ⚙️ Working Pipeline

1. Capture live video from webcam.
2. Detect human body landmarks using MediaPipe Pose.
3. Extract posture-related features.
4. Feed extracted features into the trained Random Forest model.
5. Classify posture as:
   - ✅ Correct
   - ❌ Incorrect
6. Display live posture feedback on the screen.

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **96.9%** |
| Precision | **97.0%** |
| Recall | **96.9%** |
| F1 Score | **97.0%** |

The model demonstrates strong classification performance for real-time sitting posture assessment.

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/satwikagannavaram-hash/Human-sitting-posture-Assessment-.git
```

### Navigate to the project directory

```bash
cd Human-sitting-posture-Assessment-
```

### Create a virtual environment (Optional)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the application using:

```bash
python run.py
```

The webcam will open automatically and begin analyzing the user's sitting posture in real time.

---

## 📊 Results

The system:

- Detects upper-body landmarks in real time
- Extracts posture features
- Predicts sitting posture accurately
- Provides instant visual feedback
- Operates with low latency suitable for real-time applications

---

## 🔮 Future Improvements

- Multi-class posture classification
- Personalized posture recommendations
- Posture score calculation
- Automatic posture correction suggestions
- Session duration monitoring
- Daily posture analytics dashboard
- Mobile application integration
- Deep learning-based posture recognition

---

## 🎯 Applications

- Workplace ergonomics
- Remote learning
- Smart office environments
- Health monitoring systems
- Rehabilitation assistance
- Personal wellness applications

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push the branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Gannavaram Lakshmi Satwika**

