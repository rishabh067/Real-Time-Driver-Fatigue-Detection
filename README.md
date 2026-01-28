# 🚗 Real-Time Driver Fatigue Detection System

A **real-time computer vision system** for detecting driver fatigue using facial landmark analysis, designed for **stable performance under real-world driving conditions**.

The system analyzes **eye closure, yawning, and head posture** from live camera input and triggers alerts when fatigue thresholds are crossed, while maintaining **25–30 FPS** on consumer hardware.

---

## 🧠 Motivation

Driver fatigue is a major contributor to road accidents, especially during long-duration and nighttime driving.  
Most demo or academic systems fail in real-world usage due to:
- Noisy facial landmark detection
- Frequent false positives
- Poor temporal stability

This project focuses on **robust fatigue inference**, **temporal smoothing**, and **real-time performance**, making it suitable for practical deployment.

---

## 🏗️ System Architecture

**Camera Input → Landmark Detection → Feature Extraction → Temporal Analysis → Alert**

1. Live camera feed capture  
2. Face landmark detection using MediaPipe Face Mesh (468 landmarks)  
3. Feature extraction:
   - Eye Aspect Ratio (EAR)
   - Mouth Aspect Ratio (MAR)
   - Head posture cues  
4. Temporal smoothing and threshold-based fatigue inference  
5. Real-time alert triggering (audio / visual)

---

## ⚙️ Core Features

- Real-time processing at **25–30 FPS**
- Robust facial landmark tracking using **MediaPipe Face Mesh**
- **EAR & MAR–based fatigue metrics**
- Temporal smoothing to reduce noise and false alerts
- Multi-condition fatigue detection:
  - Prolonged eye closure
  - Excessive yawning
  - Abnormal head posture
- Modular and extensible design

---

## 📐 Fatigue Metrics

### 👁️ Eye Aspect Ratio (EAR)
- Measures eye openness using eyelid landmarks
- Detects prolonged eye closure indicating drowsiness

### 👄 Mouth Aspect Ratio (MAR)
- Measures mouth openness
- Used to detect frequent or extended yawning

### ⏱️ Temporal Smoothing
- Sliding-window aggregation of EAR and MAR values
- Prevents alert flickering due to transient landmark noise
- Improves stability in real driving conditions

---

## 🚀 Performance

| Metric | Value |
|------|------|
| Frame Rate | 25–30 FPS |
| Face Landmarks | 468 (Face Mesh) |
| Real-Time Capability | Yes |
| False Alerts | Reduced via temporal smoothing |

Tested on live webcam input under varying lighting and head movement conditions.

---

## 🛠️ Tech Stack

**Languages**
- Python
- TypeScript

**Computer Vision**
- OpenCV
- MediaPipe Face Mesh

**Mobile / Frontend**
- React Native

---

## ▶️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rishabh067/Real-Time-Driver-Fatigue-Detection.git
cd Real-Time-Driver-Fatigue-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python main.py
```
Make sure a webcam is connected before running.

### 📂 Project Structure
├── main.py 
├── face_mesh.py
├── fatigue_metrics.py                                                                                                                                                                      
├── smoothing.py

├── config.py                                                                                                                                                                                                
├── utils.py                                                                                                                                                                                                   
├── requirements.txt                                                                                                                                                                                                   
└── README.md

### 🔮 Future Enhancements
📱 Mobile deployment using React Native                                                                                                                                                                              
🧠 ML-based fatigue classification instead of rule-based thresholds                                                                                                                                                   
👤 Personalized fatigue thresholds per driver                                                                                                                                                                        
🚗 Integration with in-vehicle ADAS systems

### 📌 Key Takeaways
- Designed for real-world robustness, not just demo accuracy                                                                                                                                                         
- Focus on performance, stability, and engineering trade-offs                                                                                                                                                         
- Easily extensible to mobile and embedded platforms

⭐ If you find this project useful or interesting, feel free to explore the code and experiments.
