# 🎯 SentinelAI — Intelligent Surveillance System

Real-time weapon, crowd, and unattended bag detection powered by a **2-Stage Zero False-Positive Pipeline** (RT-DETR-X + EfficientNet-B4) and **Streamlit** (with a React dashboard planned).

> **Important:** This repository is currently transitioning from a baseline YOLOv8 prototype to a production-grade **Zero False-Positive** architecture. Please read the full [ROADMAP.md](./ROADMAP.md) for the step-by-step production plan.

---

## 📌 What This Project Does

SentinelAI is a real-time smart surveillance system designed to monitor CCTV / camera feeds and automatically alert security personnel when anomalies occur:
- 🔫 **Weapon Detection:** Identifies firearms (pistols, rifles) and sharp objects (knives).
- 🎒 **Unattended Bag Detection:** Alerts if a bag is left alone without its owner nearby.
- 👥 **Crowd Detection:** Monitors group density and flags overcrowded areas.
- 🥊 **Altercation Detection:** Detects physical fights or aggressive behavior.

---

## 🧠 How It Works — End-to-End Pipeline (Planned 2-Stage Architecture)

To achieve a **< 0.5% False Positive Rate**, the system uses a 3-gate verification process:

```
┌─────────────────┐
│ Camera / Video  │ (Input: MP4, RTSP CCTV, or Live Webcam)
└────────┬────────┘
         │  Frame-by-Frame (OpenCV)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. GATE 1: FAST DETECTOR (RT-DETR-X)                        │
│    Scans frame for potential threats (Conf > 0.65)          │
│    ➜ Transformer-based attention, no NMS needed             │
└────────┬────────────────────────────────────────────────────┘
         │  Threat Candidates
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GATE 2: VERIFIER CLASSIFIER (EfficientNet-B4)            │
│    Crops candidate region and re-examines (Conf > 0.85)     │
│    ➜ Specifically trained on hard negatives (e.g., remotes) │
└────────┬────────────────────────────────────────────────────┘
         │  Verified Threats
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GATE 3: TEMPORAL RULES ENGINE (src/rules.py)             │
│    Evaluates business logic over time                       │
│    ➜ Weapon must persist for 8 consecutive frames           │
└────────┬────────────────────────────────────────────────────┘
         │  Active Alerts
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. VISUALIZATION & DASHBOARD (src/streamlit_app.py)         │
│    Streams live video feed & alerts to UI                   │
│    ➜ (Future: React + Tailwind CSS dashboard)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure & Codebase Mapping

```
DetectiveAI-camera/
├── src/
│   ├── detection.py       # 🔍 2-Stage Pipeline (RT-DETR-X + EfficientNet)
│   ├── tracking.py        # 🎯 DeepSORT object tracker (assigns persistent IDs)
│   ├── rules.py           # 🧠 Business Logic (Weapon persist, Bag timer, Crowd count)
│   ├── visualize.py       # 🎨 Draws bounding boxes, IDs, and alerts
│   └── streamlit_app.py   # 💻 Monitoring Dashboard UI
├── models/
│   ├── best.onnx          # 🤖 Gate 1 Detector (RT-DETR-X)
│   └── verifier.onnx      # 🔍 Gate 2 Verifier (EfficientNet)
├── videos/
│   └── cam1.mp4           # 📹 Test surveillance video footage
├── requirements.txt       # 📦 Python dependency list
├── ROADMAP.md             # 🗺️ Full production & training roadmap
└── README.md              # 📖 Project documentation
```

---

## ⚡ Performance & Benchmarks

| Hardware | Provider | Inference Latency | FPS |
|---|---|---|---|
| **NVIDIA GPU (CUDA)** | `CUDAExecutionProvider` | **~15 ms** (2-stage) | **~60 FPS** |
| **NVIDIA RTX 2050** | `CUDAExecutionProvider` | **~25 ms** (2-stage) | **~40 FPS** |
| **Standard Intel CPU** | `CPUExecutionProvider` | ~100 ms | ~10 FPS |

To enable GPU on NVIDIA machines:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## 🚀 Quickstart & Setup

### Step 1 — Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Dashboard
```bash
streamlit run src/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎛️ Key Configuration Parameters (Production Grade)

In `src/rules.py` (designed for Zero False Positives):

```python
# Confidence Gates
WEAPON_CONF_GATE1 = 0.65       # Gate 1 initial scan threshold
WEAPON_CONF_GATE2 = 0.85       # Gate 2 verification threshold

# Temporal Rules
WEAPON_PERSIST_FRAMES = 8      # Frames weapon must persist before alert
BAG_STATIONARY_SECONDS = 30    # Time bag must be alone & still
CROWD_THRESHOLD = 15           # Number of people required to trigger crowd alert
CROWD_PERSIST_FRAMES = 5       # Frames crowd must persist
```

---

## 🔭 Project Roadmap

We are moving from a basic prototype to a multi-stage production deployment. **Please see [ROADMAP.md](./ROADMAP.md) for the complete phase-by-phase execution plan**, including:
1. Data collection & hard-negative datasets.
2. Free cloud GPU training strategies (Kaggle P100).
3. Exporting to ONNX and integrating the 2-stage pipeline.
4. Building a React + Tailwind website.
5. Production deployment (Docker, RTSP IP cameras).
