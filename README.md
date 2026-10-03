# 🛡️ DetectiveAI
### *AI-Powered Real-Time Surveillance Intelligence*

> **Turning dumb cameras into smart security systems — detecting threats before they escalate.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11m-00FFAA?style=for-the-badge)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge)](https://onnxruntime.ai)

---

## 📑 Slide 1 — The Problem

> **1 Billion+ CCTV cameras are installed worldwide. Less than 5% have AI.**

The rest are just recording — not watching.

| Problem | Impact |
|---------|--------|
| 👁️ Humans can't watch 100s of feeds at once | Threats go unnoticed until it's too late |
| ⏰ CCTV is reviewed AFTER an incident | Zero prevention, only evidence collection |
| 🔔 Motion-based systems trigger false alarms constantly | Alert fatigue — security teams stop paying attention |
| 💸 24/7 human monitoring is expensive | A single security operations center costs crores annually |
| 🌙 Night / low-light footage is missed | Most incidents happen when visibility is poor |

**The gap: Smart AI that watches, understands, and alerts in real time.**

---

## 📑 Slide 2 — Our Solution

**DetectiveAI is a software layer that makes any existing CCTV camera intelligent.**

No new cameras needed. No hardware upgrades. Just plug in the AI.

```
Existing Camera  →  DetectiveAI AI Engine  →  Instant Alert + Live Dashboard
```

| What We Do | How |
|------------|-----|
| Detect weapons, fights, bags, crowds | Custom-trained YOLOv11m AI model |
| Track every person/object across frames | DeepSORT persistent tracking |
| Avoid false alarms intelligently | Smart multi-frame rules engine |
| Show everything live | Streamlit real-time dashboard |
| Work on any existing hardware | CPU + GPU both supported |

---

## 📑 Slide 3 — Features

| Feature | What It Does |
|---------|-------------|
| 🔫 **Weapon Detection** | Detects guns, pistols, rifles, knives in live CCTV footage |
| 🥊 **Fight Detection** | Identifies physical altercations and aggressive behavior |
| 🎒 **Unattended Bag Alert** | Flags bags left alone without owner for configurable time |
| 👥 **Crowd Monitoring** | Triggers alert when headcount exceeds safe limit |
| 📊 **Live Dashboard** | Real-time video with bounding boxes, labels, and alerts |
| ⚙️ **Adjustable Sensitivity** | Security team can tune thresholds without touching code |
| 🚫 **Anti-False-Alarm Logic** | Threat must appear in multiple consecutive frames to trigger |
| 🌙 **Low-Light Ready** | Trained on real CCTV footage — blurry, dark, distant scenes |
| 📁 **Works with Any Input** | Video file, live webcam, or RTSP IP camera stream |

---

## 📑 Slide 4 — Use Cases

| Scenario | DetectiveAI Response |
|----------|-------------------|
| 🔫 Someone pulls out a weapon in a corridor | Alert fires within 100ms to security team |
| 🥊 Two people start fighting on campus | Detected within seconds of physical contact |
| 🎒 Bag left unattended near a gate | Alert after configurable wait (30 sec to 5 min) |
| 👥 Crowd surging at a stadium exit | Instant warning to prevent stampede |
| 🌙 Suspicious activity at 3 AM | AI keeps watching even when no human is |
| 📷 Limited security staff, many cameras | One dashboard monitors all feeds simultaneously |
| 🏛️ VIP / restricted area access | Alert when unauthorized person detected in zone |

---

## 📑 Slide 5 — Target Organizations

| Organization | Deployment Areas | Key Benefit |
|---|---|---|
| 🏛️ **Government Buildings** | Parliament, courts, offices, police stations | Proactive threat detection for sensitive zones |
| 🎓 **Colleges & Universities** | Campus gates, hostels, labs, parking | Student safety — weapon + fight detection |
| 🏫 **Schools** | Main gates, hallways, cafeteria | Early threat detection in student-heavy areas |
| 🚉 **Railways & Metro** | Platforms, concourses, staircases | Unattended bag + crowd surge alerts |
| ✈️ **Airports** | Security queues, lounges, parking | Multi-threat monitoring at scale |
| 🏥 **Hospitals** | Emergency ward, pharmacy, parking | After-hours security with no extra staff |
| 🏟️ **Stadiums & Events** | Entrances, stands, VIP zones | Real-time crowd management |
| 🏪 **Retail & Malls** | Stores, food courts, parking | Theft and violence prevention |
| 🏢 **Corporate Offices** | Reception, server rooms, restricted zones | Access control and perimeter security |

---

## 📑 Slide 6 — Tech Stack

| Technology | What It Does in DetectiveAI |
|---|---|
| **YOLOv11m** | Core AI model — detects guns, people, bags, knives per frame |
| **ONNX Runtime** | Runs the model in production — fast, lightweight, hardware-agnostic |
| **OpenCV** | Reads frames from video file, webcam, or RTSP stream |
| **DeepSORT** | Gives every detected object a persistent ID across frames |
| **Rules Engine** | Custom Python logic — decides when an alert actually fires |
| **Streamlit** | Live browser dashboard — video + alerts + stats in real time |
| **PyTorch** | Used during AI model training (not needed at runtime) |
| **Roboflow** | Dataset platform — organized and versioned 38K CCTV images |
| **Kaggle T4 GPU** | Cloud GPU used to train the model (15GB VRAM, free) |
| **NumPy / Pillow** | Frame manipulation and image preprocessing utilities |

---

## 📑 Slide 7 — Roadmap & Vision

### What We Have Built (Today)
| Status | Capability |
|--------|-----------|
| ✅ Done | Weapon Detection (guns + knives) |
| ✅ Done | Fight / Altercation Detection |
| ✅ Done | Unattended Bag Alert |
| ✅ Done | Crowd Monitoring |
| ✅ Done | Live Streamlit Dashboard |
| ✅ Done | Custom AI trained on 38K real CCTV images |

### What's Next
| Phase | What's Coming | Timeline |
|-------|--------------|----------|
| 🔄 **Phase 2** | FastAPI backend + React web dashboard | Q1 2025 |
| 🔄 **Phase 2** | Multi-camera grid view | Q1 2025 |
| 🔄 **Phase 2** | SMS / WhatsApp instant alerts | Q1 2025 |
| 🔮 **Phase 3** | Mobile app for security teams | Q2 2025 |
| 🔮 **Phase 3** | LLM-powered incident summary reports | Q2 2025 |
| 🔮 **Phase 3** | License plate & face recognition | Q2 2025 |
| 🔮 **Phase 3** | Cloud SaaS — deploy for any organization | Q3 2025 |

---

<div align="center">

### 🚀 Quick Start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install ultralytics streamlit opencv-python-headless numpy \
               deep-sort-realtime torch pillow onnxruntime "setuptools<70" onnx
streamlit run src/streamlit_app.py
```

Open **`http://localhost:8501`**

---

**Built to make every camera smarter. Built for a safer world.**

*DetectiveAI — See Everything. Miss Nothing.*

</div>
