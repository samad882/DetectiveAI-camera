# 🤖 SentinelAI — Custom AI Model Training Guide

> **Goal:** Train a single YOLO11 model that detects all 6 classes:  
> `pistol` · `knife` · `rifle` · `person` · `bag` · `fight`

---

## 📋 What We Are Building

```
Current model (best.onnx):          New custom model:
─────────────────────────           ──────────────────────────────
Classes: pistol, knife              Classes: pistol, knife, rifle,
Accuracy: medium                             person, bag, fight
Training data: unknown              Accuracy: HIGH (custom trained)
Can't detect crowds/bags            Detects ALL 6 threat types
```

---

## 🗓️ Timeline

```
Day 1 → Collect datasets
Day 2 → Label + organize on Roboflow  
Day 3 → Train on Google Colab (3-4 hrs)
Day 4 → Test + evaluate
Day 5 → Export + plug into SentinelAI ✅
```

---

## ✅ STEP 1 — Collect Your Dataset

You need images for all 6 classes. Use these **free sources**:

---

### 🔫 Class 1, 2, 3 — Pistol + Knife + Rifle

> Download **BOTH** — merge them on Roboflow for best coverage.

**Dataset 1 — Primary (has Person class too!)**
| | |
|--|--|
| 🔗 Link | [weapon-detection by maheshchhetri](https://universe.roboflow.com/maheshchhetri/weapon-detection-e6otc) |
| 📸 Images | 1,870 |
| 🏷️ Classes | handgun, pistol, rifle, knife, **Person** |
| ✅ Why | Best all-rounder — covers weapons + person |

**Dataset 2 — Merge for more knife variety**
| | |
|--|--|
| 🔗 Link | [weapon-detection by buildx](https://universe.roboflow.com/buildx/weapon-detection-7kro8) |
| 📸 Images | 9,520 |
| 🏷️ Classes | knife, rifle, ak, cleaver, cutter, ax... |
| ✅ Why | Biggest dataset — great blade/knife diversity |

> **How to download:** Open link → Click **"Download Dataset"** → Format: **YOLOv8** → Download ZIP

> **After download — rename these classes on Roboflow before exporting:**
> `handgun` → `pistol` | `ak` → `rifle` | `cleaver/ax/cutter` → `knife`

---

### 👥 Class 4 — Person (Crowd Detection)

> ✅ Already included in Dataset 1 above (maheshchhetri has `Person` class) — no separate download needed!

---

### 🎒 Class 5 — Bag (Unattended Bag)

**Dataset — Baggage by Shudarshan Kongkham**
| | |
|--|--|
| 🔗 Link | [Baggage by Shudarshan Kongkham](https://universe.roboflow.com/shudarshan-kongkham/baggage-jowci) |
| 📸 Images | 3,450 |
| 🏷️ Classes | bag, suitcase, baggage, luggage |
| ✅ Why | Most images, clean labels, real surveillance shots |

> **Download:** Open link → **Download Dataset** → Format: **YOLOv8** → ZIP
> Rename all → `bag` (baggage / luggage / suitcase → all become `bag`)

---

### 🥊 Class 6 — Fight / Aggression

**Dataset — Fight by BBC**
| | |
|--|--|
| 🔗 Link | [Fight by BBC](https://universe.roboflow.com/bbc-4k9ak/fight-n5va8) |
| 📸 Images | 5,790 |
| 🏷️ Classes | 0, 1 |
| ✅ Why | Biggest fight dataset — real surveillance footage |

> **Download:** Open link → **Download Dataset** → Format: **YOLOv8** → ZIP
> **Important:** Rename class `1` → `fight` and delete/ignore class `0` (non-fight) on Roboflow before exporting


---

### 📊 Minimum Images Per Class

| Class   | Minimum | Good  | Excellent |
|---------|---------|-------|-----------|
| pistol  | 300     | 800   | 2000+     |
| knife   | 300     | 800   | 2000+     |
| rifle   | 200     | 500   | 1500+     |
| person  | 500     | 1500  | 5000+     |
| bag     | 300     | 800   | 2000+     |
| fight   | 200     | 500   | 1000+     |

> **Rule: More images = higher accuracy. Don't skip this step.**

---

## ✅ STEP 2 — Organize Data on Roboflow

```
Go to: https://roboflow.com  (free account)

1. New Project → Object Detection
2. Name: "SentinelAI-v1"
3. Upload all images (drag and drop)
4. Label images:
     - Draw tight boxes around each object
     - Assign class: pistol / knife / rifle / person / bag / fight
5. Add Augmentation:
     ✅ Flip (horizontal)
     ✅ Rotation -15° to +15°
     ✅ Brightness -25% to +25%
     ✅ Blur 0 to 2px (CCTV simulation)
     ✅ Grayscale 15% (B&W cameras)
     ✅ Noise 0.5%
6. Export → YOLOv8 format → Download ZIP
7. Extract to: AI_Model/dataset/
```

---

## ✅ STEP 3 — Setup data.yaml

```yaml
# AI_Model/dataset/data.yaml

path: /content/dataset
train: train/images
val: valid/images
test: test/images

nc: 6

names:
  0: pistol
  1: knife
  2: rifle
  3: person
  4: bag
  5: fight
```

---

## ✅ STEP 4 — Train on Google Colab (FREE GPU)

```
Go to: https://colab.research.google.com
New Notebook → Runtime → Change Runtime → T4 GPU
```

### CELL 1 — Install
```python
!pip install ultralytics -q
```

### CELL 2 — Upload Dataset
```python
from google.colab import files
import zipfile

uploaded = files.upload()   # upload your dataset.zip
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('/content/dataset')
print("Dataset ready!")
```

### CELL 3 — Train
```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")   # medium — best balance

results = model.train(
    data="/content/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    name="sentinelai_v1",
    project="/content/runs",
    mosaic=1.0,
    mixup=0.1,
    degrees=10.0,
)
print("Training complete!")
```

### CELL 4 — Evaluate
```python
best_model = YOLO("/content/runs/sentinelai_v1/weights/best.pt")
metrics = best_model.val(data="/content/dataset/data.yaml")

print(f"mAP@50:    {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")
```

### CELL 5 — Export to ONNX
```python
best_model.export(format="onnx", imgsz=640, simplify=True, opset=17)
```

### CELL 6 — Download
```python
from google.colab import files
files.download("/content/runs/sentinelai_v1/weights/best.onnx")
files.download("/content/runs/sentinelai_v1/results.png")
```

---

## ✅ STEP 5 — Read Your Results

```
mAP@50 Score Guide:
  < 0.50   → Bad    — need more data or epochs
  0.50-0.70 → Okay  — decent, can improve
  0.70-0.85 → Good ✅ — ready to use
  0.85-0.95 → Excellent 🔥
  > 0.95   → Exceptional
```

### If accuracy is low:
```
mAP < 0.50?
  ✅ Add more images (most impactful fix)
  ✅ Increase epochs to 200
  ✅ Check label quality
  ✅ Switch to yolo11l (larger model)

Out of memory?
  ✅ Reduce batch to 8 or 4
  ✅ Reduce imgsz to 416
```

---

## ✅ STEP 6 — Plug Into SentinelAI

```bash
# Replace old model with your new trained model:
cp ~/Downloads/best.onnx ../models/best.onnx
```

Done! The app auto-reads class names from the model.
`rules.py` will automatically enable all 6 rules:

```
pistol + knife + rifle  →  WEAPON alert rule ✅
person                  →  CROWD alert rule  ✅
bag                     →  UNATTENDED BAG rule ✅
fight                   →  ALTERCATION rule  ✅
```

---

## 📁 This Folder Structure

```
AI_Model/
├── README.md              ← This file — follow step by step
├── dataset/               ← Put your Roboflow downloaded dataset here
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── train_colab.ipynb  ← Save your Colab notebook here
└── exports/
    └── best.onnx          ← Your trained model → copy to ../models/
```

---

## 🔗 Quick Links

| Resource | URL |
|----------|-----|
| Roboflow Universe | https://universe.roboflow.com |
| Google Colab | https://colab.research.google.com |
| YOLOv8 Docs | https://docs.ultralytics.com |
| Kaggle Datasets | https://www.kaggle.com/datasets |
| COCO Dataset | https://cocodataset.org |
| RWF-2000 Fight Dataset | https://arxiv.org/abs/1911.05913 |
| UCF Crime Dataset | https://www.crcv.ucf.edu/research/real-world-anomaly-detection/ |

---

## ⚠️ Common Mistakes to Avoid

```
❌ Too few images (< 200 per class) → poor accuracy
❌ Inconsistent labels (too loose/tight boxes)
❌ Wrong class names in data.yaml
❌ No augmentation → model won't generalize to CCTV
❌ Too few epochs (< 50)
❌ Not testing on a separate test set
```

---

## ✅ Pre-Training Checklist

```
□ 300+ images per class collected
□ All images labeled on Roboflow
□ Augmentation enabled (flip, brightness, blur)
□ data.yaml paths + class names correct
□ Google Colab on T4 GPU runtime
□ Dataset ZIP uploaded to Colab
□ Ready to train! 🚀
```

---

## 💻 STEP 7 — Local Training Script (RTX 3050 / Any NVIDIA GPU)

> Copy the script below, save as `train_local.py` in your **project root**, and run it on your Windows laptop.

**Setup first (run once on Windows laptop):**
```bash
# 1. Clone repo
git clone https://github.com/YOUR/DetectiveAI-camera.git
cd DetectiveAI-camera

# 2. Virtual env
python -m venv .venv
.venv\Scripts\activate

# 3. PyTorch WITH CUDA (must use this link!)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics supervision

# 4. Verify GPU detected
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: GPU: True NVIDIA GeForce RTX 3050
```

**Run training:**
```bash
python train_local.py
```

---

### `train_local.py` — Full Script

```python
"""
SentinelAI — Local Training Script (RTX 3050)
Run: python train_local.py
Time: ~3-4 hours on RTX 3050
"""
import torch, time, shutil
from pathlib import Path

# ── Verify GPU ─────────────────────────────────────────────────
print("=" * 60)
print("  SentinelAI Model Trainer")
print("=" * 60)

if torch.cuda.is_available():
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = 0
else:
    print("  No GPU — install: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    DEVICE = "cpu"
print("=" * 60)

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "AI_Model" / "dataset" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "AI_Model" / "runs"

if not DATASET_YAML.exists():
    print(f"ERROR: data.yaml not found at {DATASET_YAML}")
    print("→ Download from Roboflow → extract to AI_Model/dataset/")
    exit(1)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
print(f"  Dataset : Found at {DATASET_YAML}")

# ── Load Model ─────────────────────────────────────────────────
from ultralytics import YOLO

# RTX 3050 (4GB VRAM) options:
#   yolo11n  ~ 1 hr   fastest, less accurate
#   yolo11s  ~ 2 hr
#   yolo11m  ~ 3-4 hr  ← RECOMMENDED
#   yolo11l  ~ 8 hr   (may OOM on 4GB VRAM)
MODEL_SIZE = "yolo11m.pt"
model = YOLO(MODEL_SIZE)
print(f"  Model   : {MODEL_SIZE} loaded")

# ── Train ──────────────────────────────────────────────────────
print("\n  Training started. ~3-4 hrs on RTX 3050. Do not close.\n")
start = time.time()

model.train(
    data=str(DATASET_YAML),
    epochs=100,
    imgsz=640,
    batch=8,           # 4GB VRAM → 8. OOM error? → reduce to 4
    device=DEVICE,
    workers=4,
    patience=20,
    lr0=0.01,
    weight_decay=0.0005,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    name="sentinelai_v1",
    project=str(RUNS_DIR),
    exist_ok=True,
    verbose=True,
)
print(f"\n  Training done in {(time.time()-start)/3600:.1f} hours")

# ── Evaluate ───────────────────────────────────────────────────
print("\n  Evaluating accuracy...")
best_pt    = RUNS_DIR / "sentinelai_v1" / "weights" / "best.pt"
best_model = YOLO(str(best_pt))
metrics    = best_model.val(data=str(DATASET_YAML))

map50 = metrics.box.map50
print(f"\n  mAP@50    : {map50:.3f}")
print(f"  Precision : {metrics.box.mp:.3f}")
print(f"  Recall    : {metrics.box.mr:.3f}")

if map50 >= 0.75:   print("  Verdict   : Excellent — production ready!")
elif map50 >= 0.55: print("  Verdict   : Good — add more data for higher accuracy")
else:               print("  Verdict   : Low — add more images and retrain")

# ── Export ONNX ────────────────────────────────────────────────
print("\n  Exporting to ONNX...")
best_model.export(format="onnx", imgsz=640, simplify=True, opset=17)

# ── Auto-replace in project ────────────────────────────────────
onnx_src    = RUNS_DIR / "sentinelai_v1" / "weights" / "best.onnx"
onnx_models = PROJECT_ROOT / "models" / "best.onnx"

shutil.copy2(onnx_src, onnx_models)
print(f"  Replaced : models/best.onnx")

print("""
  DONE!
  git add models/best.onnx
  git commit -m "trained model v1"
  git push   →   git pull on Mac   →   app updated!
""")
```

---

## ⏱️ Training Time Reference (RTX 3050)

| Model | Dataset | Epochs | Time |
|-------|---------|--------|------|
| YOLO11n | 2000 imgs | 100 | ~1 hour |
| **YOLO11m** | **2000 imgs** | **100** | **~3-4 hours ✅ Recommended** |
| YOLO11m | 5000 imgs | 100 | ~6-8 hours |
| YOLO11l | 2000 imgs | 100 | ~8 hours (may OOM on 4GB) |

---

## 📁 This Folder

```
AI_Model/
└── README.md    ← This file — complete guide + training script inside
```

> Dataset, runs folders are created automatically when training starts.
> Only copy `models/best.onnx` back to Mac via git push/pull.
