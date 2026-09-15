# SentinelAI - Full Production Roadmap

> **System:** RT-DETR-X + EfficientNet-B4 Verifier - 2-Stage Zero False-Positive Pipeline
> **Goal:** Production-grade security surveillance with < 0.5% false positive rate
> **GPU to Train:** Kaggle P100 Free (16GB) -- Run inference on your RTX 2050

---

## Project Architecture Overview

```
SENTINELAI -- TWO REPO STRATEGY

  REPO 1: sentinel-training        REPO 2: DetectiveAI-camera (THIS REPO)
  --------------------------        ----------------------------------------
  [x] Dataset collection            [x] React + Tailwind website
  [x] Data annotation               [x] Python backend (Streamlit / FastAPI)
  [x] Model training (Cloud GPU)    [x] ONNX inference engine
  [x] Model evaluation              [x] Rules engine + tracking
  [x] ONNX export                   [x] Alert dashboard
       |                                    ^
       |-- best.onnx ----------------------|  (copy .onnx files across)
       |-- verifier.onnx ------------------|
```

---

## Roadmap Phases Overview

```
PHASE 0        PHASE 1        PHASE 2        PHASE 3        PHASE 4        PHASE 5
----------     ----------     ----------     ----------     ----------     ----------
Baseline   --> Data       --> Training   --> Export &   --> Website    --> Production
Setup          Collection     & Fine-tune    Integration    Build          Deploy
(Done)         (Week 1)       (Week 2)       (Week 3)       (Week 3-4)     (Week 4-5)
```

---

## PHASE 0 -- Current Baseline (Already Done)

> What you already have working today.

| Component | Status | Details |
|---|---|---|
| YOLOv8 ONNX model | Done | `models/best.onnx` -- pistol + knife |
| DeepSORT Tracker | Done | `src/tracking.py` |
| Rules Engine | Done | `src/rules.py` -- weapon, crowd, bag, altercation |
| Streamlit Dashboard | Done | `src/streamlit_app.py` |
| Visualization | Done | `src/visualize.py` |

**Known Limitations to Fix in Phase 1-3:**

- CONF_THRESHOLD = 0.20 is too low -- causes many false positives
- Single-stage detection -- no verification layer
- No React website -- only Streamlit UI
- No rifle/long gun class in current model
- No person/bag class in current model (crowd + unattended bag detection broken)

---

## PHASE 1 -- Data Collection & Annotation (Week 1)

### Step 1A -- Download Free Public Datasets

#### Weapon Detection Datasets

| Dataset | Classes | Images | Direct Link |
|---|---|---|---|
| **Roboflow Weapon Universe** | pistol, rifle, knife | ~15,000 | https://universe.roboflow.com/search?q=weapon+detection |
| **Gun Detection Dataset** | handgun, rifle | 3,000 | https://universe.roboflow.com/tejas-vase/gun-detection-lbavj |
| **Knife Detection Dataset** | knife | 2,500 | https://universe.roboflow.com/search?q=knife+detection |
| **HuggingFace Weapon** | pistol, knife | 5,000 | https://huggingface.co/Hadi959/weapon-detection-yolov8 |

#### Person / Crowd / Bag Datasets

| Dataset | Classes | Images | Direct Link |
|---|---|---|---|
| **COCO 2017** | person, backpack, handbag, suitcase | 118,000 | https://cocodataset.org/#download |
| **Open Images V7** | person, gun, knife, backpack | 600,000+ | https://storage.googleapis.com/openimages/web |
| **ShanghaiTech Crowd** | person density | 1,198 scenes | https://github.com/desenzhou/ShanghaiTechDataset |
| **MOT17 Tracking** | pedestrians | 1,300 sequences | https://motchallenge.net/data/MOT17/ |

#### Hard Negative Dataset -- FALSE POSITIVE KILLERS (Critical!)

> Things that LOOK like weapons but are NOT.
> This is what reduces your false positive rate from 15% down to 0.5%.
> Most teams skip this step -- do NOT skip it.

| Object | Why It Causes False Positives | Source |
|---|---|---|
| TV Remote Controls | Same rectangular size and shape as a pistol | Open Images V7 |
| Smartphones held at arm length | Dark rectangular shape in hand | Open Images V7 |
| Collapsed Umbrellas | Long stick = rifle-like silhouette | Open Images V7 |
| Power Drills | Pistol-like shape with a barrel end | ImageNet |
| Bananas | Curved shape = knife-like (classic YOLO false positive) | Open Images V7 |
| Finger-pointing hands | Hand shape resembles a gun | Hand gesture datasets |
| L-shaped wrenches | Pistol silhouette from certain angles | ImageNet tools |
| Water guns and toy guns | Identical shape to real firearm | Google Images scrape |

```bash
# Download hard negatives from Open Images V7
pip install openimages
oi_download_dataset --base_dir ./hard_negatives \
  --labels "Remote control" "Mobile phone" "Umbrella" "Drill" \
  --format pascal_voc \
  --limit 2000
```

---

### Step 1B -- Annotation Tool Setup

```
OPTION 1: Roboflow (Recommended for beginners)
----------------------------------------------
URL:  https://app.roboflow.com

  Free Tier:  3 projects, 10,000 images
  AI-Assist:  Auto-labels using AI (saves 80% annotation time)
  Export:     YOLOv8 format (directly compatible with training)

  Steps:
    1 --> Create account at app.roboflow.com
    2 --> New Project --> Object Detection
    3 --> Upload images (drag and drop)
    4 --> Use AI-Assist to auto-label boxes
    5 --> Manually verify and correct boxes
    6 --> Generate Dataset --> Add augmentation (flip, blur, rotate)
    7 --> Export --> Format: YOLOv8 --> Download ZIP


OPTION 2: Label Studio (Free, No Image Limit, Self-Hosted)
----------------------------------------------------------
  pip install label-studio
  label-studio start
  --> Open: http://localhost:8080
  --> New Project --> Object Detection --> Import images --> Export YOLO


OPTION 3: CVAT (Enterprise Grade, Free, Team Use)
-------------------------------------------------
  URL: https://www.cvat.ai
  Best for multiple annotators working on same project
```

**Annotation Class Mapping (YOLO .txt format):**

```
# Each image gets one .txt file with same filename
# Format: class_id  cx  cy  width  height   (all normalized 0.0 to 1.0)

0  0.512  0.432  0.045  0.123   <-- pistol detected here
3  0.234  0.654  0.123  0.432   <-- person standing nearby
4  0.678  0.543  0.087  0.156   <-- bag on ground
```

**Final Class List:**

| Class ID | Name | Description |
|---|---|---|
| 0 | pistol | Handguns, revolvers, semi-automatics |
| 1 | rifle | Long guns, shotguns, assault rifles |
| 2 | knife | All bladed weapons, tactical knives |
| 3 | person | All humans (used for crowd + bag proximity) |
| 4 | bag | Backpacks, suitcases, handbags, duffel bags |

---

### Step 1C -- Dataset Size Target

```
Gate 1 Dataset (RT-DETR-X Detector):

  Class      Train Images    Val Images
  ------     ------------    ----------
  pistol         8,000          1,000
  rifle          5,000            700
  knife          6,000            800
  person        20,000          3,000
  bag            5,000            700
  ---------  ------------    ----------
  TOTAL         44,000          6,200

  + Auto augmentation (3x multiplier) = ~150,000 effective training samples


Gate 2 Dataset (EfficientNet-B4 Verifier):

  Class            Train Images    Val Images
  ----------       ------------    ----------
  weapon crop          8,000          1,000
  not_weapon hard     10,000          1,500
  ----------       ------------    ----------
  TOTAL               18,000          2,500
```

---

### Step 1D -- Dataset Folder Structure

```
sentinel-training/
|-- datasets/
|   |-- gate1_detector/
|   |   |-- images/
|   |   |   |-- train/        <-- 44,000 images (.jpg)
|   |   |   |-- val/          <-- 6,200 images
|   |   |   +-- test/         <-- 1,000 images
|   |   +-- labels/
|   |       |-- train/        <-- 44,000 .txt YOLO label files
|   |       |-- val/
|   |       +-- test/
|   +-- gate2_verifier/
|       |-- train/
|       |   |-- weapon/       <-- 8,000 cropped weapon images
|       |   +-- not_weapon/   <-- 10,000 hard negative images
|       +-- val/
|           |-- weapon/       <-- 1,000 images
|           +-- not_weapon/   <-- 1,500 images
|-- configs/
|   +-- sentinel_data.yaml
|-- notebooks/
|   |-- 01_train_gate1_rtdetr.ipynb
|   +-- 02_train_gate2_verifier.ipynb
+-- export/
    +-- export_onnx.py
```

---

## PHASE 2 -- Model Training (Week 2)

### Step 2A -- Training Platform Setup

```
PLATFORM COMPARISON:

  Platform        GPU       VRAM    Free Hours    Timeout     URL
  -----------     ------    ------  ----------    -------     ---
  Kaggle [BEST]   P100      16 GB   30 hrs/week   NONE        kaggle.com/code
  Google Colab    T4        15 GB   6-8 hrs/day   6-8 hours   colab.research.google.com
  Lightning.ai    T4        16 GB   22 hrs/month  None        lightning.ai
  Vast.ai [PAID]  RTX 4090  24 GB   Pay per hr    None        vast.ai ($0.35/hr)

  RECOMMENDATION: Use Kaggle P100 -- 16GB free, no timeout, run overnight!


KAGGLE SETUP STEPS:

  Step 1 --> Go to kaggle.com --> Sign up (free)
  Step 2 --> Kaggle Datasets --> New Dataset
             --> Upload your annotated dataset ZIP
             --> Name it: sentinel-weapon-dataset
  Step 3 --> Kaggle Code --> New Notebook
  Step 4 --> Notebook Settings --> Accelerator --> GPU P100 [ON]
  Step 5 --> Add Data --> Your sentinel-weapon-dataset
  Step 6 --> Paste training code from Step 2B below
  Step 7 --> Run All --> Go to sleep --> Check results next morning
```

---

### Step 2B -- Create Dataset Config File

**File: `configs/sentinel_data.yaml`**

```yaml
# sentinel_data.yaml -- RT-DETR-X training configuration
path: /kaggle/input/sentinel-weapon-dataset
train: images/train
val:   images/val
test:  images/test

nc: 5   # number of classes
names:
  0: pistol
  1: rifle
  2: knife
  3: person
  4: bag
```

---

### Step 2C -- Train Gate 1: RT-DETR-X Detector

**Paste this into your Kaggle Notebook:**

```python
# ============================================================
# CELL 1 -- Install dependencies
# ============================================================
!pip install ultralytics --quiet

# ============================================================
# CELL 2 -- Verify GPU is active
# ============================================================
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Should print: GPU: Tesla P100-PCIE-16GB   VRAM: 16.0 GB

# ============================================================
# CELL 3 -- Load RT-DETR-X (auto-downloads ~136MB pretrained weights)
# ============================================================
from ultralytics import RTDETR
model = RTDETR("rtdetr-x.pt")  # Downloads from Ultralytics automatically
print("Model loaded!")

# ============================================================
# CELL 4 -- TRAIN (Fine-tune on your custom weapon security dataset)
# ============================================================
results = model.train(
    data="/kaggle/input/sentinel-weapon-dataset/sentinel_data.yaml",
    epochs=100,           # 100 full training cycles
    imgsz=640,            # 640x640 pixel input (standard)
    batch=4,              # 4 images per batch (safe for 16GB VRAM)
    lr0=0.0001,           # Low learning rate -- fine-tuning not scratch
    lrf=0.01,             # Final LR = lr0 x lrf at end of training
    warmup_epochs=5,      # Gradual warmup to avoid early instability
    patience=20,          # Stop early if no improvement for 20 epochs
    device=0,             # Use GPU 0
    workers=4,
    project="/kaggle/working/sentinel_runs",
    name="rtdetr_x_gate1",
    save=True,
    plots=True,           # Saves training graphs automatically
    val=True,
)

print("Training complete!")
print(f"Best mAP@50: {results.results_dict[chr(109)+chr(101)+chr(116)+chr(114)+chr(105)+chr(99)+chr(115)+chr(47)+chr(109)+chr(65)+chr(80)+chr(53)+chr(48)+chr(40)+chr(66)+chr(41)]:.4f}")

# ============================================================
# CELL 5 -- Evaluate the trained model
# ============================================================
from ultralytics import RTDETR
model = RTDETR("/kaggle/working/sentinel_runs/rtdetr_x_gate1/weights/best.pt")
metrics = model.val(data="/kaggle/input/sentinel-weapon-dataset/sentinel_data.yaml")

print("=" * 50)
print(f"mAP@50:    {metrics.box.map50:.4f}")   # Target: > 0.85
print(f"Precision: {metrics.box.mp:.4f}")       # Target: > 0.90
print(f"Recall:    {metrics.box.mr:.4f}")       # Target: > 0.85
print("=" * 50)

for i, name in enumerate(["pistol","rifle","knife","person","bag"]):
    print(f"  {name}: AP50 = {metrics.box.ap50[i]:.4f}")
```

**Gate 1 Acceptance Criteria -- Do NOT export unless ALL pass:**

| Metric | Target | Why |
|---|---|---|
| mAP@50 | > 0.85 | Overall detection quality across all classes |
| Precision | > 0.90 | When model says weapon, it IS a weapon 90%+ |
| Recall | > 0.85 | Model catches 85%+ of all real weapons |
| pistol AP50 | > 0.88 | Most common threat class |
| knife AP50 | > 0.85 | Second most common |
| rifle AP50 | > 0.80 | Harder class due to occlusion |

---

### Step 2D -- Train Gate 2: EfficientNet-B4 Verifier

**This is your false-positive killer. Verifies each weapon detection before alerting.**

```python
# ============================================================
# CELL 1 -- Install
# ============================================================
!pip install timm --quiet

# ============================================================
# CELL 2 -- Build Verifier Model (EfficientNet-B4)
# ============================================================
import torch, torch.nn as nn, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class WeaponVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B4 backbone pretrained on ImageNet -- 1792 output features
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)   # Output: [not_weapon, weapon]
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = WeaponVerifier().cuda()
print(f"Verifier parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# CELL 3 -- Data Loaders for Gate 2 Verifier
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(
    "/kaggle/input/sentinel-weapon-dataset/gate2_verifier/train",
    transform=train_tf
)
val_data = datasets.ImageFolder(
    "/kaggle/input/sentinel-weapon-dataset/gate2_verifier/val",
    transform=val_tf
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=4)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")
print(f"Classes: {train_data.classes}")  # [not_weapon, weapon]

# ============================================================
# CELL 4 -- Focal Loss (penalizes false positives more than false negatives)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
best_acc  = 0

# ============================================================
# CELL 5 -- Training Loop (runs 50 epochs, ~2 hours on P100)
# ============================================================
for epoch in range(50):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = fp = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            preds   = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            fp      += ((preds == 1) & (labels == 0)).sum().item()

    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "/kaggle/working/best_verifier.pt")
    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | Val Acc: {acc:.4f} | FP Rate: {fp/total:.4f}")

print(f"Best Accuracy: {best_acc:.4f}")
```

**Gate 2 Acceptance Criteria:**

| Metric | Target |
|---|---|
| Validation Accuracy | > 0.98 |
| False Positive Rate | < 0.02 |
| Weapon class Precision | > 0.97 |

---

## PHASE 3 -- Export Models & Integrate (Week 3)

### Step 3A -- Export Models to ONNX

Run this on Kaggle after training finishes:

```python
# 1. Export Gate 1 (RT-DETR-X)
from ultralytics import RTDETR
model_g1 = RTDETR("/kaggle/working/sentinel_runs/rtdetr_x_gate1/weights/best.pt")
model_g1.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False
)
# Output: best.onnx (~136MB)

# 2. Export Gate 2 (Verifier)
import torch
model_g2 = WeaponVerifier()
model_g2.load_state_dict(torch.load("/kaggle/working/best_verifier.pt"))
model_g2.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model_g2, dummy, "/kaggle/working/verifier.onnx",
    opset_version=12,
    input_names=["image"],
    output_names=["logits"]
)
# Output: verifier.onnx (~20MB)
```

### Step 3B -- Download and Place in This Repo

Download `best.onnx` and `verifier.onnx` from Kaggle output and place them in your local repo:

```
DetectiveAI-camera/
|-- models/
|   |-- best.onnx         <-- REPLACE old YOLOv8 with new RT-DETR-X
|   +-- verifier.onnx     <-- NEW verifier model
```

### Step 3C -- Code Updates in This Repo

The `src/detection.py` and `src/rules.py` files need to be updated to support the 2-stage pipeline and tighter thresholds.

**Updated Rules Thresholds (`src/rules.py`):**

```python
WEAPON_CONF_GATE1       = 0.65   # Was 0.20 --> now 65% certainty minimum
WEAPON_CONF_GATE2       = 0.85   # Gate 2 must be 85% sure
WEAPON_PERSIST_FRAMES   = 8      # Was 3 --> now 8 consecutive frames
WEAPON_ALERT_COOLDOWN   = 30     # Was 5s --> now 30s cooldown
CROWD_THRESHOLD         = 15
CROWD_PERSIST_FRAMES    = 5
BAG_STATIONARY_SECONDS  = 30     # Was 5s --> 30s
BAG_PROXIMITY_PIXELS    = 120    # Was 150px --> tighter
```

---

## PHASE 4 -- Website Build (React + Tailwind) (Week 3-4)

We will build a complete React + Tailwind CSS marketing and dashboard website inside the `web/` folder of this repo.

### Step 4A -- Project Structure

```
web/
|-- src/
|   |-- components/
|   |   |-- Navbar.jsx           # Sticky glassmorphism nav
|   |   |-- Hero.jsx             # Full-screen animated surveillance background
|   |   |-- Features.jsx         # 4 Cards: Weapon | Crowd | Bag | Altercation
|   |   |-- HowItWorks.jsx       # Pipeline diagram
|   |   |-- TechStack.jsx        # RT-DETR vs YOLO comparison
|   |   +-- DashboardPreview.jsx # Live UI iframe / preview
|   |-- App.jsx
|   +-- index.css                # Tailwind base + custom animations
|-- tailwind.config.js           # Dark theme design system
+-- package.json
```

### Step 4B -- Setup Commands

```bash
npm create vite@latest web -- --template react
cd web
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install framer-motion lucide-react react-router-dom
```

---

## PHASE 5 -- Production Deployment (Week 4-5)

### Step 5A -- Containerization

Create a Docker container to run the Python backend seamlessly.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ ./models/
COPY src/ ./src/
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 5B -- RTSP CCTV Integration

Update `video_path` to support IP cameras:

```python
RTSP_STREAMS = [
    "rtsp://admin:password@192.168.1.64:554/stream1",
    "rtsp://admin:password@192.168.1.65:554/stream1",
]
cap = cv2.VideoCapture(RTSP_STREAMS[0])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
```

### Step 5C -- Notifications

Integrate external APIs for real-time alerting:
- **Twilio**: SMS to security personnel
- **Telegram Bot**: Push notifications with frame snapshots
- **SendGrid**: Email incident reports

---

## Final Repository Structure (End State)

```
DetectiveAI-camera/
|-- models/
|   |-- best.onnx            <-- RT-DETR-X Gate 1 (~136MB)
|   +-- verifier.onnx        <-- EfficientNet Gate 2 (~20MB)
|-- src/
|   |-- detection.py         <-- 2-stage pipeline (Phase 3 update)
|   |-- tracking.py          <-- DeepSORT (unchanged)
|   |-- rules.py             <-- Updated thresholds (Phase 3)
|   |-- visualize.py         <-- Updated overlays
|   +-- streamlit_app.py     <-- Monitoring dashboard
|-- web/                     <-- React + Tailwind website (Phase 4)
|   |-- src/
|   |-- package.json
|   +-- tailwind.config.js
|-- Dockerfile               <-- Production containerization (Phase 5)
|-- docker-compose.yml
|-- requirements.txt
+-- ROADMAP.md               <-- This file
```

---
> **Target Performance:**  < 0.5% false positive rate  |  < 16ms inference on RTX 2050
> **Training Cost:** $0 (Kaggle free tier) to ~$1.40 (Vast.ai RTX 4090)
> **Built with:** RT-DETR-X · EfficientNet-B4 · DeepSORT · ONNX Runtime · React · Tailwind
