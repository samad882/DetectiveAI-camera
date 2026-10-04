import numpy as np
import time

# ┌─────────────────────────────────────────┐
# │           DETECTION MODULE              │
# │  Loads ONNX model → runs on each frame  │
# │  Returns bounding boxes + class names   │
# └─────────────────────────────────────────┘

class Detector:

    # ┌──────────────────────────────────────────────────────────┐
    # │  SETUP                                                   │
    # │  Load YOLO model + pick CPU or GPU                       │
    # │                                                          │
    # │  LATENCY TIPS:                                           │
    # │    • GPU  → install onnxruntime-gpu  → ~4ms per frame    │
    # │    • CPU  → default (Apple/Intel)    → ~31ms per frame   │
    # │    • img_size=320 instead of 640     → ~2x faster        │
    # └──────────────────────────────────────────────────────────┘
    def __init__(self, model_path="yolov8n.pt", device=None):
        # 📌 KEY CONCEPT: Conditional Expression (Ternary Operator)
        # Checks if CUDA GPU is available — selects "0" (GPU) or "cpu" without a full if-else block
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        import torch

        # Auto-pick GPU ("0") if CUDA available, else CPU
        # On Apple Silicon: always CPU (no CUDA support)
        # On NVIDIA: uses GPU → much faster
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")

        # 📌 KEY CONCEPT: ONNX Runtime Model Loading
        # ONNX is a hardware-independent model format — same file runs on GPU or CPU without code changes
        # Load the ONNX model from disk
        self.model = YOLO(model_path, task="detect")

        # 📌 KEY CONCEPT: Dictionary (HashMap) for Class Index → Name Mapping
        # self.names = {0: "pistol", 1: "knife"} — O(1) lookup to resolve integer class IDs to strings
        # Store class names: e.g. {0: "pistol", 1: "knife"}
        self.names = getattr(self.model, "names", None)

        # 📌 KEY CONCEPT: Clean up raw dataset class names
        # Some Roboflow datasets have weird names like "0" or "1" for class IDs
        self.class_remap = {
            # Normalize dataset overlap for persons
            "0": "armed man", 
            "1": "person",
            "man": "person",
            
            # Normalize dataset overlap for guns
            "handgun": "gun",
            "handgunth": "gun",
            "revolver": "gun",
            "semi automatic": "gun",
            "ak": "gun",
            "m16": "gun",
            "rifle": "gun",
            "shotgun": "gun",
            "shotgunth": "gun",
            
            # Normalize dataset overlap for blades
            "multiknife": "knife",
            "sword": "knife",
            "long sword": "knife",
            "short sword": "knife",
            "ax": "knife",
            "cleaver": "knife",
            "cutter": "knife",
            "spear": "knife",
            "eto": "knife",
            
            # Normalize baggage
            "bag": "baggage"
        }

        # 📌 KEY CONCEPT: Sliding Window List (Rolling Log)
        # Only the last 30 inference times are kept — old values removed with pop(0) to keep memory constant
        # Latency tracking — running average over last 30 frames
        self._latency_log = []

    # ┌──────────────────────────────────────────────────────────────┐
    # │  DETECT                                                      │
    # │  Takes one video frame → returns list of detected objects    │
    # │                                                              │
    # │  Input:  frame (numpy image array from OpenCV)               │
    # │  Output: [(x1, y1, x2, y2, confidence, "class_name"), ...]   │
    # │                                                              │
    # │  WHERE LATENCY COMES FROM:                                   │
    # │    ┌─────────────────────────────────────────────────────┐   │
    # │    │  model.predict()  ← 95% of the time is spent here   │   │
    # │    │    img_size=640   → resizes frame → runs 80 layers  │   │
    # │    │    img_size=320   → 2x faster, slightly less detail │   │
    # │    └─────────────────────────────────────────────────────┘   │
    # │    post-processing (box parsing) ← ~5% of time               │
    # └──────────────────────────────────────────────────────────────┘
    def detect(self, frame, conf_threshold=0.25, img_size=640):
        # 📌 KEY CONCEPT: High-Resolution Timer (time.perf_counter)
        # More accurate than time.time() — used for millisecond-level latency measurement

        # ── STEP 1: Run model inference — this is the slow part ────────
        # img_size=640 → standard, balanced accuracy vs speed
        # img_size=320 → faster (use if speed matters more than accuracy)
        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            imgsz=img_size,       # ← reduce to 320 for 2x speed boost
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        # 📌 KEY CONCEPT: Sliding Window List (Rolling Log)
        # Only last 30 samples are kept — pop(0) removes oldest entry to maintain constant memory usage
        # ── Measure and log how long inference took ─────────────────────
        inference_ms = (time.perf_counter() - t0) * 1000
        self._latency_log.append(inference_ms)
        if len(self._latency_log) > 30:
            self._latency_log.pop(0)          # Keep only last 30 samples

        if not results:
            return []

        r = results[0]
        detections = []

        # 📌 KEY CONCEPT: Tensor → NumPy Conversion + Type Safety
        # PyTorch tensors cannot be used directly in OpenCV — .cpu().numpy() converts them to NumPy arrays
        # ── STEP 2: Parse boxes — fast, ~1ms ───────────────────────────
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                try:
                    # Extract class index + confidence as plain Python numbers
                    cls_i = int(box.cls[0].item()) if hasattr(box.cls, "__len__") and len(box.cls) > 0 else int(box.cls.item())
                    conf  = float(box.conf[0].item()) if hasattr(box.conf, "__len__") and len(box.conf) > 0 else float(box.conf.item())

                    # Extract corner coordinates [x1, y1, x2, y2]
                    xyxy_tensor = box.xyxy[0] if box.xyxy.ndim > 1 else box.xyxy
                    xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                except Exception:
                    continue  # Skip bad boxes

                # 📌 KEY CONCEPT: HashMap O(1) Lookup
                # Resolving class index (int) → class name (string) — dict lookup is instant regardless of size
                # ── Resolve class index → class name ───────────────────
                if isinstance(self.names, dict):
                    name = self.names.get(cls_i, str(cls_i))
                elif isinstance(self.names, (list, tuple)):
                    try:
                        name = self.names[int(cls_i)]
                    except Exception:
                        name = str(cls_i)
                else:
                    name = str(cls_i)

                name = name.lower()

                # Optional rename (e.g. "guns" → "pistol")
                if self.class_remap and name in self.class_remap:
                    name = self.class_remap[name]

                # ── FIX B: Skip tiny boxes ─────────────────────────────────
                box_w = x2 - x1
                box_h = y2 - y1

                # CCTV cameras show distant objects → smaller bounding boxes
                # Using 20px minimum to catch weapons at distance while filtering noise
                ALL_WEAPON_CLASSES = {
                    "pistol", "rifle", "ak", "m16", "revolver",
                    "semi automatic", "shotgun", "handgun", "handgunth",
                    "shotgunth", "gun", "armed man", "person with a gun",
                    "knife", "multiknife", "sword", "long sword", "short sword",
                    "ax", "cleaver", "cutter", "spear", "eto"
                }
                min_box = 20 if name in ALL_WEAPON_CLASSES else 18
                if box_w < min_box or box_h < min_box:
                    continue  # Too small → shadow/noise, skip

                # ── Per-class confidence boosts ───────────────────────────────
                # Confidence boosts per class to reduce false positives:
                # "armed man" is very noisy in fight scenes → needs high boost
                # Melee weapons: high false positive rate → need confidence boost
                # Guns (handgun/rifle/etc): no boost — model sees them at low conf naturally
                class_boost = {
                    "armed man":          0.25,  # Very noisy in fights — require conf > 0.50
                    "person with a gun":  0.20,  # Also noisy
                    "knife":              0.12,
                    "cutter":             0.12,
                    "cleaver":            0.10,
                    "sword":              0.10,
                    "long sword":         0.10,
                    "short sword":        0.10,
                    "ax":                 0.10,
                    "spear":              0.08,
                    "multiknife":         0.10,
                }
                boost = class_boost.get(name, 0.0)
                min_conf = min(conf_threshold + boost, 0.80)
                if conf < min_conf:
                    continue  # Below per-class threshold → skip

                # 📌 KEY CONCEPT: List of Tuples (Structured Output)
                # Each detection is a tuple: (x1, y1, x2, y2, confidence, class_name) — fixed-width, fast to unpack
                # Add to results
                detections.append((x1, y1, x2, y2, float(conf), name))

        # ── Agnostic NMS (Non-Maximum Suppression) ────────────────────
        # Since we mapped many overlapping classes to the same name (e.g. 'gun'),
        # we might have multiple 'gun' bounding boxes for the exact same object.
        # We need to suppress the lower confidence duplicates.
        def _iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

        final_detections = []
        # Sort by confidence descending
        detections.sort(key=lambda x: x[4], reverse=True)
        for d in detections:
            overlap = False
            for fd in final_detections:
                # If they have the same unified class (e.g., both are 'gun') and overlap heavily
                if d[5] == fd[5] and _iou(d, fd) > 0.45:
                    overlap = True
                    break
            if not overlap:
                final_detections.append(d)

        return final_detections

    # ┌──────────────────────────────────────────────────────────────┐
    # │  GET LATENCY STATS                                           │
    # │  Call anytime to see actual measured inference speed         │
    # │                                                              │
    # │  Usage:  detector.latency_stats()                            │
    # │  Output: {"avg_ms": 31.3, "min_ms": 24.1, "device": "cpu"}  │
    # └──────────────────────────────────────────────────────────────┘
    def latency_stats(self):
        # 📌 KEY CONCEPT: Aggregation on a List (Mean + Min)
        # sum(list)/len(list) = rolling average — min() gives best-case (fastest frame) for benchmarking
        if not self._latency_log:
            return {"avg_ms": None, "min_ms": None, "device": self.device}
        avg = sum(self._latency_log) / len(self._latency_log)
        mn  = min(self._latency_log)
        return {
            "avg_ms": round(avg, 1),
            "min_ms": round(mn, 1),
            "device": self.device,
            # Quick note on how to improve:
            # "tip": "Use img_size=320 for 2x faster, or onnxruntime-gpu for ~4ms on NVIDIA"
        }
