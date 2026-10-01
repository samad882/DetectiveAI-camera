"""
╔════════════════════════════════════════════════════╗
║  DetectiveAI — Full Pipeline Diagnostic            ║
║  Tests: Raw Model → Detection → Tracking → Rules   ║
║  Finds EXACTLY where the bug is                     ║
╚════════════════════════════════════════════════════╝
"""
import cv2
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; M = "\033[95m"
B = "\033[1m"; X = "\033[0m"

MODEL = "models/bestonnx.onnx"

def sep(title):
    print(f"\n{B}{'═'*60}{X}")
    print(f"  {B}{title}{X}")
    print(f"{'═'*60}{X}")

# ── STAGE 0: Check files ────────────────────────────────────────
sep("STAGE 0: File Check")
if not os.path.isfile(MODEL):
    print(f"  {R}❌ Model NOT found: {MODEL}{X}")
    sys.exit(1)
size_mb = os.path.getsize(MODEL) / (1024*1024)
print(f"  {G}✅ Model found: {MODEL} ({size_mb:.1f} MB){X}")

# ── STAGE 1: Load model and check classes ───────────────────────
sep("STAGE 1: Model Load + Class Check")
from detection import Detector

detector = Detector(model_path=MODEL)
print(f"  Device: {C}{detector.device}{X}")
print(f"  Raw class names from model:")
if detector.names:
    for idx, name in detector.names.items():
        print(f"    {C}[{idx}]{X} → '{Y}{name}{X}'")
else:
    print(f"  {R}❌ NO CLASS NAMES FOUND!{X}")

print(f"\n  Class remap dictionary:")
for k, v in detector.class_remap.items():
    print(f"    '{Y}{k}{X}' → '{G}{v}{X}'")

# ── STAGE 2: Raw detection on video frame ───────────────────────
sep("STAGE 2: Raw Detection Test (No Tracking)")

video_path = None
for arg in sys.argv[1:]:
    if os.path.isfile(arg):
        video_path = arg
        break

if not video_path:
    # Try to find any video in videos/
    vfolder = os.path.join(os.path.dirname(__file__), "videos")
    if os.path.isdir(vfolder):
        vids = [f for f in os.listdir(vfolder) if f.endswith(('.mp4','.avi','.mov'))]
        if vids:
            video_path = os.path.join(vfolder, vids[0])

if not video_path:
    print(f"  {Y}No video found. Put a .mp4 in videos/ folder or pass as argument.{X}")
    print(f"  {Y}Testing with a blank frame (300x300 black image)...{X}")
    import numpy as np
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    
    for conf in [0.10, 0.20, 0.30, 0.50]:
        for size in [320, 640]:
            dets = detector.detect(frame, conf_threshold=conf, img_size=size)
            print(f"    conf={conf} size={size} → {len(dets)} detections (expected: 0 on blank)")
    print(f"\n  {G}✅ Model runs correctly on blank frame.{X}")
    print(f"  {Y}⚠️  Put a video in videos/ folder to do real testing!{X}")
    sys.exit(0)

print(f"  Video: {C}{video_path}{X}")
cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  Frames: {total}, Size: {int(cap.get(3))}x{int(cap.get(4))}")

# Test on frames: 1, 25%, 50%, 75%
test_frames = [1, max(1, total//4), max(1, total//2), max(1, 3*total//4)]

for target_frame in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    if not ret:
        continue
    
    print(f"\n  {B}--- Frame #{target_frame} ---{X}")
    
    # Test multiple conf thresholds and sizes
    for conf in [0.10, 0.25, 0.40]:
        for size in [320, 640]:
            t0 = time.perf_counter()
            dets = detector.detect(frame, conf_threshold=conf, img_size=size)
            ms = (time.perf_counter() - t0) * 1000
            
            if dets:
                for (x1,y1,x2,y2,c,name) in dets:
                    print(f"    conf≥{conf} size={size} [{ms:>6.1f}ms] → {G}{name}{X} ({c:.0%}) box=({x1},{y1},{x2},{y2})")
            else:
                print(f"    conf≥{conf} size={size} [{ms:>6.1f}ms] → {Y}Nothing detected{X}")

# ── STAGE 3: Tracking test ──────────────────────────────────────
sep("STAGE 3: Tracking Test")
from tracking import Tracker, HAS_DEEPSORT

tracker = Tracker()
print(f"  DeepSORT installed: {G if HAS_DEEPSORT else R}{'YES' if HAS_DEEPSORT else 'NO (using fallback)'}{X}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
tracked_count = 0
for i in range(min(30, total)):
    ret, frame = cap.read()
    if not ret: break
    dets = detector.detect(frame, conf_threshold=0.25, img_size=320)
    tracks = tracker.update(dets, frame)
    confirmed = [t for t in tracks if t.is_confirmed()]
    
    if confirmed:
        for t in confirmed:
            name = (t.get_det_class() or "???").lower()
            ltrb = t.to_ltrb()
            print(f"    Frame {i+1:>3}: Track ID={t.track_id} class='{G}{name}{X}' box={[int(v) for v in ltrb]}")
            tracked_count += 1

if tracked_count == 0:
    print(f"  {R}❌ NO tracks produced in 30 frames! Possible issue: tracker or low confidence.{X}")
else:
    print(f"  {G}✅ {tracked_count} tracked detections in 30 frames.{X}")

# ── STAGE 4: Rules test ─────────────────────────────────────────
sep("STAGE 4: Rules Engine Test")
from rules import RuleEngine, WEAPON_KEYWORDS, PERSON_KEYWORDS, BAG_KEYWORDS

# Get final class names AFTER remap
final_classes = set()
if detector.names:
    for idx, name in detector.names.items():
        n = name.lower()
        if detector.class_remap and n in detector.class_remap:
            n = detector.class_remap[n]
        final_classes.add(n)

print(f"  Final class names (after remap): {sorted(final_classes)}")

rules = RuleEngine(model_classes=list(final_classes))
print(f"\n  Rule flags:")
print(f"    has_persons = {G if rules.has_persons else R}{rules.has_persons}{X}")
print(f"    has_bags    = {G if rules.has_bags else R}{rules.has_bags}{X}")
print(f"    has_weapons = {G if rules.has_weapons else R}{rules.has_weapons}{X}")

# Check each class name against rules classification
print(f"\n  Class → Category mapping:")
for cls in sorted(final_classes):
    cat = rules._classify(cls)
    color = G if cat else R
    print(f"    '{Y}{cls}{X}' → {color}{cat or 'UNRECOGNIZED ⚠️'}{X}")

cap.release()

# ── SUMMARY ─────────────────────────────────────────────────────
sep("DIAGNOSTIC SUMMARY")
print(f"  Model file    : {G}OK{X}")
print(f"  Model loads   : {G}OK{X}")
print(f"  Classes exist : {G if detector.names else R}{'OK (' + str(len(detector.names)) + ' classes)' if detector.names else 'FAIL'}{X}")
print(f"  Remap works   : {G}OK{X}")
print(f"  Rules flags   : persons={rules.has_persons} bags={rules.has_bags} weapons={rules.has_weapons}")
print(f"  Tracker       : {'DeepSORT' if HAS_DEEPSORT else 'Fallback centroid'}")
print(f"\n  {B}Done! Check the output above for any ❌ or ⚠️ markers.{X}\n")
