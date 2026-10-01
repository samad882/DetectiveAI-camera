"""
╔══════════════════════════════════════════════════╗
║        DetectiveAI — Local Video Tester          ║
║                                                  ║
║  Usage:                                          ║
║    python test_model.py                          ║
║    python test_model.py --video path/to/vid.mp4  ║
║    python test_model.py --size 320               ║
║                                                  ║
║  Controls (when video window is open):           ║
║    Q  → Quit                                     ║
║    P  → Pause / Unpause                          ║
╚══════════════════════════════════════════════════╝
"""

import cv2
import time
import argparse
import sys
import os

# Add src/ to path so we can import Detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detection import Detector

# Colors for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "pistol":  (0,   50,  255),
    "gun":     (0,   50,  255),
    "knife":   (0,   165, 255),
    "fight":   (0,   0,   220),
    "person":  (180, 180, 0  ),
}
DEFAULT_COLOR = (0, 255, 100)

# Terminal colors
RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

def cls_color(name):
    danger = {"pistol", "gun", "knife", "fight", "rifle", "sword", "cleaver"}
    return RED if name.lower() in danger else GREEN

def print_frame_stats(frame_num, fps, detections, latency_ms):
    fps_color = GREEN if fps >= 15 else YELLOW if fps >= 8 else RED
    if detections:
        parts = [f"{cls_color(n)}{n}{RESET}({c:.0%})" for (_, _, _, _, c, n) in detections]
        det_str = "  DETECTED -> " + "  ".join(parts)
    else:
        det_str = f"  {CYAN}No detections{RESET}"
    print(
        f"  Frame {BOLD}{frame_num:>5}{RESET} | "
        f"{fps_color}FPS: {fps:>5.1f}{RESET} | "
        f"Latency: {MAGENTA}{latency_ms:>6.1f}ms{RESET} |"
        f"{det_str}"
    )

def draw_boxes(frame, detections):
    for (x1, y1, x2, y2, conf, name) in detections:
        color = CLASS_COLORS.get(name.lower(), DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name.upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def draw_hud(frame, fps, frame_num, total_frames, latency_ms, model_path):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (370, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    fps_color = (0, 220, 0) if fps >= 15 else (0, 180, 255) if fps >= 8 else (0, 0, 220)
    cv2.putText(frame, f"DetectiveAI  |  {os.path.basename(model_path)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
    cv2.putText(frame, f"Latency: {latency_ms:.1f} ms",
                (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame

def main():
    parser = argparse.ArgumentParser(description="DetectiveAI - Local Video Tester")
    parser.add_argument("--video", "-v", default=None, help="Path to video file")
    parser.add_argument("--model", "-m", default="models/bestonnx.onnx", help="Model path")
    parser.add_argument("--conf",  "-c", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--size",  "-s", type=int,   default=640,  help="Inference size (640 or 320)")
    args = parser.parse_args()

    video_path = args.video
    if not video_path:
        print(f"\n{CYAN}No video path given.{RESET}")
        video_path = input("  Enter full path to your video file: ").strip().strip("'\"")

    if not os.path.isfile(video_path):
        print(f"{RED}Video not found: {video_path}{RESET}")
        sys.exit(1)

    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"{RED}Model not found: {model_path}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"  {BOLD}DetectiveAI - Video Test{RESET}")
    print(f"{'─'*55}")
    print(f"  Model  : {CYAN}{model_path}{RESET}")
    print(f"  Video  : {CYAN}{video_path}{RESET}")
    print(f"  Conf   : {args.conf}    ImgSize: {args.size}")
    print(f"{'─'*55}")
    print(f"  {YELLOW}Loading model...{RESET}", flush=True)

    detector = Detector(model_path=model_path)
    print(f"  {GREEN}Done!{RESET}  Device: {BOLD}{detector.device}{RESET}")
    if detector.names:
        print(f"  Classes: {list(detector.names.values())}")
    print(f"{'─'*55}")
    print(f"  {YELLOW}Controls: [Q] Quit   [P] Pause{RESET}")
    print(f"{'─'*55}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{RED}Could not open video!{RESET}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {int(cap.get(3))}x{int(cap.get(4))}  |  {source_fps:.1f} FPS  |  {total_frames} frames\n")

    frame_num  = 0
    paused     = False
    fps        = 0.0
    latency_ms = 0.0
    prev_time  = time.perf_counter()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\n{GREEN}Video finished!{RESET}")
                break

            frame_num += 1
            detections = detector.detect(frame, conf_threshold=args.conf, img_size=args.size)

            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            stats = detector.latency_stats()
            latency_ms = stats["avg_ms"] or 0.0

            print_frame_stats(frame_num, fps, detections, latency_ms)
            frame = draw_boxes(frame, detections)
            frame = draw_hud(frame, fps, frame_num, total_frames, latency_ms, model_path)

        cv2.imshow("DetectiveAI  |  Q=Quit  P=Pause", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print(f"\n{YELLOW}Stopped by user.{RESET}")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}")

    cap.release()
    cv2.destroyAllWindows()

    final = detector.latency_stats()
    print(f"\n{'─'*55}")
    print(f"  {BOLD}Final Stats:{RESET}")
    print(f"  Avg Latency  : {MAGENTA}{final['avg_ms']} ms{RESET}")
    print(f"  Best Latency : {GREEN}{final['min_ms']} ms{RESET}")
    print(f"  Device Used  : {CYAN}{final['device']}{RESET}")
    print(f"  Total Frames : {frame_num}")
    print(f"{'─'*55}\n")

if __name__ == "__main__":
    main()
