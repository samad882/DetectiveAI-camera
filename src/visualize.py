import cv2
import time

# ┌──────────────────────────────────────────────────────────────────┐
# │                    VISUALIZE MODULE                              │
# │  Draws bounding boxes + alert text on each video frame           │
# │                                                                  │
# │  draw_tracks() → draws a box + ID label for every tracked object │
# │  draw_alerts() → draws alert text at top of frame in red         │
# └──────────────────────────────────────────────────────────────────┘


# ┌──────────────────────────────────────────────────────────────────┐
# │  DRAW TRACKS                                                     │
# │  For every tracked object → draw a yellow bounding box           │
# │  and show the track ID + class name above it                     │
# │                                                                  │
# │  Input:  frame + list of tracks from tracker                     │
# │  Output: frame with boxes drawn on it                            │
# └──────────────────────────────────────────────────────────────────┘
def draw_tracks(frame, tracks):
    out = frame.copy()

    for t in tracks:
        try:
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            tid  = getattr(t, "track_id", None)
            name = t.get_det_class() if hasattr(t, "get_det_class") else "obj"
            name_lower = name.lower()

            # ── PROFESSIONAL CLEAN COLOR CODING ──
            # Weapon: Red (0, 0, 239) in BGR
            # Person: Blue (235, 99, 37) in BGR (Tailwind blue-600)
            # Bag/Other: Purple (182, 36, 139) in BGR
            if "gun" in name_lower or "armed" in name_lower or "knife" in name_lower or "sword" in name_lower or "ax" in name_lower:
                color = (40, 40, 239) # Red (BGR)
                txt_color = (255, 255, 255)
            elif "person" in name_lower or "man" in name_lower:
                color = (235, 99, 37) # Blue (BGR)
                txt_color = (255, 255, 255)
            else:
                color = (182, 36, 139) # Purple (BGR)
                txt_color = (255, 255, 255)

            # Draw clean thin bounding box
            thickness = 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # ── DRAW CLEAN LABEL BACKGROUND & TEXT ──
            label = f"{name.title()} #{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Label box background (solid color, no transparency)
            cv2.rectangle(out, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            
            # Label text
            cv2.putText(out, label, (x1 + 2, y1 - 3), font, font_scale, txt_color, 1, cv2.LINE_AA)

        except Exception:
            pass 
            
    return out


# ┌──────────────────────────────────────────────────────────────────┐
# │  DRAW ALERTS                                                     │
# │  Prints up to 5 alert messages at the top-left of the frame      │
# │  in red text so they're visible on any background                │
# │                                                                  │
# │  Input:  frame + list of current alert dicts                     │
# │  Output: frame with alert text overlaid                          │
# └──────────────────────────────────────────────────────────────────┘
def draw_alerts(frame, alerts):
    # 📌 KEY CONCEPT: Defensive Copy + List Slicing [:5]
    # Drawing on a copy avoids mutating the original frame; [:5] caps output to prevent UI overflow
    out = frame.copy()
    y = 20  # Starting Y position for first line of text

    for a in alerts[:5]:  # Show max 5 alerts on-screen at once
        # 📌 KEY CONCEPT: Unix Timestamp → Human-Readable Time (time.strftime)
        # Alert timestamps are stored as float seconds (Unix epoch) — strftime formats them for display
        # Format: "20:13:07 [WEAPON] Weapon (pistol) detected!"
        ts  = time.strftime('%H:%M:%S', time.localtime(a['timestamp']))
        txt = f"{ts} [{a['type']}] {a['message']}"

        # Draw red text onto the frame
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25  # Move down for next alert line

    return out
