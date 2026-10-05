import time
import math

# ┌──────────────────────────────────────────────────────────────────┐
# │                      RULES ENGINE                                │
# │  Reads confirmed tracks each frame → decides what is dangerous   │
# │                                                                  │
# │  3 rules:                                                        │
# │    WEAPON  → pistol/knife seen for N consecutive frames          │
# │    CROWD   → too many people in frame at once                    │
# │    BAG     → bag left alone (no person nearby) for N seconds     │
# │                                                                  │
# │  Rules auto-enable/disable based on what the model can detect    │
# └──────────────────────────────────────────────────────────────────┘

# ── Keyword lists — tuned for the 5-class best.onnx model ───────────
# Classes: person | armed man | gun | knife | baggage
PERSON_KEYWORDS = ["person"]
BAG_KEYWORDS    = ["baggage"]
WEAPON_KEYWORDS = ["gun", "knife", "armed man"]


class RuleEngine:

    # ┌──────────────────────────────────────────────────────────────┐
    # │  SETUP                                                       │
    # │  Receives model class names → auto-enables matching rules    │
    # └──────────────────────────────────────────────────────────────┘
    def __init__(self,
                 crowd_threshold=20,
                 bag_stationary_seconds=10,
                 weapon_persist_frames=6,
                 min_track_frames_before_bag=8,
                 alert_cooldowns=None,
                 model_classes=None):

        # 📌 KEY CONCEPT: Instance Variables (Object State)
        # All thresholds stored as instance variables so they can be tuned at runtime without changing code
        # Thresholds
        self.crowd_threshold           = crowd_threshold
        self.bag_stationary_seconds    = bag_stationary_seconds
        self.weapon_persist_frames     = weapon_persist_frames
        self.min_track_frames_before_bag = min_track_frames_before_bag

        # Cooldowns: seconds between repeated alerts of same type
        self.alert_cooldowns = alert_cooldowns or {"weapon": 8.0, "bag": 10.0, "crowd": 10.0}

        # 📌 KEY CONCEPT: Multiple HashMaps for Per-Object State Tracking
        # Each dict is a separate 'track_id → state' mapping — this is the system's memory across frames
        # ── State memory across frames ────────────────────────────
        self.bag_track_info            = {}  # tracks bag positions + timers
        self.weapon_persist             = {}  # counts consecutive frames weapon seen
        self.last_alert_time            = {"weapon": 0.0, "bag": 0.0, "crowd": 0.0, "fight": 0.0}
        self.track_frame_counts         = {}  # how many frames each track has been seen
        self.last_weapon_alert_for_tid  = {}  # per-track weapon alert cooldown
        # Fight proximity tracking: how many frames are 2+ persons overlapping
        self.fight_proximity_counter    = 0
        # 📌 KEY CONCEPT: Position History for Velocity Calculation
        # Stores last N positions per track_id so we can compute speed (pixels/frame)
        self.person_pos_history         = {}  # {tid: [(cx, cy), ...]} last 8 positions
        self.pair_dist_history          = {}  # {(tid1,tid2): [dist, ...]} last 8 distances

        # 📌 KEY CONCEPT: List Comprehension + any() for Keyword Matching
        # Auto-detects which rules to enable based on what class names the loaded model supports
        # ── Check which rules are relevant for this model ────────
        classes = [c.lower() for c in (model_classes or [])]
        self.has_persons = any(any(k in c for k in PERSON_KEYWORDS) for c in classes)
        self.has_bags    = any(any(k in c for k in BAG_KEYWORDS)    for c in classes)
        self.has_weapons = any(any(k in c for k in WEAPON_KEYWORDS) for c in classes)

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  CLASSIFY                                                        │
    # │  Maps a class name string → category (person / bag / weapon)     │
    # └──────────────────────────────────────────────────────────────────┘
    def _classify(self, name):
        # Multi-label classification: returns a SET of categories
        # This is the core ontology fix — 'armed man' is BOTH a person AND a weapon
        # Old system returned a single string and caused cascading failures.
        n = name.lower()
        labels = set()
        if any(k in n for k in WEAPON_KEYWORDS): labels.add("weapon")
        if any(k in n for k in BAG_KEYWORDS):    labels.add("bag")
        # Person check: 'person' keyword OR 'armed man' (he is still a human)
        if any(k in n for k in PERSON_KEYWORDS) or n == "armed man": labels.add("person")
        return labels  # e.g. {"weapon", "person"} for "armed man"

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  PROCESS                                                         │
    # │  Main function — called every frame with the current track list  │
    # │                                                                  │
    # │  Input:  tracks (list of track objects from tracker)             │
    # │  Output: list of alert dicts [{"type", "message", ...}, ...]     │
    # └──────────────────────────────────────────────────────────────────┘
    def process(self, tracks, frame_index, frame_timestamp=None):
        now = frame_timestamp if frame_timestamp else time.time()
        alerts = []

        # 📌 KEY CONCEPT: Set for O(1) Membership Lookup
        # Using a Set instead of a List — 'tid not in active_tids' is O(1) vs O(N) for a list
        active_tids      = set()

        # 📌 KEY CONCEPT: Separate HashMaps per Category
        # Splitting into 3 dicts lets us count persons with len(), loop bags separately, etc.
        persons_tracked  = {}
        bags_tracked     = {}
        weapon_candidates = {}

        # ── Step 1: Sort all confirmed tracks into categories ────────
        for t in tracks:
            # 📌 KEY CONCEPT: Guard Clause (Early Continue)
            # Skip unconfirmed tracks to reduce false positives — cleaner than deep nesting
            if not t.is_confirmed():
                continue

            tid = t.track_id
            active_tids.add(tid)

            # 📌 KEY CONCEPT: dict.get() with Default Value
            # Avoids KeyError on first encounter — returns 0 if track_id is new
            # Count how many frames this track has existed
            self.track_frame_counts[tid] = self.track_frame_counts.get(tid, 0) + 1

            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Get class name + confidence safely
            name = (t.get_det_class() or "unknown").lower()
            conf = t.get_det_conf() if hasattr(t, "get_det_conf") else 0.5

            # ── FIX 1: Multi-Label Ontology ───────────────────────────────
            # _classify() now returns a SET. This means "armed man" maps to
            # {"person", "weapon"} simultaneously — he IS a person AND a threat.
            # Before this fix: armed man vanished from crowd/fight detection.
            categories = self._classify(name)

            if "person" in categories:
                persons_tracked[tid] = {"bbox": (x1, y1, x2, y2), "center": (cx, cy)}
            if "bag" in categories:
                bags_tracked[tid] = {"bbox": (x1, y1, x2, y2), "center": (cx, cy)}
            if "weapon" in categories:
                weapon_candidates[tid] = {"bbox": (x1, y1, x2, y2), "center": (cx, cy),
                                          "name": name, "conf": conf}

        # 📌 KEY CONCEPT: Garbage Collection / Manual Memory Management
        # Remove disappeared track IDs from all state dicts — prevents unbounded memory growth
        # ── Step 2: Clean up memory for tracks that are gone ────────
        for tid in list(self.track_frame_counts.keys()):
            if tid not in active_tids:
                self.track_frame_counts.pop(tid, None)
                self.bag_track_info.pop(tid, None)
                self.weapon_persist.pop(tid, None)
                self.last_weapon_alert_for_tid.pop(tid, None)
                self.person_pos_history.pop(tid, None)  # Clean up motion history

        # ── RULE 1: CROWD ────────────────────────────────────────────
        # Only runs if model can detect persons
        if self.has_persons:
            # 📌 KEY CONCEPT: len() on a dict is O(1)
            # No loop needed — Python tracks dict size internally as a counter
            person_count = len(persons_tracked)
            if person_count >= self.crowd_threshold:
                if now - self.last_alert_time["crowd"] >= self.alert_cooldowns["crowd"]:
                    alerts.append({
                        "type": "CROWD",
                        "message": f"Crowd detected: {person_count} people",
                        "timestamp": now,
                        "frame_idx": frame_index
                    })
                    self.last_alert_time["crowd"] = now

        # ── RULE 2: UNATTENDED BAG ───────────────────────────────────
        # Only runs if model can detect bags
        if self.has_bags:
            for tid, info in bags_tracked.items():
                # Wait for track to be stable before alerting
                if self.track_frame_counts.get(tid, 0) < self.min_track_frames_before_bag:
                    continue

                cx, cy = info["center"]

                # First time seeing this bag — record position + time
                if tid not in self.bag_track_info:
                    self.bag_track_info[tid] = {"last_center": (cx, cy), "first_seen": now, "last_seen": now}
                    continue

                entry = self.bag_track_info[tid]

                # 📌 KEY CONCEPT: Euclidean Distance (math.hypot = Pythagoras theorem)
                # math.hypot(dx, dy) = sqrt(dx² + dy²) — measures pixel distance between two centers
                dist  = math.hypot(entry["last_center"][0] - cx, entry["last_center"][1] - cy)

                if dist > 10:
                    # Bag moved — reset the timer
                    entry["last_center"] = (cx, cy)
                    entry["first_seen"]  = now
                entry["last_seen"] = now

                # 📌 KEY CONCEPT: Timestamp-based Stationary Timer (FPS-independent)
                # Using real wall-clock seconds instead of frame counts — works at any FPS
                duration = entry["last_seen"] - entry["first_seen"]

                # 📌 KEY CONCEPT: Generator Expression + any() for Proximity Check
                # Lazy evaluation — any() stops at the first person found within range (no full scan)
                # Check if any person is close to the bag (within 150px)
                has_owner = any(
                    math.hypot(pinfo["center"][0] - cx, pinfo["center"][1] - cy) < 150
                    for pinfo in persons_tracked.values()
                )

                # Alert if bag is alone and has been stationary long enough
                if not has_owner and duration >= self.bag_stationary_seconds:
                    if now - self.last_alert_time["bag"] >= self.alert_cooldowns["bag"]:
                        alerts.append({
                            "type": "UNATTENDED_BAG",
                            "message": f"Bag {tid} unattended for {duration:.1f}s",
                            "timestamp": now, "frame_idx": frame_index,
                            "bbox": info["bbox"], "track_id": tid
                        })
                        self.last_alert_time["bag"] = now
                        entry["first_seen"] = now  # Reset timer after alert

        # ── RULE 3: WEAPON (Enterprise-Grade) ───────────────────────────
        if self.has_weapons:

            # Helper: does weapon centroid fall inside (or near) any person bbox?
            def _weapon_has_owner(w_bbox, p_dict, px_margin=80):
                """
                FIX 3 — Spatial Weapon Ownership (Bi-partite Matching).
                A weapon floating in empty air = CCTV glare/noise.
                A weapon inside/near a person bbox = real threat.
                px_margin accounts for arm/hand reach beyond person bbox.
                """
                wx1, wy1, wx2, wy2 = w_bbox
                wcx = (wx1 + wx2) / 2
                wcy = (wy1 + wy2) / 2
                for pinfo in p_dict.values():
                    px1, py1, px2, py2 = pinfo["bbox"]
                    # Expand person box by margin for hand/arm reach
                    if (px1 - px_margin) <= wcx <= (px2 + px_margin) and \
                       (py1 - px_margin) <= wcy <= (py2 + px_margin):
                        return True
                return False

            for tid, winfo in weapon_candidates.items():

                # Suppress armed man alerts during active fights (fight is noisy)
                is_fight_active = (now - self.last_alert_time.get("fight", 0.0)) < 15.0
                if is_fight_active and winfo["name"] == "armed man":
                    continue

                # FIX 2: Aspect ratio filter REMOVED.
                # Brittle heuristic that caused false negatives in top-down CCTV.
                # A gun aimed at the camera = square box — old filter was dropping it.
                # Trust the neural network confidence instead.

                # FIX 3: Weapon Ownership Check
                # gun/knife must be held by someone — not floating in empty air.
                # armed man IS the person, so skip the check for him.
                if winfo["name"] != "armed man":
                    if not _weapon_has_owner(winfo["bbox"], persons_tracked):
                        # No nearby person → likely noise/glare. Reset counter.
                        self.weapon_persist.pop(tid, None)
                        continue

                # FIX 4: EMA Confidence Averaging
                # EMA smooths out one-frame spikes: real weapons sustain high conf.
                # Formula: ema = 0.7 * new_conf + 0.3 * old_ema
                current_conf = winfo.get("conf", 0.5)
                prev = self.weapon_persist.get(tid, {"count": 0, "ema_conf": current_conf})
                prev["count"] = prev.get("count", 0) + 1
                old_ema = prev.get("ema_conf", current_conf)
                prev["ema_conf"] = 0.7 * current_conf + 0.3 * old_ema
                self.weapon_persist[tid] = prev

                # Per-class EMA threshold: armed man needs highest bar
                ema_min = {"gun": 0.48, "knife": 0.50, "armed man": 0.58}.get(winfo["name"], 0.48)

                # Alert only after N frames AND EMA conf sustained above threshold
                if (prev["count"] >= self.weapon_persist_frames and
                        prev["ema_conf"] >= ema_min):
                    last_tid_time = self.last_weapon_alert_for_tid.get(tid, 0.0)
                    if (now - last_tid_time >= self.alert_cooldowns["weapon"] and
                            now - self.last_alert_time["weapon"] >= self.alert_cooldowns["weapon"]):
                        weapon_display_name = winfo["name"].title()
                        alerts.append({
                            "type": f"THREAT: {weapon_display_name}",
                            "message": f"Weapon detected! (conf={prev['ema_conf']:.0%})",
                            "timestamp": now, "frame_idx": frame_index,
                            "bbox": winfo["bbox"], "track_id": tid
                        })
                        self.last_alert_time["weapon"] = now
                        self.last_weapon_alert_for_tid[tid] = now




        # ── RULE 4: SMART FIGHT DETECTION (Proximity + Motion) ───────
        # Real fights have 3 signatures:
        #   A. People are CLOSE (overlap or dist < threshold)
        #   B. At least one person is moving FAST (velocity > threshold) — friends talking are still!
        #   C. OR pair distance is RAPIDLY DECREASING (sudden approach)
        # All 3 conditions prevent false positives from groups standing/talking.
        num_persons = len(persons_tracked)
        if self.has_persons and 2 <= num_persons <= 6:
            FIGHT_DIST_THRESHOLD   = 120   # px — close proximity
            FIGHT_PROXIMITY_FRAMES = 8     # consecutive frames needed
            MIN_VELOCITY           = 6.0   # px/frame — below this = person is standing still
            MIN_APPROACH_SPEED     = 4.0   # px/frame — pairs closing in fast
            HISTORY_LEN            = 8     # frames to keep in history

            # ── Update position history for each tracked person ──
            person_ids   = list(persons_tracked.keys())
            person_list  = list(persons_tracked.values())

            for tid, info in persons_tracked.items():
                cx, cy = info["center"]
                hist = self.person_pos_history.get(tid, [])
                hist.append((cx, cy))
                self.person_pos_history[tid] = hist[-HISTORY_LEN:]  # keep last N

            # ── Compute per-person velocity (avg speed over last frames) ──
            def _velocity(tid):
                hist = self.person_pos_history.get(tid, [])
                if len(hist) < 2:
                    return 0.0
                speeds = [math.hypot(hist[i][0]-hist[i-1][0], hist[i][1]-hist[i-1][1])
                          for i in range(1, len(hist))]
                return sum(speeds) / len(speeds)

            # ── Compute approach speed between a pair ──
            def _approach_speed(tid1, tid2):
                key = (min(tid1,tid2), max(tid1,tid2))
                h1 = self.person_pos_history.get(tid1, [])
                h2 = self.person_pos_history.get(tid2, [])
                n  = min(len(h1), len(h2))
                if n < 3:
                    return 0.0
                dists = [math.hypot(h1[-(n-i)][0]-h2[-(n-i)][0],
                                    h1[-(n-i)][1]-h2[-(n-i)][1])
                         for i in range(n)]
                # Positive = getting closer, negative = moving apart
                return (dists[0] - dists[-1]) / max(1, n - 1)

            close_pair_found   = False
            motion_detected    = False

            for i in range(len(person_list)):
                for j in range(i + 1, len(person_list)):
                    cx1, cy1 = person_list[i]["center"]
                    cx2, cy2 = person_list[j]["center"]
                    dist = math.hypot(cx1 - cx2, cy1 - cy2)

                    # Bounding box overlap check
                    b1 = person_list[i]["bbox"]; b2 = person_list[j]["bbox"]
                    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
                    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
                    overlap_area = max(0, ix2-ix1) * max(0, iy2-iy1)
                    min_area = min(
                        max(1, (b1[2]-b1[0])*(b1[3]-b1[1])),
                        max(1, (b2[2]-b2[0])*(b2[3]-b2[1]))
                    )
                    overlap_ratio = overlap_area / min_area

                    # Condition A: Are they close/overlapping?
                    pair_close = (dist < FIGHT_DIST_THRESHOLD or overlap_ratio > 0.10)

                    if pair_close:
                        tid1 = person_ids[i]
                        tid2 = person_ids[j]

                        # Condition B: Is anyone moving fast? (not just standing)
                        v1 = _velocity(tid1)
                        v2 = _velocity(tid2)
                        any_motion = (v1 > MIN_VELOCITY or v2 > MIN_VELOCITY)

                        # Condition C: Are they rapidly approaching each other?
                        approach = _approach_speed(tid1, tid2)
                        rapid_approach = approach > MIN_APPROACH_SPEED

                        # Fight = close + (motion OR rapid approach)
                        if any_motion or rapid_approach:
                            close_pair_found = True
                            motion_detected  = True
                            break

                if close_pair_found:
                    break

            if close_pair_found and motion_detected:
                self.fight_proximity_counter += 1
            else:
                # Decay counter faster if no motion (was just static proximity)
                self.fight_proximity_counter = max(0, self.fight_proximity_counter - 2)

            if self.fight_proximity_counter >= FIGHT_PROXIMITY_FRAMES:
                if now - self.last_alert_time["fight"] >= 10.0:  # 10s cooldown
                    alerts.append({
                        "type": "FIGHT",
                        "message": f"⚠️ (FIGHT) A physical altercation is detected!",
                        "timestamp": now,
                        "frame_idx": frame_index
                    })
                    self.last_alert_time["fight"] = now
        else:
            # 0-1 persons or 7+ persons → reset counter
            self.fight_proximity_counter = max(0, self.fight_proximity_counter - 1)

        return alerts
