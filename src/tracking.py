import math

# ┌──────────────────────────────────────────────────────────────┐
# │                    TRACKING MODULE                           │
# │  Takes detections from each frame → links them over time     │
# │  Assigns a persistent ID to each object (e.g. "Pistol #3")  │
# │                                                             │
# │  Uses DeepSORT if installed, otherwise falls back to a      │
# │  simple centroid-based tracker                              │
# └──────────────────────────────────────────────────────────────┘

# ── Try to import DeepSORT (best tracker) ─────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except Exception:
    DeepSort = None
    HAS_DEEPSORT = False


class Tracker:

    # ┌──────────────────────────────────────────────────────────┐
    # │  SETUP                                                   │
    # │  Init DeepSORT or fallback centroid tracker              │
    # └──────────────────────────────────────────────────────────┘
    def __init__(self, max_age=30):
        # 📌 KEY CONCEPT: Graceful Fallback Pattern
        # If DeepSORT is not installed, system falls back to a simpler centroid tracker — never crashes
        if HAS_DEEPSORT:
            # n_init=1 → confirm track on first detection (no delay)
            self.tracker = DeepSort(max_age=max_age, n_init=1)
            self.use_deepsort = True
        else:
            # Fallback: simple centroid matcher
            self.use_deepsort = False
            self.next_id = 1
            self.objects = {}
            self.max_age = max_age

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  UPDATE                                                          │
    # │  Called every frame with fresh detections                        │
    # │                                                                  │
    # │  Input:  detections → [(x1,y1,x2,y2, conf, "class_name"), ...]   │
    # │  Output: list of track objects, each has:                        │
    # │    track.track_id        → persistent ID string e.g. "3"         │
    # │    track.is_confirmed()  → True if object has been seen enough   │
    # │    track.to_ltrb()       → [x1, y1, x2, y2] bounding box        │
    # │    track.get_det_class() → class name e.g. "pistol"              │
    # └──────────────────────────────────────────────────────────────────┘
    def update(self, detections, frame):

        # ── DeepSORT Path ───────────────────────────────────────────────
        if self.use_deepsort:
            # DeepSORT needs [x1, y1, width, height] not [x1,y1,x2,y2]
            # We also store a conf lookup so rules.py can do EMA confidence averaging
            conf_lookup = {}  # class_name+bbox key → conf
            ds_input = []
            for (x1, y1, x2, y2, conf, name) in detections:
                ds_input.append(([x1, y1, x2 - x1, y2 - y1], conf, name))
                # Key: class name (one weapon per class per frame — good enough)
                conf_lookup[name] = max(conf_lookup.get(name, 0.0), conf)

            tracks = self.tracker.update_tracks(ds_input, frame=frame)

            # Inject conf into each track object so rules.py can access it
            # DeepSORT tracks don't have get_det_conf() — we add it dynamically
            for track in tracks:
                det_class = track.get_det_class() or ""
                stored_conf = conf_lookup.get(det_class.lower(), 0.0)
                track.get_det_conf = lambda c=stored_conf: c  # closure captures value

            return tracks

        # ── Fallback Centroid Tracker ────────────────────────────────────
        else:
            new_objects = {}
            centroids = []

            # 📌 KEY CONCEPT: Centroid Computation
            # Center of a bounding box = midpoint of corners — single point representing each object
            # Compute center point of each detected box
            for (x1, y1, x2, y2, conf, name) in detections:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append(((x1, y1, x2, y2, conf, name), (cx, cy)))

            unmatched = set(self.objects.keys())

            # 📌 KEY CONCEPT: Greedy Nearest-Neighbour Matching (O(N²))
            # Each new detection is matched to the closest existing track using Euclidean distance
            # ── Match each detection to nearest existing tracked object ──
            for det, centroid in centroids:
                best_id, best_dist = None, 1e9
                for oid, obj in self.objects.items():
                    ox, oy = obj["centroid"]
                    dist = math.hypot(ox - centroid[0], oy - centroid[1])
                    if dist < best_dist and dist < 100:
                        best_dist = dist
                        best_id = oid

                # New object if no match found
                if best_id is None:
                    oid = self.next_id
                    self.next_id += 1
                else:
                    oid = best_id
                    if oid in unmatched:
                        unmatched.remove(oid)

                x1, y1, x2, y2, conf, name = det
                cx, cy = centroid
                new_objects[oid] = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "centroid": (cx, cy),
                    "name": name,
                    "conf": float(conf),   # ← preserve confidence for EMA in rules.py
                    "age": 0,
                    "hits": self.objects.get(oid, {}).get("hits", 0) + 1
                }

            # 📌 KEY CONCEPT: Max-Age Dead Reckoning (Track Persistence)
            # Tracks that had no matching detection are kept alive up to max_age frames — handles brief occlusions
            # ── Keep unmatched objects alive for max_age frames ──────────
            for oid in unmatched:
                obj = self.objects.get(oid)
                if obj is not None:
                    obj["age"] = obj.get("age", 0) + 1
                    if obj["age"] <= self.max_age:
                        new_objects[oid] = obj

            self.objects = new_objects

            # 📌 KEY CONCEPT: Adapter / Wrapper Class (SimpleTrack)
            # Wraps a plain dict into an object with the same interface as DeepSORT tracks —
            # so the rest of the pipeline (rules.py, visualize.py) works identically for both trackers
            # ── Wrap raw dicts into track-like objects ─────────────────────
            class SimpleTrack:
                def __init__(self, tid, obj):
                    self.track_id = tid
                    self._obj = obj

                def is_confirmed(self):
                    return self._obj.get("hits", 0) > 0

                def to_ltrb(self):
                    x1, y1, x2, y2 = self._obj["bbox"]
                    return [x1, y1, x2, y2]

                def get_det_class(self):
                    return self._obj.get("name", "unknown")

                def get_det_conf(self):
                    # Returns latest detection confidence (preserved from detection.py output)
                    return self._obj.get("conf", 0.0)

            return [SimpleTrack(tid, obj) for tid, obj in self.objects.items()]
