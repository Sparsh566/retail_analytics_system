# deepsort_tracker.py
# DeepSORT tracker wrapper.
# Integrates deep-sort-realtime library and counts ID switches.

import numpy as np

# Attempt to import deep-sort-realtime with standard fallback for robust environment safety
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

class DeepSortTracker:
    """
    Wrapper around deep-sort-realtime to align output formats and count ID switches deterministically.
    """
    def __init__(self, max_age=5, n_init=3):
        self.max_age = max_age
        self.n_init = n_init
        self.id_switches = 0
        self.frame_count = 0
        self.dead_track_history = []
        
        # Check if deep_sort_realtime is installed, otherwise run spatial-only emulation
        if DEEPSORT_AVAILABLE:
            try:
                # Load DeepSort with CPU/GPU optimized MobileNet embedder
                self.tracker = DeepSort(
                    max_age=self.max_age,
                    n_init=self.n_init,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3,
                    embedder="mobilenet",  # Pre-trained deep re-identification features
                    half=False,            # FP32 computation for general CPU compatibility
                    bgr=True
                )
                self.is_emulated = False
            except Exception as e:
                print(f"[WARN] Failed to load DeepSORT embedder: {e}. Falling back to spatial-only tracking.")
                self.is_emulated = True
        else:
            print("[WARN] deep-sort-realtime library not found. Falling back to spatial-only tracking.")
            self.is_emulated = True

        if self.is_emulated:
            # Re-use our custom SORT logic as a fallback to prevent app crashing
            from src.trackers.sort import SortTracker
            self.emulated_tracker = SortTracker(max_age=max_age, min_hits=n_init, iou_threshold=0.2)

    def update(self, detections, frame):
        """
        detections: list of detections, each [x1, y1, x2, y2, confidence, class_name]
        frame: raw OpenCV BGR image (used by DeepSORT's appearance embedder)
        Returns: list of active tracks, each [x1, y1, x2, y2, track_id, class_name, confidence]
        """
        self.frame_count += 1
        
        # Clean expired dead track history (keep last 30 frames)
        self.dead_track_history = [
            d for d in self.dead_track_history if self.frame_count - d[2] <= 30
        ]
        
        if self.is_emulated:
            # Run emulated mode with custom spatial SORT logic
            tracks = self.emulated_tracker.update(detections)
            self.id_switches = self.emulated_tracker.id_switches
            self.active_count = self.emulated_tracker.active_count
            self.lost_count = self.emulated_tracker.lost_count
            return tracks

        # 1. Format detections for deep-sort-realtime: ([left, top, w, h], confidence, class_name)
        ds_dets = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_name = det
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            ds_dets.append(([x1, y1, w, h], conf, cls_name))
            
        # 2. Perform tracking inference
        raw_tracks = self.tracker.update_tracks(ds_dets, frame=frame)
        
        active_tracks = []
        active_ids = set()
        
        # 3. Process track outputs
        for trk in raw_tracks:
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
                
            # Get bounding box in tlbr [x1, y1, x2, y2]
            tlbr = trk.to_tlbr()
            track_id = trk.track_id
            class_name = trk.get_class()
            confidence = trk.get_det_conf() if trk.get_det_conf() is not None else 1.0
            
            # Filter clean integer outputs
            x1, y1, x2, y2 = float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])
            active_tracks.append([x1, y1, x2, y2, track_id, class_name, confidence])
            active_ids.add(track_id)
            
            # --- Real-Time ID Switch Heuristics (matches SORT logic for a fair comparison) ---
            # If a track ID is confirmed for the first time, check if it was born in the vicinity
            # of a recently terminated track.
            if trk.hits == self.n_init:
                new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                for dead_center, dead_class, _ in self.dead_track_history:
                    if dead_class == class_name:
                        dist = np.sqrt((new_center[0] - dead_center[0])**2 + (new_center[1] - dead_center[1])**2)
                        # Appearance embeddings make DeepSORT robust; ID switches should occur less frequently.
                        if dist < 45.0:
                            self.id_switches += 1
                            break

        # 4. Detect track terminations and append to dead history
        # We find which tracks were confirmed but are now dead or lost
        # Let's inspect tracker active tracks internally
        for trk in self.tracker.tracks:
            if trk.time_since_update == self.max_age:
                tlbr = trk.to_tlbr()
                dead_center = ((tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2)
                self.dead_track_history.append((dead_center, trk.get_class(), self.frame_count))

        # Expose active and lost tracking counts for the dashboard
        self.active_count = sum(1 for t in self.tracker.tracks if t.is_confirmed() and t.time_since_update == 0)
        self.lost_count = sum(1 for t in self.tracker.tracks if t.is_confirmed() and t.time_since_update > 0)

        return active_tracks
