# sort_tracker.py
# Custom SORT (Simple Online and Realtime Tracking) implementation.
# Uses filterpy's KalmanFilter and scipy's Hungarian algorithm (linear_sum_assignment).

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes: [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / float(union_area)

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 1  # Track IDs start from 1

    def __init__(self, bbox, class_name="unknown", confidence=0.0):
        """
        Initializes a tracker using a bounding box [x1, y1, x2, y2].
        """
        # State vector: [u, v, s, r, u_dot, v_dot, s_dot]^T
        # u, v: center coordinates. s: scale (area). r: aspect ratio.
        # u_dot, v_dot, s_dot: velocity components.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix F
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix H (we only measure [u, v, s, r])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariance matrices tuned for pedestrian/object motion
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # High initial uncertainty for velocity
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state with measured coordinates
        self.kf.x[:4] = self.bbox_to_z(bbox)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_name = class_name
        self.confidence = confidence

    def update(self, bbox, class_name, confidence):
        """
        Updates the state vector with a new observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z(bbox))
        self.class_name = class_name
        self.confidence = confidence

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.z_to_bbox(self.kf.x[:4]))
        return self.history[-1][0]

    def get_state(self):
        """
        Returns the current bounding box estimate [x1, y1, x2, y2].
        """
        return self.z_to_bbox(self.kf.x[:4])[0]

    def bbox_to_z(self, bbox):
        """
        Converts bounding box [x1, y1, x2, y2] to measurement vector [u, v, s, r]^T.
        """
        w = max(1e-5, bbox[2] - bbox[0])
        h = max(1e-5, bbox[3] - bbox[1])
        u = bbox[0] + w / 2.0
        v = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([u, v, s, r]).reshape((4, 1))

    def z_to_bbox(self, z):
        """
        Converts state measurement vector [u, v, s, r]^T to bounding box [x1, y1, x2, y2].
        """
        u, v, s, r = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
        w = np.sqrt(max(0.0, s * r))
        h = np.sqrt(max(0.0, s / r)) if r > 0 else 0.0
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0
        return np.array([x1, y1, x2, y2]).reshape((1, 4))


class SortTracker:
    """
    Manages multiple KalmanBoxTrackers and associates them to detections using IoU matching.
    """
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
        # Real-time ID Switch Heuristics (identity fragmentation analysis)
        self.id_switches = 0
        self.dead_track_history = []  # Records (box_center, class_name, frame_index)

    def update(self, detections):
        """
        detections: list of detections, each [x1, y1, x2, y2, confidence, class_name]
        Returns: list of active tracks, each [x1, y1, x2, y2, track_id, class_name, confidence]
        """
        self.frame_count += 1
        
        # 1. Advance all existing trackers and get their predictions
        predictions = []
        for t in range(len(self.trackers) - 1, -1, -1):
            pred = self.trackers[t].predict()
            # Remove trackers with invalid states
            if np.any(np.isnan(pred)):
                self.trackers.pop(t)
            else:
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Clean expired dead track history (keep last 30 frames)
        self.dead_track_history = [
            d for d in self.dead_track_history if self.frame_count - d[2] <= 30
        ]
        
        # 2. Hungarian Association
        matched, unmatched_dets, unmatched_trks = [], [], []
        
        if len(detections) == 0:
            unmatched_trks = list(range(len(self.trackers)))
        elif len(self.trackers) == 0:
            unmatched_dets = list(range(len(detections)))
        else:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
            for t, trk in enumerate(self.trackers):
                trk_box = trk.get_state()
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = calculate_iou(trk_box, det[:4])
            
            # Solve matching (scipy Hungarian algorithm maximizes IoU, so cost is negative IoU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            # Filter matches by threshold
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append((r, c))
            
            # Identify remaining unmatched
            for t in range(len(self.trackers)):
                if t not in row_ind:
                    unmatched_trks.append(t)
            for d in range(len(detections)):
                if d not in col_ind:
                    unmatched_dets.append(d)
        
        # 3. Update matched trackers
        for r, c in matched:
            det = detections[c]
            self.trackers[r].update(det[:4], det[5], det[4])
            
        # 4. Handle unmatched detections -> Create brand-new trackers
        for c in unmatched_dets:
            det = detections[c]
            new_trk = KalmanBoxTracker(det[:4], det[5], det[4])
            self.trackers.append(new_trk)
            
            # --- Real-Time ID Switch Heuristics ---
            # Check if this new tracker is starting close to a recently lost track of the same class.
            # If yes, we count it as an ID Switch (identity fragmentation due to occlusion or poor IoU)
            new_box = det[:4]
            new_center = ((new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2)
            
            for dead_center, dead_class, _ in self.dead_track_history:
                if dead_class == det[5]:
                    dist = np.sqrt((new_center[0] - dead_center[0])**2 + (new_center[1] - dead_center[1])**2)
                    if dist < 45.0:  # Threshold of 45 pixels
                        self.id_switches += 1
                        break
                        
        # 5. Handle unmatched trackers -> Check age and remove if expired
        ret = []
        for t in range(len(self.trackers) - 1, -1, -1):
            trk = self.trackers[t]
            # If the tracker has been updated in this frame
            if trk.time_since_update < 1:
                # Minimum hit streak required before displaying the track
                if (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                    bbox = trk.get_state()
                    ret.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.id, trk.class_name, trk.confidence])
            
            # If track is dead (too many lost frames)
            if trk.time_since_update > self.max_age:
                dead_box = trk.get_state()
                dead_center = ((dead_box[0] + dead_box[2]) / 2, (dead_box[1] + dead_box[3]) / 2)
                # Store the dead track's spatial exit point and metadata
                self.dead_track_history.append((dead_center, trk.class_name, self.frame_count))
                self.trackers.pop(t)
                
        # Expose active and lost tracking counts for the dashboard
        self.active_count = sum(1 for t in self.trackers if t.time_since_update == 0)
        self.lost_count = sum(1 for t in self.trackers if t.time_since_update > 0)
        
        return ret
