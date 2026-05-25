# detector.py
# YOLOv8 object detector and high-fidelity synthetic CCTV retail simulator.
# Handles live camera feeds and provides fallback simulation data.

import cv2
import numpy as np
import time
import random
from src import config

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class Detector:
    """
    Manages the camera feed and performs YOLOv8 inference.
    If camera is unavailable or YOLO fails, it runs a fully functional shop simulator.
    """
    def __init__(self):
        self.simulation_mode = False
        self.yolo_model = None
        self.cap = None
        
        # Try loading YOLOv8
        if YOLO_AVAILABLE:
            try:
                # Load tiny/nano model. Will attempt download if not present locally
                self.yolo_model = YOLO(config.YOLO_MODEL)
                print(f"[INFO] YOLOv8 model '{config.YOLO_MODEL}' loaded successfully.")
            except Exception as e:
                print(f"[WARN] Failed to load YOLOv8 model: {e}. Switching to Simulation Mode.")
                self.simulation_mode = True
        else:
            print("[WARN] ultralytics package not available. Switching to Simulation Mode.")
            self.simulation_mode = True

        # Try initializing Web Camera
        if not self.simulation_mode:
            try:
                self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
                if not self.cap.isOpened():
                    print(f"[WARN] Web camera index {config.CAMERA_INDEX} not accessible. Using Shop Simulator.")
                    self.simulation_mode = True
                else:
                    # Configure camera properties
                    self.cap.set(cv3_width_prop_index(), config.FRAME_WIDTH)
                    self.cap.set(cv3_height_prop_index(), config.FRAME_HEIGHT)
            except Exception as e:
                print(f"[WARN] Camera capture init failed: {e}. Using Shop Simulator.")
                self.simulation_mode = True

        # Initialize Simulation Engine State
        if self.simulation_mode:
            self._init_simulation()

    def get_frame(self):
        """
        Grabs next frame and returns:
          frame: BGR image
          detections: list of [x1, y1, x2, y2, confidence, class_name]
          is_simulated: bool
        """
        if self.simulation_mode:
            return self._generate_simulated_frame()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to grab frame from webcam. Defaulting to Simulator.")
            self.simulation_mode = True
            self._init_simulation()
            return self._generate_simulated_frame()

        # Resize frame to standardized processing size
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        # Run YOLO Inference
        detections = []
        try:
            results = self.yolo_model(frame, verbose=False)[0]
            for box in results.boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo_model.names[cls_id]
                
                # Check confidence threshold
                if conf >= config.CONFIDENCE_THRESHOLD:
                    detections.append([
                        float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                        conf, cls_name
                    ])
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}. Falling back to simulation.")
            self.simulation_mode = True
            self._init_simulation()
            return self._generate_simulated_frame()

        return frame, detections, False

    def close(self):
        """
        Releases system resources.
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

    # -------------------------------------------------------------------------
    # SYNTHETIC RETAIL CAMERA SIMULATOR ENGINE
    # -------------------------------------------------------------------------
    def _init_simulation(self):
        """
        Initializes virtual shoppers and retail shop zones for 2D simulation.
        """
        print("[INFO] Initializing High-Fidelity 2D Retail Shop CCTV Simulator...")
        self.sim_frame_count = 0
        
        # Shop floor landmarks (Aisles, Shelves, Checkout)
        self.zones = [
            {"name": "ENTRANCE", "box": (30, 150, 90, 210), "color": (50, 50, 50)},
            {"name": "APPAREL ZONE", "box": (150, 40, 280, 140), "color": (40, 40, 40)},
            {"name": "ELECTRONICS", "box": (350, 40, 480, 140), "color": (40, 40, 40)},
            {"name": "FOOD & FRESH", "box": (150, 220, 280, 320), "color": (40, 40, 40)},
            {"name": "CHECKOUT 1", "box": (420, 220, 500, 320), "color": (60, 50, 40)}
        ]
        
        # Shopper agents
        self.shoppers = []
        self.next_shopper_id = 1
        
        # Spawn initial shoppers
        for _ in range(5):
            self._spawn_shopper()

    def _spawn_shopper(self):
        """
        Spawns a single customer with random browsing waypoints.
        """
        # Start at Entrance or a random position
        entrance = (50, 180)
        x = entrance[0] + random.randint(-15, 15)
        y = entrance[1] + random.randint(-25, 25)
        
        # Bounding box sizing
        w = random.randint(30, 45)
        h = random.randint(70, 95)
        
        # Shopping behavior profile
        waypoints = [
            (215, 90),   # Apparel center
            (415, 90),   # Electronics center
            (215, 270),  # Food center
            (460, 270),  # Checkout
            (50, 180)    # Exit (Entrance)
        ]
        
        self.shoppers.append({
            "id": self.next_shopper_id,
            "x": float(x),
            "y": float(y),
            "w": w,
            "h": h,
            "waypoints": waypoints,
            "current_wp_idx": random.randint(0, len(waypoints) - 1),
            "speed": random.uniform(1.2, 2.2),
            "dwell_ticks": 0,
            "browse_ticks_limit": random.randint(30, 80),
            "state": "walking", # walking or browsing
            "class_name": "person",
            "confidence": random.uniform(0.85, 0.99),
            "color": config.get_id_color_bgr(self.next_shopper_id)
        })
        self.next_shopper_id += 1

    def _generate_simulated_frame(self):
        """
        Updates virtual shoppers, processes occlusions, injects detector noise, 
        and renders a dark security monitor feed.
        """
        self.sim_frame_count += 1
        
        # 1. Base CCTV Background Frame (Dark Charcoal Palette)
        frame = np.ones((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8) * 10
        
        # Draw dynamic security scanning scanlines and grid overlay
        for y in range(0, config.FRAME_HEIGHT, 40):
            cv2.line(frame, (0, y), (config.FRAME_WIDTH, y), (18, 18, 18), 1)
        for x in range(0, config.FRAME_WIDTH, 40):
            cv2.line(frame, (x, 0), (x, config.FRAME_HEIGHT), (18, 18, 18), 1)
            
        # Draw Shop Layout zones
        for zone in self.zones:
            x1, y1, x2, y2 = zone["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), config_to_bgr(config.COLOR_BORDER), 1)
            
            # Subdued monospace labelling on floor plan
            cv2.putText(
                frame, zone["name"], (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA
            )

        # 2. Update Virtual Shoppers Position & States
        active_detections = []
        
        for idx in range(len(self.shoppers) - 1, -1, -1):
            s = self.shoppers[idx]
            
            # Destination Waypoint
            tx, ty = s["waypoints"][s["current_wp_idx"]]
            
            if s["state"] == "walking":
                # Vector to target
                dx = tx - s["x"]
                dy = ty - s["y"]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 5.0:
                    s["state"] = "browsing"
                    s["dwell_ticks"] = 0
                else:
                    # Move towards waypoint
                    s["x"] += (dx / dist) * s["speed"]
                    s["y"] += (dy / dist) * s["speed"]
            elif s["state"] == "browsing":
                s["dwell_ticks"] += 1
                if s["dwell_ticks"] >= s["browse_ticks_limit"]:
                    # Select next waypoint in loop
                    s["current_wp_idx"] = (s["current_wp_idx"] + 1) % len(s["waypoints"])
                    # If they looped back to entrance, they exited!
                    if s["current_wp_idx"] == 0 and random.random() < 0.4:
                        self.shoppers.pop(idx)
                        continue
                    s["state"] = "walking"
                    s["browse_ticks_limit"] = random.randint(30, 80)
            
            # Draw Stylized Customer Avatar on Background (overhead perspective)
            cx, cy = int(s["x"]), int(s["y"])
            hw, hh = s["w"] // 2, s["h"] // 2
            
            # Draw outline path
            cv2.circle(frame, (cx, cy), 10, s["color"], -1) # Head
            cv2.ellipse(frame, (cx, cy), (18, 8), 0, 0, 360, s["color"], 2) # Shoulders
            
            # Generate detection bounding box (bounding box corners)
            x1 = cx - hw
            y1 = cy - hh
            x2 = cx + hw
            y2 = cy + hh
            
            # 3. Inject Synthetic Detection Noise (To test SORT vs DeepSORT)
            # A) Bounding Box Coordinate Jitter
            x1 += random.randint(-2, 2)
            y1 += random.randint(-2, 2)
            x2 += random.randint(-2, 2)
            y2 += random.randint(-2, 2)
            
            # B) Occlusion Logic / Crossing Paths
            # Occurs naturally when two shoppers' coordinates overlap.
            
            # C) Missed Detections (2% frame drop rate to test tracker memory age)
            if random.random() < 0.02:
                continue  # Skip adding this detection for this frame!
                
            active_detections.append([x1, y1, x2, y2, s["confidence"], s["class_name"]])

        # Ingest dynamic spawning (maintain at least 4 customers)
        if len(self.shoppers) < 5:
            if random.random() < 0.05: # Gradually spawn
                self._spawn_shopper()

        # D) Inject False Positives (occasional noise: bag or laptop on shelves)
        if random.random() < 0.03:
            fx = random.randint(150, 450)
            fy = random.randint(40, 300)
            active_detections.append([
                fx - 15, fy - 15, fx + 15, fy + 15,
                random.uniform(0.51, 0.65), random.choice(["bag", "chair", "umbrella"])
            ])

        # Render top corner status bar on security feed
        cv2.putText(
            frame, "SIMULATOR CAM_01 // SECURE FEED", (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, config_to_bgr(config.COLOR_CYAN), 1, cv2.LINE_AA
        )
        cv2.circle(frame, (config.FRAME_WIDTH - 25, 20), 5, (0, 0, 255), -1) # Red blinking rec dot
        
        return frame, active_detections, True

# -------------------------------------------------------------------------
# COMPATIBILITY HELPERS
# -------------------------------------------------------------------------
def cv3_width_prop_index():
    return cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3

def cv3_height_prop_index():
    return cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4

def config_to_bgr(hex_color):
    """
    Converts Tkinter config Hex color to OpenCV BGR color tuple.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)
