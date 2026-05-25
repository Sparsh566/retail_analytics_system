# detector.py
# Optimized YOLOv8 object detector and synthetic CCTV retail simulator.
# Employs Threaded Camera Grabbing to completely eliminate webcam hardware capture latency.

import cv2
import numpy as np
import time
import random
import threading
from src import config

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class Detector:
    """
    Manages camera streams in an independent thread and performs optimized YOLOv8 inference.
    """
    def __init__(self):
        self.simulation_mode = False
        self.yolo_model = None
        self.cap = None
        
        # Threaded Camera Grabbing states
        self.latest_raw_frame = None
        self.camera_thread_active = False
        self.cam_thread = None
        
        # Try loading YOLOv8
        if YOLO_AVAILABLE:
            try:
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
                    # Test grab a single frame immediately to ensure camera device is functional
                    ret, test_frame = self.cap.read()
                    if not ret or test_frame is None:
                        print("[WARN] Web camera opened but failed to grab test frame. Using Shop Simulator.")
                        self.cap.release()
                        self.cap = None
                        self.simulation_mode = True
                    else:
                        self.cap.set(cv3_width_prop_index(), config.FRAME_WIDTH)
                        self.cap.set(cv3_height_prop_index(), config.FRAME_HEIGHT)
                        
                        # Spawn independent background thread for threaded frame grabbing
                        self.camera_thread_active = True
                        self.cam_thread = threading.Thread(target=self._grab_camera_frames, daemon=True)
                        self.cam_thread.start()
                        print("[INFO] Threaded Camera Grabbing activated successfully.")
            except Exception as e:
                print(f"[WARN] Camera capture init failed: {e}. Using Shop Simulator.")
                self.simulation_mode = True

        # Initialize Simulation Engine State
        if self.simulation_mode:
            self._init_simulation()

    def _grab_camera_frames(self):
        """
        Runs continuously in a separate thread to poll hardware frames as fast as possible.
        """
        consecutive_failures = 0
        while self.camera_thread_active:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_raw_frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        print("[WARN] Web camera failing repeatedly. Disabling camera capture and switching to simulation.")
                        self.camera_thread_active = False
                        self.simulation_mode = True
            time.sleep(0.01) # Yield CPU briefly

    def get_frame(self):
        """
        Returns latest processed camera or simulation frame instantly without blocking.
        """
        if self.simulation_mode:
            return self._generate_simulated_frame()

        # Grab latest threaded frame
        frame = self.latest_raw_frame
        
        # If camera hasn't finished loading the first frame, wait briefly
        if frame is None:
            time.sleep(0.1)
            frame = self.latest_raw_frame
            if frame is None:
                print("[WARN] Camera frame buffer timeout. Activating Shop Simulator.")
                self.simulation_mode = True
                self._init_simulation()
                return self._generate_simulated_frame()

        # Resize frame to standardized processing size
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        # Run YOLO Inference (optimized image size imgsz=320 for 300% faster CPU inference)
        detections = []
        try:
            results = self.yolo_model(frame, imgsz=320, verbose=False)[0]
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo_model.names[cls_id]
                
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
        Clean release of hardware handles.
        """
        self.camera_thread_active = False
        if self.cam_thread:
            self.cam_thread.join(timeout=0.5)
            
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

    # -------------------------------------------------------------------------
    # SYNTHETIC RETAIL CAMERA SIMULATOR ENGINE
    # -------------------------------------------------------------------------
    def _init_simulation(self):
        print("[INFO] Initializing High-Fidelity 2D Retail Shop CCTV Simulator...")
        self.sim_frame_count = 0
        self.zones = [
            {"name": "ENTRANCE", "box": (30, 150, 90, 210), "color": (50, 50, 50)},
            {"name": "APPAREL ZONE", "box": (150, 40, 280, 140), "color": (40, 40, 40)},
            {"name": "ELECTRONICS", "box": (350, 40, 480, 140), "color": (40, 40, 40)},
            {"name": "FOOD & FRESH", "box": (150, 220, 280, 320), "color": (40, 40, 40)},
            {"name": "CHECKOUT 1", "box": (420, 220, 500, 320), "color": (60, 50, 40)}
        ]
        self.shoppers = []
        self.next_shopper_id = 1
        for _ in range(5):
            self._spawn_shopper()

    def _spawn_shopper(self):
        entrance = (50, 180)
        x = entrance[0] + random.randint(-15, 15)
        y = entrance[1] + random.randint(-25, 25)
        w = random.randint(30, 45)
        h = random.randint(70, 95)
        waypoints = [
            (215, 90),   # Apparel center
            (415, 90),   # Electronics center
            (215, 270),  # Food center
            (460, 270),  # Checkout
            (50, 180)    # Exit
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
            "state": "walking",
            "class_name": "person",
            "confidence": random.uniform(0.85, 0.99),
            "color": config.get_id_color_bgr(self.next_shopper_id)
        })
        self.next_shopper_id += 1

    def _generate_simulated_frame(self):
        self.sim_frame_count += 1
        frame = np.ones((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8) * 10
        
        for y in range(0, config.FRAME_HEIGHT, 40):
            cv2.line(frame, (0, y), (config.FRAME_WIDTH, y), (18, 18, 18), 1)
        for x in range(0, config.FRAME_WIDTH, 40):
            cv2.line(frame, (x, 0), (x, config.FRAME_HEIGHT), (18, 18, 18), 1)
            
        for zone in self.zones:
            x1, y1, x2, y2 = zone["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), config_to_bgr(config.COLOR_BORDER), 1)
            cv2.putText(
                frame, zone["name"], (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA
            )

        active_detections = []
        for idx in range(len(self.shoppers) - 1, -1, -1):
            s = self.shoppers[idx]
            tx, ty = s["waypoints"][s["current_wp_idx"]]
            
            if s["state"] == "walking":
                dx = tx - s["x"]
                dy = ty - s["y"]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 5.0:
                    s["state"] = "browsing"
                    s["dwell_ticks"] = 0
                else:
                    s["x"] += (dx / dist) * s["speed"]
                    s["y"] += (dy / dist) * s["speed"]
            elif s["state"] == "browsing":
                s["dwell_ticks"] += 1
                if s["dwell_ticks"] >= s["browse_ticks_limit"]:
                    s["current_wp_idx"] = (s["current_wp_idx"] + 1) % len(s["waypoints"])
                    if s["current_wp_idx"] == 0 and random.random() < 0.4:
                        self.shoppers.pop(idx)
                        continue
                    s["state"] = "walking"
                    s["browse_ticks_limit"] = random.randint(30, 80)
            
            cx, cy = int(s["x"]), int(s["y"])
            hw, hh = s["w"] // 2, s["h"] // 2
            cv2.circle(frame, (cx, cy), 10, s["color"], -1)
            cv2.ellipse(frame, (cx, cy), (18, 8), 0, 0, 360, s["color"], 2)
            
            x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
            x1 += random.randint(-2, 2)
            y1 += random.randint(-2, 2)
            x2 += random.randint(-2, 2)
            y2 += random.randint(-2, 2)
            
            if random.random() < 0.02:
                continue
                
            active_detections.append([x1, y1, x2, y2, s["confidence"], s["class_name"]])

        if len(self.shoppers) < 5:
            if random.random() < 0.05:
                self._spawn_shopper()

        if random.random() < 0.03:
            fx = random.randint(150, 450)
            fy = random.randint(40, 300)
            active_detections.append([
                fx - 15, fy - 15, fx + 15, fy + 15,
                random.uniform(0.51, 0.65), random.choice(["bag", "chair", "umbrella"])
            ])

        cv2.putText(
            frame, "SIMULATOR CAM_01 // SECURE FEED", (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, config_to_bgr(config.COLOR_CYAN), 1, cv2.LINE_AA
        )
        cv2.circle(frame, (config.FRAME_WIDTH - 25, 20), 5, (0, 0, 255), -1)
        
        return frame, active_detections, True

def cv3_width_prop_index():
    return cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3

def cv3_height_prop_index():
    return cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4

def config_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)
