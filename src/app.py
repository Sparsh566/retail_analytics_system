# src/app.py
# Web-native concurrent orchestration controller using Flask.
# Implements Decoupled Queue-Based Parallel Threading to run trackers concurrently.

import os
import sys
import time
import signal
import threading
import queue
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from src import config
from src.vision.detector import Detector
from src.trackers.sort import SortTracker
from src.trackers.deepsort import DeepSortTracker

class MainController:
    """
    Concurrent pipeline orchestrator hosting the Flask server.
    """
    def __init__(self):
        self.app = Flask(__name__)
        
        # State variables
        self.is_paused = False
        self.loop_active = False
        self.lock = threading.Lock()
        
        # Thread handles
        self.det_thread = None
        self.sort_thread = None
        self.ds_thread = None
        
        # Thread-safe queues with a capacity of 1 to ensure zero-latency lag (always newest frame)
        self.sort_queue = queue.Queue(maxsize=1)
        self.deepsort_queue = queue.Queue(maxsize=1)
        
        # Subsystems
        self.detector = None
        self.sort_tracker = None
        self.deepsort_tracker = None
        
        # JPEG streaming frame buffers
        self.latest_sort_frame = None
        self.latest_ds_frame = None
        
        # Analytical Registers
        self.sort_unique_people = set()
        self.ds_unique_people = set()
        self.dwell_times_sort = {}
        self.dwell_times_ds = {}
        self.prev_sort_persons = set()
        self.prev_ds_persons = set()
        
        # Session Event Logs
        self.event_logs = []
        
        # Global stats dictionary
        self.stats = {
            "sort_active": 0, "sort_lost": 0,
            "ds_active": 0, "ds_lost": 0,
            "fps_sort": 0.0, "fps_ds": 0.0,
            "dets_count": 0,
            "switches_sort": 0, "switches_ds": 0,
            "footfall": 0,
            "dwell_list": "",
            "tracked_objects": [],
            "status_text": "PAUSED - CLICK START TO INITIALIZE",
            "simulation_mode": False
        }
        
        # Bind flask endpoints
        self._setup_routes()

    # -------------------------------------------------------------------------
    # ROUTE DEFINITIONS
    # -------------------------------------------------------------------------
    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return Response(self._get_html_page(), mimetype="text/html")
            
        @self.app.route("/video_feed/sort")
        def video_feed_sort():
            return Response(self._stream_frame("sort"), mimetype="multipart/x-mixed-replace; boundary=frame")
            
        @self.app.route("/video_feed/deepsort")
        def video_feed_deepsort():
            return Response(self._stream_frame("deepsort"), mimetype="multipart/x-mixed-replace; boundary=frame")
            
        @self.app.route("/api/start")
        def api_start():
            if not self.loop_active:
                self.start_tracking()
            return jsonify({"status": "started"})
            
        @self.app.route("/api/stats")
        def api_stats():
            with self.lock:
                return jsonify({
                    "stats": self.stats,
                    "logs": self.event_logs[-15:]  # Return last 15 log lines
                })
                
        @self.app.route("/api/pause", methods=["POST"])
        def api_pause():
            self.is_paused = not self.is_paused
            with self.lock:
                if self.is_paused:
                    self.stats["status_text"] = "PAUSED - SENSOR GRABBING DEACTIVATED"
                    self._add_log("SYSTEM WARNING: Live pipeline paused by operator.")
                else:
                    self.stats["status_text"] = "RUNNING - BOTH TRACKERS ACTIVE"
                    self._add_log("SYSTEM INFO: Live pipeline tracking resumed.")
            return jsonify({"paused": self.is_paused})
            
        @self.app.route("/api/reset", methods=["POST"])
        def api_reset():
            self.reset_system_stats()
            self._add_log("SYSTEM ACTION: Analytical records and trackers reset completed.")
            return jsonify({"status": "reset"})
            
        @self.app.route("/api/quit", methods=["POST"])
        def api_quit():
            self._add_log("SYSTEM WARNING: Server termination requested. Shuting down...")
            self.stop_tracking()
            os.kill(os.getpid(), signal.SIGINT)
            return jsonify({"status": "quitting"})

    # -------------------------------------------------------------------------
    # PIPELINE LIFECYCLE MANAGEMENT
    # -------------------------------------------------------------------------
    def start_tracking(self):
        """
        Loads models, creates queues, and spawns the three concurrent pipeline threads.
        """
        # Init components
        self.detector = Detector()
        self.sort_tracker = SortTracker(max_age=15, min_hits=3, iou_threshold=0.25)
        self.deepsort_tracker = DeepSortTracker(max_age=15, n_init=3)
        
        self.reset_system_stats()
        self._add_log("SYSTEM ACTION: Activating concurrent sensors and tracking threads...")
        
        # Reset queues
        self.sort_queue = queue.Queue(maxsize=1)
        self.deepsort_queue = queue.Queue(maxsize=1)
        
        # Spawn three parallel threads
        self.loop_active = True
        self.det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.sort_thread = threading.Thread(target=self._sort_loop, daemon=True)
        self.ds_thread = threading.Thread(target=self._deepsort_loop, daemon=True)
        
        self.det_thread.start()
        self.sort_thread.start()
        self.ds_thread.start()
        print("[INFO] Decoupled Queue-Based Parallel Threads initialized successfully.")

    def stop_tracking(self):
        """
        Stops loops and joins threads cleanly.
        """
        self.loop_active = False
        
        # Clear queues to unblock threads if they are waiting
        try:
            self.sort_queue.put_nowait(None)
            self.deepsort_queue.put_nowait(None)
        except:
            pass
            
        if self.det_thread:
            self.det_thread.join(timeout=0.5)
        if self.sort_thread:
            self.sort_thread.join(timeout=0.5)
        if self.ds_thread:
            self.ds_thread.join(timeout=0.5)
            
        if self.detector:
            self.detector.close()
            self.detector = None

    def reset_system_stats(self):
        """
        Wipes analytics registers and re-instantiates trackers.
        """
        with self.lock:
            self.sort_unique_people.clear()
            self.ds_unique_people.clear()
            self.dwell_times_sort.clear()
            self.dwell_times_ds.clear()
            self.prev_sort_persons.clear()
            self.prev_ds_persons.clear()
            self.event_logs.clear()
            
            if self.sort_tracker:
                self.sort_tracker.id_switches = 0
                self.sort_tracker.trackers.clear()
                self.sort_tracker.dead_track_history.clear()
                
            if self.deepsort_tracker:
                if not self.deepsort_tracker.is_emulated:
                    self.deepsort_tracker = DeepSortTracker(max_age=15, n_init=3)
                else:
                    self.deepsort_tracker.id_switches = 0
                    self.deepsort_tracker.dead_track_history.clear()
                    self.deepsort_tracker.emulated_tracker.id_switches = 0
                    self.deepsort_tracker.emulated_tracker.trackers.clear()
                    self.deepsort_tracker.emulated_tracker.dead_track_history.clear()

    # -------------------------------------------------------------------------
    # CONCURRENT THREAD LOOPS
    # -------------------------------------------------------------------------
    def _detection_loop(self):
        """
        THREAD 1: Grabs frames and runs YOLOv8 ONCE, then pushes to concurrent queues.
        """
        while self.loop_active:
            if self.is_paused:
                time.sleep(0.05)
                continue
                
            # Grab raw frame and execute optimized YOLOv8 (imgsz=320)
            frame, detections, is_simulated = self.detector.get_frame()
            
            # Feed frame and detections to SORT and DeepSORT queues
            # Drop older frames if queues are full to guarantee real-time latency
            for q in [self.sort_queue, self.deepsort_queue]:
                if q.full():
                    try:
                        q.get_nowait() # Discard old frame
                    except queue.Empty:
                        pass
                try:
                    q.put((frame.copy(), detections.copy(), is_simulated), block=False)
                except queue.Full:
                    pass
                    
            with self.lock:
                mode_text = "SIMULATION ACTIVE" if is_simulated else "LIVE WEBCAM ACTIVE"
                self.stats["status_text"] = f"RUNNING - BOTH TRACKERS ACTIVE ({mode_text})"
                self.stats["simulation_mode"] = is_simulated
                self.stats["dets_count"] = len(detections)
                
            time.sleep(0.01) # Yield to parallel threads

    def _sort_loop(self):
        """
        THREAD 2: Pulls from sort_queue, runs custom SORT, and updates SORT JPEG feed instantly.
        """
        while self.loop_active:
            try:
                item = self.sort_queue.get(timeout=0.1)
                if item is None: # Exit sentinel
                    break
            except queue.Empty:
                continue
                
            frame, detections, is_simulated = item
            
            # Benchmark SORT
            t0 = time.perf_counter()
            sort_tracks = self.sort_tracker.update(detections)
            t_sort = time.perf_counter() - t0
            fps_sort = 1.0 / max(1e-5, t_sort)
            
            # Draw green overlays
            self._draw_overlay(frame, sort_tracks, theme_color=(0, 255, 136))
            sort_resized = cv2.resize(frame, (384, 216))
            _, sort_jpg = cv2.imencode('.jpg', sort_resized)
            
            # Update Dwell Times & Log entries
            current_time = time.time()
            active_sort_persons = set()
            for trk in sort_tracks:
                x1, y1, x2, y2, trk_id, class_name, conf = trk
                if class_name == config.RETAIL_CLASS_FILTER:
                    active_sort_persons.add(trk_id)
                    with self.lock:
                        self.sort_unique_people.add(trk_id)
                    if trk_id not in self.dwell_times_sort:
                        self.dwell_times_sort[trk_id] = {"first_seen": current_time, "last_seen": current_time}
                        self._add_log(f"ENTERED: SORT ID {trk_id} (person)")
                    else:
                        self.dwell_times_sort[trk_id]["last_seen"] = current_time

            # Log exits
            for prev_id in self.prev_sort_persons:
                if prev_id not in active_sort_persons:
                    dwell_sec = current_time - self.dwell_times_sort[prev_id]["first_seen"]
                    self._add_log(f"EXITED:  SORT ID {prev_id} (person) - Dwell: {dwell_sec:.1f}s")
            self.prev_sort_persons = active_sort_persons

            # Expose SORT frames and statistics
            with self.lock:
                self.latest_sort_frame = sort_jpg.tobytes()
                self.stats["sort_active"] = self.sort_tracker.active_count
                self.stats["sort_lost"] = self.sort_tracker.lost_count
                self.stats["fps_sort"] = round(fps_sort, 1)
                self.stats["switches_sort"] = self.sort_tracker.id_switches

    def _deepsort_loop(self):
        """
        THREAD 3: Pulls from deepsort_queue, runs Re-ID, updates DeepSORT JPEG feed and retail metrics.
        """
        while self.loop_active:
            try:
                item = self.deepsort_queue.get(timeout=0.1)
                if item is None: # Exit sentinel
                    break
            except queue.Empty:
                continue
                
            frame, detections, is_simulated = item
            
            # Benchmark DeepSORT (running Deep Re-ID MobileNet in parallel core)
            t0 = time.perf_counter()
            ds_tracks = self.deepsort_tracker.update(detections, frame)
            t_ds = time.perf_counter() - t0
            fps_ds = 1.0 / max(1e-5, t_ds)
            
            # Draw cyan overlays
            self._draw_overlay(frame, ds_tracks, theme_color=(255, 207, 0))
            ds_resized = cv2.resize(frame, (384, 216))
            _, ds_jpg = cv2.imencode('.jpg', ds_resized)
            
            # Update Dwell Times & Log entries
            current_time = time.time()
            active_ds_persons = set()
            for trk in ds_tracks:
                x1, y1, x2, y2, trk_id, class_name, conf = trk
                if class_name == config.RETAIL_CLASS_FILTER:
                    active_ds_persons.add(trk_id)
                    with self.lock:
                        self.ds_unique_people.add(trk_id)
                    if trk_id not in self.dwell_times_ds:
                        self.dwell_times_ds[trk_id] = {"first_seen": current_time, "last_seen": current_time}
                        self._add_log(f"ENTERED: Customer DS_{trk_id} (person)")
                    else:
                        self.dwell_times_ds[trk_id]["last_seen"] = current_time

            # Log exits
            for prev_id in self.prev_ds_persons:
                if prev_id not in active_ds_persons:
                    dwell_sec = current_time - self.dwell_times_ds[prev_id]["first_seen"]
                    self._add_log(f"EXITED:  Customer DS_{prev_id} (person) - Dwell: {dwell_sec:.1f}s")
            self.prev_ds_persons = active_ds_persons

            # Expose DeepSORT frames and retail statistics
            with self.lock:
                self.latest_ds_frame = ds_jpg.tobytes()
                self.stats["ds_active"] = self.deepsort_tracker.active_count
                self.stats["ds_lost"] = self.deepsort_tracker.lost_count
                self.stats["fps_ds"] = round(fps_ds, 1)
                self.stats["switches_ds"] = self.deepsort_tracker.id_switches
                self.stats["footfall"] = len(self.ds_unique_people)
                
                # Render top 3 dwell times
                dwells_desc = ""
                top_visible = sorted(
                    [(pid, current_time - self.dwell_times_ds[pid]["first_seen"]) for pid in active_ds_persons if pid in self.dwell_times_ds],
                    key=lambda x: x[1], reverse=True
                )[:3]
                if top_visible:
                    for pid, d_time in top_visible:
                        dwells_desc += f"- ID DS_{pid} (person): {d_time:.1f}s<br>"
                else:
                    dwells_desc = "No shoppers currently visible in-frame."
                self.stats["dwell_list"] = dwells_desc
                
                # Build database table list (combines database logs)
                tracked_db_list = []
                for trk in ds_tracks:
                    _, _, _, _, trk_id, class_name, conf = trk
                    tracked_db_list.append([f"DS_{trk_id}", class_name, round(conf, 3), "DeepSORT"])
                # Fallback to SORT tracks inside database if DeepSORT is empty
                for trk in self.sort_tracker.trackers:
                    if trk.time_since_update == 0:
                        bbox = trk.get_state()
                        tracked_db_list.append([f"SORT_{trk.id}", trk.class_name, round(trk.confidence, 3), "SORT"])
                
                try:
                    tracked_db_list = sorted(tracked_db_list, key=lambda x: (x[3], x[0]))
                except:
                    pass
                self.stats["tracked_objects"] = tracked_db_list[:10]

    # -------------------------------------------------------------------------
    # OVERLAY DRAWING (OPENCV)
    # -------------------------------------------------------------------------
    def _draw_overlay(self, frame, tracks, theme_color):
        """
        Draws bounding boxes, text badges, and linear top progress lines on frame.
        """
        for trk in tracks:
            x1, y1, x2, y2, track_id, class_name, conf = trk
            cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
            
            box_color = config.get_id_color_bgr(track_id)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), box_color, 2)
            
            badge_text = f"{class_name.upper()} #{track_id}"
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (cx1, cy1 - th - 8), (cx1 + tw + 10, cy1), box_color, -1)
            cv2.putText(frame, badge_text, (cx1 + 5, cy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Linear confidence progress bar on the top edge of box
            cv2.line(frame, (cx1, cy1), (cx2, cy1), (0, 0, 180), 2)
            width = cx2 - cx1
            cv2.line(frame, (cx1, cy1), (cx1 + int(width * conf), cy1), theme_color, 2)

    def _add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.event_logs.append(f"[{timestamp}] {msg}")

    # -------------------------------------------------------------------------
    # STREAM MULTIPART WRAPPER
    # -------------------------------------------------------------------------
    def _stream_frame(self, tracker_name):
        """
        Generates JPEG streaming frames in standard boundary protocol.
        """
        last_frame = None
        while True:
            frame = None
            with self.lock:
                if tracker_name == "sort":
                    frame = self.latest_sort_frame
                else:
                    frame = self.latest_ds_frame
                    
            if frame is not None and frame != last_frame:
                last_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.02)  # High-frequency 50Hz polling for extreme smoothness

    # -------------------------------------------------------------------------
    # SERVER RUN BOOTSTRAP
    # -------------------------------------------------------------------------
    def run(self):
        """
        Launches the Flask web server on port 5000.
        """
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        print("\n" + "="*80)
        print(" [ONLINE] OBJECT DETECTION AND TRACKING WEB SERVER ACTIVATED")
        print("          EXPOSING PIPELINE LIVE AT: http://localhost:5000")
        print("="*80 + "\n")
        
        self.app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

    # -------------------------------------------------------------------------
    # HTML IN-MEMORY RENDERER
    # -------------------------------------------------------------------------
    def _get_html_page(self):
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OBJECT DETECTION & TRACKING SYSTEM - Live Web Portal</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        body {
            background-color: #0a0a0a;
            color: #ffffff;
            overflow-x: hidden;
        }
        #splash-screen {
            position: absolute;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #0a0a0a;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 100;
        }
        .splash-title {
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 5px;
            text-align: center;
        }
        .splash-subtitle {
            font-size: 13px;
            color: #00cfff;
            margin-bottom: 45px;
            text-align: center;
        }
        .cards-container {
            display: flex;
            gap: 40px;
            margin-bottom: 45px;
        }
        .info-card {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            width: 280px;
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .card-header-sort {
            color: #00ff88;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .card-header-ds {
            color: #00cfff;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .card-desc {
            font-size: 11px;
            color: #ffffff;
            line-height: 1.5;
        }
        .btn-start {
            background-color: #00ff88;
            color: #000000;
            border: none;
            padding: 10px 25px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
            margin-right: 30px;
        }
        .btn-start:hover {
            background-color: #00cfff;
        }
        .btn-quit-outline {
            background-color: transparent;
            color: #ff4444;
            border: 1px solid #ff4444;
            padding: 10px 25px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
        }
        .btn-quit-outline:hover {
            background-color: #ff4444;
            color: #ffffff;
        }
        .splash-credit {
            position: absolute;
            bottom: 20px;
            font-size: 10px;
            color: #888888;
        }
        #dashboard-screen {
            display: none;
            width: 100vw;
            height: 100vh;
            grid-template-rows: 1fr 50px;
        }
        .main-container {
            display: grid;
            grid-template-columns: 30% 30% 40%;
            padding: 10px;
            gap: 15px;
            height: calc(100vh - 50px);
            overflow: hidden;
        }
        .panel {
            background-color: #111111;
            border: 1px solid #2a2a2a;
            display: flex;
            flex-direction: column;
            padding: 15px;
            overflow: hidden;
        }
        .panel-title-sort {
            color: #00ff88;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .panel-title-ds {
            color: #00cfff;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .panel-title-stats {
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .stream-view {
            width: 100%;
            aspect-ratio: 16/9;
            background-color: #050505;
            border: 1px solid #2a2a2a;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .stream-view img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .counter-lbl {
            font-size: 12px;
            margin-bottom: 8px;
        }
        .lbl-green { color: #00ff88; }
        .lbl-cyan { color: #00cfff; }
        .lbl-muted { color: #888888; }
        .stats-scroller {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding-right: 5px;
        }
        .stats-scroller::-webkit-scrollbar {
            width: 4px;
        }
        .stats-scroller::-webkit-scrollbar-thumb {
            background-color: #2a2a2a;
        }
        .stats-card {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            padding: 8px 12px;
        }
        .card-section-title {
            font-size: 11px;
            color: #888888;
            margin-bottom: 6px;
        }
        .stats-line {
            font-size: 12px;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
        }
        .stats-line:last-child {
            margin-bottom: 0;
        }
        .tracked-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin-top: 5px;
        }
        .tracked-table th {
            text-align: left;
            color: #888888;
            padding-bottom: 6px;
            font-weight: normal;
        }
        .tracked-table td {
            padding: 4px 0;
        }
        .td-sort { color: #00ff88; }
        .td-ds { color: #00cfff; }
        .log-terminal {
            background-color: #0c0c0c;
            color: #ffffff;
            font-size: 11px;
            padding: 6px;
            height: 100px;
            overflow-y: auto;
            border: 1px solid #2a2a2a;
            white-space: pre-line;
        }
        .bottom-bar {
            background-color: #111111;
            border-top: 1px solid #2a2a2a;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 15px;
        }
        .controls-group {
            display: flex;
            gap: 15px;
        }
        .btn-control {
            background-color: #111111;
            border: 1px solid #2a2a2a;
            color: #ffffff;
            font-size: 12px;
            padding: 6px 15px;
            cursor: pointer;
        }
        .btn-control:hover {
            background-color: #2a2a2a;
        }
        .btn-pause {
            color: #00ff88;
            border-color: #00ff88;
        }
        .btn-pause:hover {
            background-color: #00ff88;
            color: #000000;
        }
        .btn-reset {
            color: #00cfff;
            border-color: #00cfff;
        }
        .btn-reset:hover {
            background-color: #00cfff;
            color: #000000;
        }
        .btn-return {
            color: #ff4444;
            border-color: #ff4444;
        }
        .btn-return:hover {
            background-color: #ff4444;
            color: #ffffff;
        }
        .status-badge {
            font-size: 12px;
            color: #00ff88;
        }
    </style>
</head>
<body>

    <div id="splash-screen">
        <h1 class="splash-title">OBJECT DETECTION & TRACKING SYSTEM</h1>
        <h3 class="splash-subtitle">Powered by YOLOv8 + SORT + DeepSORT</h3>
        
        <div class="cards-container">
            <div class="info-card">
                <div class="card-header-sort">SORT TRACKER</div>
                <div class="card-desc">Kalman Filter +<br>Intersection over Union (IoU)<br>Real-time Fast Association</div>
            </div>
            <div class="info-card">
                <div class="card-header-ds">DeepSORT TRACKER</div>
                <div class="card-desc">Kalman Filter +<br>Deep Appearance Re-ID<br>Robust Against Occlusions</div>
            </div>
        </div>
        
        <div>
            <button class="btn-start" onclick="startTracking()">START SYSTEM</button>
            <button class="btn-quit-outline" onclick="quitSystem()">QUIT</button>
        </div>
        
        <div class="splash-credit">Enterprise Vision System v1.0 // Antigravity Core</div>
    </div>

    <div id="dashboard-screen">
        <div class="main-container">
            
            <div class="panel">
                <h2 class="panel-title-sort">SORT TRACKER FEED</h2>
                <div class="stream-view" id="sort-stream-container">
                    <div style="color: #888888; font-size:11px;">Awaiting start...</div>
                </div>
                <div>
                    <div class="counter-lbl lbl-green" id="sort-active-lbl">ACTIVE TRACKS: 0 units</div>
                    <div class="counter-lbl lbl-muted" id="sort-lost-lbl">LOST TRACKS:   0 units</div>
                </div>
            </div>
            
            <div class="panel">
                <h2 class="panel-title-ds">DeepSORT TRACKER FEED</h2>
                <div class="stream-view" id="ds-stream-container">
                    <div style="color: #888888; font-size:11px;">Awaiting start...</div>
                </div>
                <div>
                    <div class="counter-lbl lbl-cyan" id="ds-active-lbl">ACTIVE TRACKS: 0 units</div>
                    <div class="counter-lbl lbl-muted" id="ds-lost-lbl">LOST TRACKS:   0 units</div>
                </div>
            </div>
            
            <div class="panel">
                <h2 class="panel-title-stats">REAL-TIME DIAGNOSTICS & ANALYTICS</h2>
                
                <div class="stats-scroller">
                    <div class="stats-card">
                        <div class="stats-line lbl-green" id="fps-sort-lbl">SORT SPEED:     0.0 FPS</div>
                        <div class="stats-line lbl-cyan" id="fps-ds-lbl">DeepSORT SPEED: 0.0 FPS</div>
                        <div class="stats-line" id="dets-lbl" style="color:#ffffff;">YOLOv8 DETECTED: 0 objects</div>
                    </div>
                    
                    <div class="stats-card">
                        <div class="card-section-title">TRACKING IDENTIFICATION FRAGMENTATION</div>
                        <div class="stats-line lbl-green" id="switches-sort-lbl">SORT ID SWITCHES:     0 instances</div>
                        <div class="stats-line lbl-cyan" id="switches-ds-lbl">DeepSORT ID SWITCHES: 0 instances</div>
                    </div>
                    
                    <div class="stats-card">
                        <div class="card-section-title">RETAIL ANALYTICS SUMMARY</div>
                        <div class="stats-line lbl-green" id="footfall-lbl" style="font-weight:bold;">TOTAL UNIQUE CUSTOMERS: 0 persons</div>
                        <div style="font-size:11px; color:#ffffff; margin:6px 0 2px 0;">CUSTOMER DWELL DURATION (TOP 3 VISIBLE):</div>
                        <div style="font-size:10px; color:#888888; padding-left:10px; line-height:1.4;" id="dwell-list">
                            No shoppers currently visible in-frame.
                        </div>
                    </div>
                    
                    <div class="stats-card" style="flex:1; min-height:160px; display:flex; flex-direction:column;">
                        <div class="card-section-title">CURRENT ACTIVE REGISTRATION DATABASE</div>
                        <table class="tracked-table">
                            <thead>
                                <tr>
                                    <th style="width:15%;">ID</th>
                                    <th style="width:25%;">LABEL</th>
                                    <th style="width:45%;">CONFIDENCE BAR</th>
                                    <th style="width:15%;">TRACKER</th>
                                </tr>
                            </thead>
                            <tbody id="table-body">
                                <tr><td colspan="4" style="color:#888888; text-align:center; padding-top:20px;">Awaiting active tracks...</td></tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="stats-card">
                        <div class="card-section-title">SECURITY CONTROL SESSION EVENT LOG</div>
                        <div class="log-terminal" id="log-box">
                            [00:00:00] SYSTEM STANDBY: Awaiting operator start initialization...
                        </div>
                    </div>
                </div>
            </div>
            
        </div>
        
        <div class="bottom-bar">
            <div class="controls-group">
                <button class="btn-control btn-pause" id="btn-pause-lbl" onclick="togglePause()">PAUSE TRACKING</button>
                <button class="btn-control btn-reset" onclick="resetStats()">RESET STATS</button>
                <button class="btn-control btn-return" onclick="returnToMenu()">RETURN TO MENU</button>
            </div>
            <div class="status-badge" id="status-badge-lbl">RUNNING - BOTH TRACKERS ACTIVE</div>
        </div>
    </div>

    <script>
        let statsInterval = null;
        let isPaused = false;
        
        function startTracking() {
            fetch('/api/start')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('splash-screen').style.display = 'none';
                    document.getElementById('dashboard-screen').style.display = 'grid';
                    
                    // Direct stream feeds
                    document.getElementById('sort-stream-container').innerHTML = '<img src="/video_feed/sort" alt="SORT FEED">';
                    document.getElementById('ds-stream-container').innerHTML = '<img src="/video_feed/deepsort" alt="DeepSORT FEED">';
                    
                    isPaused = false;
                    document.getElementById('btn-pause-lbl').innerText = 'PAUSE TRACKING';
                    
                    if (statsInterval) clearInterval(statsInterval);
                    statsInterval = setInterval(fetchStats, 100);
                });
        }
        
        function fetchStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    const stats = data.stats;
                    const logs = data.logs;
                    
                    document.getElementById('sort-active-lbl').innerText = 'ACTIVE TRACKS: ' + stats.sort_active + ' units';
                    document.getElementById('sort-lost-lbl').innerText = 'LOST TRACKS:   ' + stats.sort_lost + ' units';
                    
                    document.getElementById('ds-active-lbl').innerText = 'ACTIVE TRACKS: ' + stats.ds_active + ' units';
                    document.getElementById('ds-lost-lbl').innerText = 'LOST TRACKS:   ' + stats.ds_lost + ' units';
                    
                    document.getElementById('fps_sort-lbl').innerText = 'SORT SPEED:     ' + stats.fps_sort + ' FPS';
                    document.getElementById('fps_ds-lbl').innerText = 'DeepSORT SPEED: ' + stats.fps_ds + ' FPS';
                    document.getElementById('dets-lbl').innerText = 'YOLOv8 DETECTED: ' + stats.dets_count + ' objects';
                    
                    document.getElementById('switches_sort-lbl').innerText = 'SORT ID SWITCHES:     ' + stats.switches_sort + ' instances';
                    document.getElementById('switches_ds-lbl').innerText = 'DeepSORT ID SWITCHES: ' + stats.switches_ds + ' instances';
                    
                    document.getElementById('footfall-lbl').innerText = 'TOTAL UNIQUE CUSTOMERS: ' + stats.footfall + ' persons';
                    document.getElementById('dwell-list').innerHTML = stats.dwell_list;
                    
                    document.getElementById('status-badge-lbl').innerText = stats.status_text;
                    if (stats.status_text.includes("PAUSED")) {
                        document.getElementById('status-badge-lbl').style.color = '#ff4444';
                    } else if (stats.status_text.includes("SIMULATION")) {
                        document.getElementById('status-badge-lbl').style.color = '#00cfff';
                    } else {
                        document.getElementById('status-badge-lbl').style.color = '#00ff88';
                    }
                    
                    const tbody = document.getElementById('table-body');
                    if (stats.tracked_objects.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" style="color:#888888; text-align:center; padding-top:20px;">No active targets registered.</td></tr>';
                    } else {
                        let html = '';
                        stats.tracked_objects.forEach(obj => {
                            const objId = obj[0];
                            const label = obj[1];
                            const conf = obj[2];
                            const tracker = obj[3];
                            
                            const numBlocks = Math.round(conf * 10);
                            const blocks = '█'.repeat(numBlocks) + '░'.repeat(10 - numBlocks);
                            const progressStr = '[' + blocks + '] ' + (conf * 100).toFixed(1) + '%';
                            
                            const cls = tracker === 'SORT' ? 'td-sort' : 'td-ds';
                            
                            html += `<tr>
                                <td class="${cls}">${objId}</td>
                                <td>${label}</td>
                                <td>${progressStr}</td>
                                <td class="${cls}">${tracker}</td>
                            </tr>`;
                        });
                        tbody.innerHTML = html;
                    }
                    
                    const logBox = document.getElementById('log-box');
                    logBox.innerHTML = logs.join('\\n');
                    logBox.scrollTop = logBox.scrollHeight;
                });
        }
        
        function togglePause() {
            fetch('/api/pause', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    isPaused = data.paused;
                    document.getElementById('btn-pause-lbl').innerText = isPaused ? 'RESUME TRACKING' : 'PAUSE TRACKING';
                });
        }
        
        function resetStats() {
            if (confirm("Confirm reset of all analytical records and tracking databases?")) {
                fetch('/api/reset', { method: 'POST' });
            }
        }
        
        function returnToMenu() {
            if (statsInterval) clearInterval(statsInterval);
            document.getElementById('sort-stream-container').innerHTML = '<div style="color: #888888; font-size:11px;">Awaiting start...</div>';
            document.getElementById('ds-stream-container').innerHTML = '<div style="color: #888888; font-size:11px;">Awaiting start...</div>';
            document.getElementById('splash-screen').style.display = 'flex';
            document.getElementById('dashboard-screen').style.display = 'none';
        }
        
        function quitSystem() {
            if (confirm("Confirm complete shutdown and termination of visual server processes?")) {
                fetch('/api/quit', { method: 'POST' })
                    .then(() => {
                        alert("Visual systems server closed cleanly. You may close this tab.");
                        window.close();
                    });
            }
        }
    </script>
</body>
</html>"""
