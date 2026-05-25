# main.py
# Main entry point and orchestration controller for the retail analytics tracking system.
# Coordinates frames, trackers, diagnostics, retail analytics, and drawing loops.

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import sys
from src import config
from src.vision.detector import Detector
from src.trackers.sort import SortTracker
from src.trackers.deepsort import DeepSortTracker
from src.ui.dashboard import DashboardApp

class MainController:
    """
    Orchestrates frame grabbing, tracking logic, stats calculation, drawing, and UI updates.
    """
    def __init__(self):
        # 1. Boot Tkinter Frame
        self.root = tk.Tk()
        self.dashboard = DashboardApp(self.root, self)
        
        # 2. State metrics and trackers
        self.detector = None
        self.sort_tracker = None
        self.deepsort_tracker = None
        
        # Retail Analytics State variables
        self.sort_unique_people = set()
        self.ds_unique_people = set()
        
        # Dictionary storing {track_id: {"first_seen": timestamp, "last_seen": timestamp}}
        self.dwell_times_sort = {}
        self.dwell_times_ds = {}
        
        # Set of active person IDs from previous frame to detect enter/exit transitions
        self.prev_sort_persons = set()
        self.prev_ds_persons = set()
        
        # Thread/Timer schedule handle
        self.after_id = None
        
        # Window closing hook
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def run(self):
        """
        Starts the desktop window main event loop.
        """
        self.root.mainloop()

    # -------------------------------------------------------------------------
    # PIPELINE LIFECYCLE MANAGEMENT
    # -------------------------------------------------------------------------
    def start_tracking(self):
        """
        Initializes core detector and trackers, clears logs, and starts the processing loop.
        """
        self.dashboard.write_event_log("SYSTEM ACTION: Activating sensors and loading models...")
        
        # Initialize detector (loads YOLOv8 or fallback simulator)
        self.detector = Detector()
        
        # Initialize trackers
        self.sort_tracker = SortTracker(max_age=15, min_hits=3, iou_threshold=0.25)
        self.deepsort_tracker = DeepSortTracker(max_age=15, n_init=3)
        
        # Reset metric registers
        self.reset_system_stats()
        
        # Set mode status label
        mode_text = "SIMULATION ENGINE ACTIVE" if self.detector.simulation_mode else "LIVE CAMERA ACTIVE"
        self.dashboard.status_lbl.config(
            text=f"RUNNING - BOTH TRACKERS ACTIVE ({mode_text})", 
            fg=config.COLOR_CYAN if self.detector.simulation_mode else config.COLOR_GREEN
        )
        self.dashboard.write_event_log(f"SYSTEM INFO: Tracking pipeline running under {mode_text}.")

        # Trigger processing loop
        self.update_loop()

    def stop_tracking(self):
        """
        Cancels pending scheduled frames and releases system handles cleanly.
        """
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
            
        if self.detector:
            self.detector.close()
            self.detector = None
            
        self.dashboard.write_event_log("SYSTEM WARNING: Sensors and tracking pipelines deactivated.")

    def reset_system_stats(self):
        """
        Resets all tracking, retail database registers, dwell times, and logs.
        """
        self.sort_unique_people.clear()
        self.ds_unique_people.clear()
        self.dwell_times_sort.clear()
        self.dwell_times_ds.clear()
        self.prev_sort_persons.clear()
        self.prev_ds_persons.clear()
        
        if self.sort_tracker:
            self.sort_tracker.id_switches = 0
            self.sort_tracker.trackers.clear()
            self.sort_tracker.dead_track_history.clear()
            
        if self.deepsort_tracker:
            self.deepsort_tracker.id_switches = 0
            self.deepsort_tracker.dead_track_history.clear()
            if not self.deepsort_tracker.is_emulated:
                self.deepsort_tracker.tracker.tracks.clear()
            else:
                self.deepsort_tracker.emulated_tracker.id_switches = 0
                self.deepsort_tracker.emulated_tracker.trackers.clear()
                self.deepsort_tracker.emulated_tracker.dead_track_history.clear()

    # -------------------------------------------------------------------------
    # MAIN REAL-TIME LOOP
    # -------------------------------------------------------------------------
    def update_loop(self):
        """
        Main tick processing: grabs frames, runs models, computes metrics, draws, and posts to canvas.
        """
        # If UI is paused, wait and tick slowly
        if self.dashboard.is_paused:
            self.after_id = self.root.after(100, self.update_loop)
            return

        # 1. Grab camera frame
        start_grab = time.time()
        frame, detections, is_simulated = self.detector.get_frame()
        
        # 2. Copy frames for separate rendering
        frame_sort = frame.copy()
        frame_ds = frame.copy()

        # 3. Benchmark SORT Inference
        t0 = time.perf_counter()
        sort_tracks = self.sort_tracker.update(detections)
        t_sort = time.perf_counter() - t0
        fps_sort = 1.0 / max(1e-5, t_sort)

        # 4. Benchmark DeepSORT Inference
        t0 = time.perf_counter()
        ds_tracks = self.deepsort_tracker.update(detections, frame)
        t_ds = time.perf_counter() - t0
        fps_ds = 1.0 / max(1e-5, t_ds)

        # 5. Process Retail Analytics (Entered/Exited, Footfall, Dwell Times)
        current_time = time.time()
        active_sort_persons = set()
        active_ds_persons = set()

        # Update SORT Dwell metrics
        for trk in sort_tracks:
            x1, y1, x2, y2, trk_id, class_name, conf = trk
            if class_name == config.RETAIL_CLASS_FILTER:
                active_sort_persons.add(trk_id)
                self.sort_unique_people.add(trk_id)
                if trk_id not in self.dwell_times_sort:
                    self.dwell_times_sort[trk_id] = {"first_seen": current_time, "last_seen": current_time}
                    self.dashboard.write_event_log(f"ENTERED: SORT ID {trk_id} (person)")
                else:
                    self.dwell_times_sort[trk_id]["last_seen"] = current_time

        # Update DeepSORT Dwell metrics (used as primary dashboard display)
        for trk in ds_tracks:
            x1, y1, x2, y2, trk_id, class_name, conf = trk
            if class_name == config.RETAIL_CLASS_FILTER:
                active_ds_persons.add(trk_id)
                self.ds_unique_people.add(trk_id)
                if trk_id not in self.dwell_times_ds:
                    self.dwell_times_ds[trk_id] = {"first_seen": current_time, "last_seen": current_time}
                    self.dashboard.write_event_log(f"ENTERED: Customer DS_{trk_id} (person)")
                else:
                    self.dwell_times_ds[trk_id]["last_seen"] = current_time

        # Detect SORT exits
        for prev_id in self.prev_sort_persons:
            if prev_id not in active_sort_persons:
                dwell_sec = current_time - self.dwell_times_sort[prev_id]["first_seen"]
                self.dashboard.write_event_log(f"EXITED:  SORT ID {prev_id} (person) - Dwell: {dwell_sec:.1f}s")
        
        # Detect DeepSORT exits
        for prev_id in self.prev_ds_persons:
            if prev_id not in active_ds_persons:
                dwell_sec = current_time - self.dwell_times_ds[prev_id]["first_seen"]
                self.dashboard.write_event_log(f"EXITED:  Customer DS_{prev_id} (person) - Dwell: {dwell_sec:.1f}s")

        self.prev_sort_persons = active_sort_persons
        self.prev_ds_persons = active_ds_persons

        # 6. Drawing Bounding Boxes & Badges onto OpenCV frame copies
        self._draw_overlay(frame_sort, sort_tracks, theme_color=(0, 255, 136)) # Green BGR for SORT
        self._draw_overlay(frame_ds, ds_tracks, theme_color=(255, 207, 0))    # Cyan BGR for DeepSORT

        # 7. Convert OpenCV frames to ImageTk standard canvas format (384x216)
        sort_resized = cv2.resize(frame_sort, (384, 216))
        ds_resized = cv2.resize(frame_ds, (384, 216))
        
        sort_rgb = cv2.cvtColor(sort_resized, cv2.COLOR_BGR2RGB)
        ds_rgb = cv2.cvtColor(ds_resized, cv2.COLOR_BGR2RGB)
        
        sort_img_pil = Image.fromarray(sort_rgb)
        ds_img_pil = Image.fromarray(ds_rgb)
        
        sort_img_tk = ImageTk.PhotoImage(image=sort_img_pil)
        ds_img_tk = ImageTk.PhotoImage(image=ds_img_pil)
        
        self.dashboard.render_canvas_frames(sort_img_tk, ds_img_tk)

        # 8. Push metrics and diagnostics updates to UI Panels
        self.dashboard.sort_active_lbl.config(text=f"ACTIVE TRACKS: {self.sort_tracker.active_count} units")
        self.dashboard.sort_lost_lbl.config(text=f"LOST TRACKS:   {self.sort_tracker.lost_count} units")
        
        self.dashboard.ds_active_lbl.config(text=f"ACTIVE TRACKS: {self.deepsort_tracker.active_count} units")
        self.dashboard.ds_lost_lbl.config(text=f"LOST TRACKS:   {self.deepsort_tracker.lost_count} units")
        
        self.dashboard.fps_sort_lbl.config(text=f"SORT SPEED:     {fps_sort:5.1f} FPS (Inf: {t_sort*1000.0:.1f}ms)")
        self.dashboard.fps_ds_lbl.config(text=f"DeepSORT SPEED: {fps_ds:5.1f} FPS (Inf: {t_ds*1000.0:.1f}ms)")
        self.dashboard.dets_lbl.config(text=f"YOLOv8 DETECTED: {len(detections)} objects")
        
        self.dashboard.switches_sort_lbl.config(text=f"SORT ID SWITCHES:     {self.sort_tracker.id_switches} instances")
        self.dashboard.switches_ds_lbl.config(text=f"DeepSORT ID SWITCHES: {self.deepsort_tracker.id_switches} instances")
        
        # Retail summary (Uses DeepSORT counts)
        self.dashboard.footfall_lbl.config(text=f"TOTAL UNIQUE CUSTOMERS: {len(self.ds_unique_people)} persons")
        
        # Render Top 3 currently visible customer dwell times
        dwells_desc = ""
        top_visible = sorted(
            [(pid, current_time - self.dwell_times_ds[pid]["first_seen"]) for pid in active_ds_persons if pid in self.dwell_times_ds],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        if top_visible:
            for pid, d_time in top_visible:
                dwells_desc += f"- ID DS_{pid} (person): {d_time:5.1f}s\n"
        else:
            dwells_desc = "No shoppers currently visible in-frame.\n"
        self.dashboard.dwell_list_lbl.config(text=dwells_desc.strip())

        # 9. Update Database Table Panel (Show last 10 active tracks)
        tracked_db_list = []
        for trk in sort_tracks:
            _, _, _, _, trk_id, class_name, conf = trk
            tracked_db_list.append((f"SORT_{trk_id}", class_name, conf, "SORT"))
        for trk in ds_tracks:
            _, _, _, _, trk_id, class_name, conf = trk
            tracked_db_list.append((f"DS_{trk_id}", class_name, conf, "DeepSORT"))
            
        self.dashboard.update_table(tracked_db_list[:10])

        # Schedule next tick (around 30 FPS target)
        self.after_id = self.root.after(20, self.update_loop)

    # -------------------------------------------------------------------------
    # DRAW OVERLAY UTILITIES (OPENCV)
    # -------------------------------------------------------------------------
    def _draw_overlay(self, frame, tracks, theme_color):
        """
        Draws dynamic retail style bounding boxes, text badges, and physical 
        top-edge confidence progress lines on the raw frame.
        """
        for trk in tracks:
            x1, y1, x2, y2, track_id, class_name, conf = trk
            cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
            
            # Bright saturated colors generated deterministically from ID
            box_color = config.get_id_color_bgr(track_id)
            
            # A) Draw bounding box
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), box_color, 2)
            
            # B) Monospace Badge on top-left of box
            badge_text = f"{class_name.upper()} #{track_id}"
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            
            # Solid color background badge matching ID color
            cv2.rectangle(frame, (cx1, cy1 - th - 8), (cx1 + tw + 10, cy1), box_color, -1)
            
            # Black high-contrast text overlay in badge
            cv2.putText(
                frame, badge_text, (cx1 + 5, cy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA
            )
            
            # C) Thin top-edge confidence progress bar directly on the frame!
            # Red background bar
            cv2.line(frame, (cx1, cy1), (cx2, cy1), (0, 0, 180), 2)
            # Green foreground bar (scaling width based on confidence)
            width = cx2 - cx1
            cv2.line(frame, (cx1, cy1), (cx1 + int(width * conf), cy1), theme_color, 2)

    # -------------------------------------------------------------------------
    # CORE CLEAN TERMINATION HOOK
    # -------------------------------------------------------------------------
    def on_close(self):
        """
        Proper release of camera handles and closing Tkinter session.
        """
        self.stop_tracking()
        self.root.destroy()
        sys.exit(0)

# -------------------------------------------------------------------------
# DIRECT SCRIPT RUN BOOTSTRAP
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app = MainController()
    app.run()
