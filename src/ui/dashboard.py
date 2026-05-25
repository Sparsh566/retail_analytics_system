# dashboard.py
# Antigravity UI styled desktop application for Real-Time Object Tracking & Retail Analytics.
# Implements Splash Screen, Live dashboard with three columns, custom canvases, tables, and controls.

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import cv2
from src import config

class DashboardApp:
    """
    Main Tkinter application manager for the Object Detection & Tracking System.
    """
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        
        # Configure Root Window settings
        self.root.title("OBJECT DETECTION & TRACKING SYSTEM - Enterprise Vision")
        self.root.geometry("1300x750")
        self.root.minsize(1280, 720)
        self.root.configure(bg=config.COLOR_BG)
        
        # UI State variables
        self.current_screen = None
        self.is_paused = False
        
        # Build Screen Frames
        self.splash_frame = tk.Frame(self.root, bg=config.COLOR_BG)
        self.dashboard_frame = tk.Frame(self.root, bg=config.COLOR_BG)
        
        # Initialize UI components
        self._build_splash_screen()
        self._build_dashboard_screen()
        
        # Show Screen 1 initially
        self.show_splash()

    # -------------------------------------------------------------------------
    # ROUTING & SCREEN CONTROL
    # -------------------------------------------------------------------------
    def show_splash(self):
        """
        Switches context to the Splash Screen (no camera capture).
        """
        if self.current_screen:
            self.current_screen.pack_forget()
        self.splash_frame.pack(fill="both", expand=True)
        self.current_screen = self.splash_frame

    def show_dashboard(self):
        """
        Switches context to the Live Dashboard (initiates camera/processing).
        """
        if self.current_screen:
            self.current_screen.pack_forget()
        self.dashboard_frame.pack(fill="both", expand=True)
        self.current_screen = self.dashboard_frame
        
        # Notify controller to start video processing
        self.controller.start_tracking()

    # -------------------------------------------------------------------------
    # SCREEN 1: SPLASH SCREEN BUILDER
    # -------------------------------------------------------------------------
    def _build_splash_screen(self):
        """
        Creates a premium dark splash screen layout with centered typography and cards.
        """
        container = tk.Frame(self.splash_frame, bg=config.COLOR_BG)
        container.place(relx=0.5, rely=0.45, anchor="center")
        
        # 1. Main Monospace Titles
        title_lbl = tk.Label(
            container, text="OBJECT DETECTION & TRACKING SYSTEM",
            font=(config.FONT_FAMILY, 24, "bold"), fg=config.COLOR_GREEN, bg=config.COLOR_BG
        )
        title_lbl.pack(pady=(0, 5))
        
        subtitle_lbl = tk.Label(
            container, text="Powered by YOLOv8 + SORT + DeepSORT",
            font=(config.FONT_FAMILY, 13, "normal"), fg=config.COLOR_CYAN, bg=config.COLOR_BG
        )
        subtitle_lbl.pack(pady=(0, 45))
        
        # 2. Side-by-Side Tracker Cards
        cards_frame = tk.Frame(container, bg=config.COLOR_BG)
        cards_frame.pack(pady=(0, 45))
        
        # SORT Card
        sort_card = tk.Frame(
            cards_frame, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1, width=280, height=140
        )
        sort_card.pack_propagate(False)
        sort_card.pack(side="left", padx=20)
        
        sort_title = tk.Label(
            sort_card, text="SORT TRACKER",
            font=config.FONT_HEADER, fg=config.COLOR_GREEN, bg=config.COLOR_CARD
        )
        sort_title.pack(pady=(20, 10))
        
        sort_desc = tk.Label(
            sort_card, text="Kalman Filter +\nIntersection over Union (IoU)\nReal-time Fast Association",
            font=config.FONT_SUBTITLE, fg=config.COLOR_TEXT, bg=config.COLOR_CARD, justify="center"
        )
        sort_desc.pack()
        
        # DeepSORT Card
        deepsort_card = tk.Frame(
            cards_frame, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1, width=280, height=140
        )
        deepsort_card.pack_propagate(False)
        deepsort_card.pack(side="left", padx=20)
        
        ds_title = tk.Label(
            deepsort_card, text="DeepSORT TRACKER",
            font=config.FONT_HEADER, fg=config.COLOR_CYAN, bg=config.COLOR_CARD
        )
        ds_title.pack(pady=(20, 10))
        
        ds_desc = tk.Label(
            deepsort_card, text="Kalman Filter +\nDeep Appearance Re-ID\nRobust Against Occlusions",
            font=config.FONT_SUBTITLE, fg=config.COLOR_TEXT, bg=config.COLOR_CARD, justify="center"
        )
        ds_desc.pack()
        
        # 3. Action Buttons (Flat UI, Green fill, Red outline)
        btns_frame = tk.Frame(container, bg=config.COLOR_BG)
        btns_frame.pack()
        
        start_btn = tk.Button(
            btns_frame, text="START SYSTEM", font=(config.FONT_FAMILY, 11, "bold"),
            bg=config.COLOR_GREEN, fg="#000000", bd=0, padx=25, pady=8,
            activebackground=config.COLOR_CYAN, activeforeground="#000000",
            command=self.show_dashboard, cursor="hand2"
        )
        start_btn.pack(side="left", padx=15)
        
        quit_btn = tk.Button(
            btns_frame, text="QUIT", font=(config.FONT_FAMILY, 11, "bold"),
            bg=config.COLOR_BG, fg=config.COLOR_WARNING, bd=1,
            highlightbackground=config.COLOR_WARNING, highlightthickness=1,
            activebackground=config.COLOR_WARNING, activeforeground="#ffffff",
            padx=25, pady=7, command=self.root.quit, cursor="hand2"
        )
        quit_btn.pack(side="left", padx=15)
        
        # 4. Bottom Credit Label
        credit_lbl = tk.Label(
            self.splash_frame, text="Enterprise Vision System v1.0 // Antigravity Core",
            font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_BG
        )
        credit_lbl.pack(side="bottom", pady=20)

    # -------------------------------------------------------------------------
    # SCREEN 2: LIVE DASHBOARD BUILDER
    # -------------------------------------------------------------------------
    def _build_dashboard_screen(self):
        """
        Constructs the three-column layout: Left (SORT), Middle (DeepSORT), Right (Stats/Retail).
        """
        # Outer border frame container
        self.dashboard_frame.grid_columnconfigure(0, weight=30)  # Left Col (30%)
        self.dashboard_frame.grid_columnconfigure(1, weight=30)  # Middle Col (30%)
        self.dashboard_frame.grid_columnconfigure(2, weight=40)  # Right Col (40%)
        self.dashboard_frame.grid_rowconfigure(0, weight=1)      # Main Content Area
        
        # ---------------------------------------------------------------------
        # LEFT COLUMN: SORT TRACKER FEED
        # ---------------------------------------------------------------------
        col1_frame = tk.Frame(
            self.dashboard_frame, bg=config.COLOR_PANEL,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1
        )
        col1_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 60))
        
        sort_header = tk.Label(
            col1_frame, text="SORT TRACKER FEED", font=config.FONT_HEADER,
            fg=config.COLOR_GREEN, bg=config.COLOR_PANEL, pady=10
        )
        sort_header.pack()
        
        # Canvas for embedding ImageTk cv2 frame (Standardized 384x216 ratio)
        self.sort_canvas = tk.Canvas(
            col1_frame, width=384, height=216, bg="#050505",
            highlightthickness=1, highlightbackground=config.COLOR_BORDER
        )
        self.sort_canvas.pack(pady=15, padx=15, fill="both", expand=True)
        
        # Track Counters
        sort_counters = tk.Frame(col1_frame, bg=config.COLOR_PANEL)
        sort_counters.pack(fill="x", padx=15, pady=5)
        
        self.sort_active_lbl = tk.Label(
            sort_counters, text="ACTIVE TRACKS: 0 units", font=config.FONT_BODY,
            fg=config.COLOR_GREEN, bg=config.COLOR_PANEL, anchor="w"
        )
        self.sort_active_lbl.pack(fill="x", pady=2)
        
        self.sort_lost_lbl = tk.Label(
            sort_counters, text="LOST TRACKS:   0 units", font=config.FONT_BODY,
            fg=config.COLOR_MUTED, bg=config.COLOR_PANEL, anchor="w"
        )
        self.sort_lost_lbl.pack(fill="x", pady=2)

        # ---------------------------------------------------------------------
        # MIDDLE COLUMN: DEEPSORT TRACKER FEED
        # ---------------------------------------------------------------------
        col2_frame = tk.Frame(
            self.dashboard_frame, bg=config.COLOR_PANEL,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1
        )
        col2_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=(10, 60))
        
        ds_header = tk.Label(
            col2_frame, text="DeepSORT TRACKER FEED", font=config.FONT_HEADER,
            fg=config.COLOR_CYAN, bg=config.COLOR_PANEL, pady=10
        )
        ds_header.pack()
        
        self.deepsort_canvas = tk.Canvas(
            col2_frame, width=384, height=216, bg="#050505",
            highlightthickness=1, highlightbackground=config.COLOR_BORDER
        )
        self.deepsort_canvas.pack(pady=15, padx=15, fill="both", expand=True)
        
        # Track Counters
        ds_counters = tk.Frame(col2_frame, bg=config.COLOR_PANEL)
        ds_counters.pack(fill="x", padx=15, pady=5)
        
        self.ds_active_lbl = tk.Label(
            ds_counters, text="ACTIVE TRACKS: 0 units", font=config.FONT_BODY,
            fg=config.COLOR_CYAN, bg=config.COLOR_PANEL, anchor="w"
        )
        self.ds_active_lbl.pack(fill="x", pady=2)
        
        self.ds_lost_lbl = tk.Label(
            ds_counters, text="LOST TRACKS:   0 units", font=config.FONT_BODY,
            fg=config.COLOR_MUTED, bg=config.COLOR_PANEL, anchor="w"
        )
        self.ds_lost_lbl.pack(fill="x", pady=2)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: SYSTEM STATISTICS & RETAIL PANEL
        # ---------------------------------------------------------------------
        col3_frame = tk.Frame(
            self.dashboard_frame, bg=config.COLOR_PANEL,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1
        )
        col3_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=(10, 60))
        
        stats_header = tk.Label(
            col3_frame, text="REAL-TIME DIAGNOSTICS & ANALYTICS", font=config.FONT_HEADER,
            fg=config.COLOR_TEXT, bg=config.COLOR_PANEL, pady=10
        )
        stats_header.pack()
        
        # Scrollable inner wrapper for stats
        stats_scroll = tk.Frame(col3_frame, bg=config.COLOR_PANEL)
        stats_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 1. Dual FPS Block & Object Detections
        fps_block = tk.Frame(
            stats_scroll, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightthickness=1
        )
        fps_block.pack(fill="x", pady=5)
        
        self.fps_sort_lbl = tk.Label(
            fps_block, text="SORT SPEED:     0.0 FPS", font=config.FONT_BODY,
            fg=config.COLOR_GREEN, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        self.fps_sort_lbl.pack(fill="x")
        
        self.fps_ds_lbl = tk.Label(
            fps_block, text="DeepSORT SPEED: 0.0 FPS", font=config.FONT_BODY,
            fg=config.COLOR_CYAN, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        self.fps_ds_lbl.pack(fill="x")
        
        self.dets_lbl = tk.Label(
            fps_block, text="YOLOv8 DETECTED: 0 objects", font=config.FONT_BODY,
            fg=config.COLOR_TEXT, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        self.dets_lbl.pack(fill="x")
        
        # 2. Tracking ID Switches Stability Card
        stability_block = tk.Frame(
            stats_scroll, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightthickness=1
        )
        stability_block.pack(fill="x", pady=5)
        
        stability_header = tk.Label(
            stability_block, text="TRACKING IDENTIFICATION FRAGMENTATION", font=config.FONT_SUBTITLE,
            fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        stability_header.pack(fill="x")
        
        self.switches_sort_lbl = tk.Label(
            stability_block, text="SORT ID SWITCHES:     0 instances", font=config.FONT_BODY,
            fg=config.COLOR_GREEN, bg=config.COLOR_CARD, anchor="w", padx=10, pady=2
        )
        self.switches_sort_lbl.pack(fill="x")
        
        self.switches_ds_lbl = tk.Label(
            stability_block, text="DeepSORT ID SWITCHES: 0 instances", font=config.FONT_BODY,
            fg=config.COLOR_CYAN, bg=config.COLOR_CARD, anchor="w", padx=10, pady=2
        )
        self.switches_ds_lbl.pack(fill="x")
        
        # 3. Retail Use-Case Card (Unique footfall and dwell times)
        retail_block = tk.Frame(
            stats_scroll, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightthickness=1
        )
        retail_block.pack(fill="x", pady=5)
        
        retail_header = tk.Label(
            retail_block, text="RETAIL ANALYTICS SUMMARY", font=config.FONT_SUBTITLE,
            fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        retail_header.pack(fill="x")
        
        # Footfall Counts
        self.footfall_lbl = tk.Label(
            retail_block, text="TOTAL UNIQUE CUSTOMERS: 0 persons", font=config.FONT_BODY,
            fg=config.COLOR_GREEN, bg=config.COLOR_CARD, anchor="w", padx=10, pady=2
        )
        self.footfall_lbl.pack(fill="x")
        
        # Dwell Times Scroll list (Displays seconds each person is active)
        self.dwell_time_lbl = tk.Label(
            retail_block, text="CUSTOMER DWELL DURATION (TOP 3 VISIBLE):", font=config.FONT_BODY,
            fg=config.COLOR_TEXT, bg=config.COLOR_CARD, anchor="w", padx=10, pady=(6, 2)
        )
        self.dwell_time_lbl.pack(fill="x")
        
        self.dwell_list_lbl = tk.Label(
            retail_block, text="- ID 1 (person): 0.0s\n- ID 2 (person): 0.0s",
            font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_CARD,
            anchor="w", justify="left", padx=15, pady=4
        )
        self.dwell_list_lbl.pack(fill="x")

        # 4. Live Tracked Objects Table Panel
        table_frame = tk.Frame(
            stats_scroll, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightthickness=1
        )
        table_frame.pack(fill="both", expand=True, pady=5)
        
        table_header = tk.Label(
            table_frame, text="CURRENT ACTIVE REGISTRATION DATABASE", font=config.FONT_SUBTITLE,
            fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        table_header.pack(fill="x")
        
        # Table Grid Headers
        grid_headers = tk.Frame(table_frame, bg=config.COLOR_CARD)
        grid_headers.pack(fill="x", padx=10)
        grid_headers.columnconfigure(0, weight=2) # ID
        grid_headers.columnconfigure(1, weight=3) # Label
        grid_headers.columnconfigure(2, weight=5) # Confidence with bar
        grid_headers.columnconfigure(3, weight=2) # Tracker
        
        tk.Label(grid_headers, text="ID", font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w").grid(row=0, column=0, sticky="ew")
        tk.Label(grid_headers, text="LABEL", font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w").grid(row=0, column=1, sticky="ew")
        tk.Label(grid_headers, text="CONFIDENCE BAR", font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w").grid(row=0, column=2, sticky="ew")
        tk.Label(grid_headers, text="TRACKER", font=config.FONT_SMALL, fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w").grid(row=0, column=3, sticky="ew")
        
        # Sub-container for dynamic grid rows (holds maximum 10 label rows)
        self.rows_container = tk.Frame(table_frame, bg=config.COLOR_CARD)
        self.rows_container.pack(fill="both", expand=True, padx=10, pady=2)
        self.table_row_widgets = []
        self._init_empty_table_rows()

        # 5. Session scrolling security events log panel
        log_frame = tk.Frame(
            stats_scroll, bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER, highlightthickness=1
        )
        log_frame.pack(fill="both", expand=True, pady=5)
        
        log_header = tk.Label(
            log_frame, text="SECURITY CONTROL SESSION EVENT LOG", font=config.FONT_SUBTITLE,
            fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w", padx=10, pady=4
        )
        log_header.pack(fill="x")
        
        self.log_text = tk.Text(
            log_frame, bg=config.COLOR_PANEL, fg=config.COLOR_TEXT, bd=0,
            font=config.FONT_SMALL, selectbackground=config.COLOR_BORDER,
            insertbackground=config.COLOR_TEXT, wrap="none", height=5
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.config(state="disabled")

        # ---------------------------------------------------------------------
        # BOTTOM BAR (FULL WIDTH)
        # ---------------------------------------------------------------------
        bottom_bar = tk.Frame(
            self.root, bg=config.COLOR_PANEL,
            highlightbackground=config.COLOR_BORDER, highlightcolor=config.COLOR_BORDER,
            highlightthickness=1, height=50
        )
        bottom_bar.pack_propagate(False)
        bottom_bar.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
        
        # 1. Action Controls
        self.pause_btn = tk.Button(
            bottom_bar, text="PAUSE TRACKING", font=config.FONT_BODY,
            bg=config.COLOR_PANEL, fg=config.COLOR_GREEN, bd=1,
            highlightbackground=config.COLOR_GREEN, highlightthickness=1,
            activebackground=config.COLOR_GREEN, activeforeground="#000000",
            padx=15, command=self._toggle_pause, cursor="hand2"
        )
        self.pause_btn.pack(side="left", padx=10, pady=8)
        
        reset_btn = tk.Button(
            bottom_bar, text="RESET STATS", font=config.FONT_BODY,
            bg=config.COLOR_PANEL, fg=config.COLOR_CYAN, bd=1,
            highlightbackground=config.COLOR_CYAN, highlightthickness=1,
            activebackground=config.COLOR_CYAN, activeforeground="#000000",
            padx=15, command=self._reset_dashboard_stats, cursor="hand2"
        )
        reset_btn.pack(side="left", padx=5, pady=8)
        
        exit_btn = tk.Button(
            bottom_bar, text="RETURN TO MENU", font=config.FONT_BODY,
            bg=config.COLOR_PANEL, fg=config.COLOR_WARNING, bd=1,
            highlightbackground=config.COLOR_WARNING, highlightthickness=1,
            activebackground=config.COLOR_WARNING, activeforeground="#ffffff",
            padx=15, command=self._return_to_splash, cursor="hand2"
        )
        exit_btn.pack(side="left", padx=5, pady=8)
        
        # 2. Status Label (Right aligned)
        self.status_lbl = tk.Label(
            bottom_bar, text="RUNNING - BOTH TRACKERS ACTIVE", font=config.FONT_BODY,
            fg=config.COLOR_GREEN, bg=config.COLOR_PANEL, padx=20
        )
        self.status_lbl.pack(side="right", fill="y")

    # -------------------------------------------------------------------------
    # DYNAMIC RENDER INTERFACES & HELPERS
    # -------------------------------------------------------------------------
    def _init_empty_table_rows(self):
        """
        Creates 10 pre-allocated blank rows in the grid to maintain flat layout stability.
        """
        for i in range(10):
            row_widgets = []
            self.rows_container.grid_columnconfigure(0, weight=2)
            self.rows_container.grid_columnconfigure(1, weight=3)
            self.rows_container.grid_columnconfigure(2, weight=5)
            self.rows_container.grid_columnconfigure(3, weight=2)
            
            # Row index labeling
            for col in range(4):
                lbl = tk.Label(
                    self.rows_container, text="", font=config.FONT_SMALL,
                    fg=config.COLOR_MUTED, bg=config.COLOR_CARD, anchor="w"
                )
                lbl.grid(row=i, column=col, sticky="ew", pady=1)
                row_widgets.append(lbl)
                
            self.table_row_widgets.append(row_widgets)

    def update_table(self, tracked_objects_list):
        """
        Updates the monospace database table with the latest tracked objects.
        tracked_objects_list: list of (id, class_name, confidence, tracker_name) tuples.
        """
        # Sort by active database ID numerically if possible
        try:
            tracked_objects_list = sorted(
                tracked_objects_list, 
                key=lambda x: (x[3], int(''.join(c for c in str(x[0]) if c.isdigit()) or 0))
            )
        except Exception:
            pass

        # Fill available grid rows (maximum 10 rows displayed)
        for i in range(10):
            row_lbls = self.table_row_widgets[i]
            if i < len(tracked_objects_list):
                obj_id, label, conf, tracker = tracked_objects_list[i]
                
                # Confidence progress bar generation: [██████░░░░] 60%
                num_blocks = int(conf * 10)
                blocks = "█" * num_blocks + "░" * (10 - num_blocks)
                progress_str = f"[{blocks}] {conf * 100:.1f}%"
                
                # Colorize cell based on tracker
                color = config.COLOR_GREEN if tracker == "SORT" else config.COLOR_CYAN
                
                row_lbls[0].config(text=str(obj_id), fg=color)
                row_lbls[1].config(text=str(label), fg=config.COLOR_TEXT)
                row_lbls[2].config(text=progress_str, fg=config.COLOR_TEXT)
                row_lbls[3].config(text=str(tracker), fg=color)
            else:
                # Reset details for empty rows
                for lbl in row_lbls:
                    lbl.config(text="", fg=config.COLOR_MUTED)

    def write_event_log(self, log_msg):
        """
        Appends a security action event log line to the scrolling monospace terminal block.
        """
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {log_msg}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, formatted)
        
        # Scroll to bottom
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def render_canvas_frames(self, sort_img_tk, ds_img_tk):
        """
        Pushes fresh ImageTk photo frames into Sort and DeepSORT views.
        """
        # Anchor top-left reference so GC doesn't delete image buffer
        self.sort_canvas.image = sort_img_tk
        self.sort_canvas.create_image(0, 0, image=sort_img_tk, anchor="nw")
        
        self.deepsort_canvas.image = ds_img_tk
        self.deepsort_canvas.create_image(0, 0, image=ds_img_tk, anchor="nw")

    # -------------------------------------------------------------------------
    # STATE MUTATION TRIGGER FUNCTIONS
    # -------------------------------------------------------------------------
    def _toggle_pause(self):
        """
        Toggles live camera frame capture.
        """
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="RESUME TRACKING", fg=config.COLOR_GREEN)
            self.status_lbl.config(text="PAUSED - SENSOR GRABBING DEACTIVATED", fg=config.COLOR_WARNING)
            self.write_event_log("SYSTEM WARNING: Live pipeline paused by operator.")
        else:
            self.pause_btn.config(text="PAUSE TRACKING", fg=config.COLOR_GREEN)
            self.status_lbl.config(text="RUNNING - BOTH TRACKERS ACTIVE", fg=config.COLOR_GREEN)
            self.write_event_log("SYSTEM INFO: Live pipeline tracking resumed.")

    def _reset_dashboard_stats(self):
        """
        Instructs the controller to wipe and reset tracking logs.
        """
        self.controller.reset_system_stats()
        
        # Clear log text box
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        self.write_event_log("SYSTEM ACTION: Analytical records and trackers reset completed.")

    def _return_to_splash(self):
        """
        Halts cameras and returns to introductory Splash screen.
        """
        # Trigger controller to clean thread handles
        self.controller.stop_tracking()
        
        # Reset pauses
        if self.is_paused:
            self._toggle_pause()
            
        self.show_splash()
