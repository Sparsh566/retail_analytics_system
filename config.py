# config.py
# Central configuration and design tokens for the retail analytics vision system.

import colorsys
import hashlib

# -------------------------------------------------------------------------
# DESIGN SYSTEM COLOR PALETTE (Strict Monospace Theme)
# -------------------------------------------------------------------------
COLOR_BG = "#0a0a0a"       # Darkest background
COLOR_PANEL = "#111111"    # Dark grey panel background
COLOR_CARD = "#1a1a1a"     # Mid grey card background
COLOR_BORDER = "#2a2a2a"   # Accent border lines (1px)
COLOR_GREEN = "#00ff88"    # Accent green (SORT / Primary Action)
COLOR_CYAN = "#00cfff"     # Accent cyan (DeepSORT)
COLOR_TEXT = "#ffffff"     # White primary text
COLOR_MUTED = "#888888"    # Muted description text
COLOR_WARNING = "#ff4444"  # Warning, errors, and close actions

# Fonts (Strictly Monospace)
FONT_FAMILY = "Consolas"    # Will fallback to Courier if unavailable
FONT_TITLE = (FONT_FAMILY, 18, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 11, "normal")
FONT_HEADER = (FONT_FAMILY, 12, "bold")
FONT_BODY = (FONT_FAMILY, 10, "normal")
FONT_SMALL = (FONT_FAMILY, 9, "normal")
FONT_DIGITS = (FONT_FAMILY, 24, "bold")

# -------------------------------------------------------------------------
# DETECTOR & WEB CAMERA CONFIG
# -------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"      # YOLOv8 nano model
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence to accept a detection
CAMERA_INDEX = 0              # Default webcam index
FRAME_WIDTH = 640             # Resized width for processing & display
FRAME_HEIGHT = 360            # Resized height for processing & display
STATS_UPDATE_MS = 100         # All stats refresh rate (10Hz)

# -------------------------------------------------------------------------
# RETAIL SETTINGS
# -------------------------------------------------------------------------
RETAIL_CLASS_FILTER = "person" # Focus unique counts/dwell times on "person"

# -------------------------------------------------------------------------
# COLOR UTILITIES
# -------------------------------------------------------------------------
def get_id_color_bgr(track_id):
    """
    Generates a bright, highly saturated color based on track_id hash (BGR format for OpenCV).
    Ensures no dark colors, pastels, or gray shades.
    """
    # Deterministic hash of the track_id
    hash_val = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    
    # Generate golden-ratio based hue to spread colors evenly
    hue = (hash_val * 0.618033988749895) % 1.0
    saturation = 0.90 + (hash_val % 10) * 0.01  # 90% - 100%
    value = 0.90 + ((hash_val >> 4) % 10) * 0.01 # 90% - 100%
    
    # Convert HSV to RGB (0-1 floats)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Return as BGR tuple (0-255) for OpenCV drawing
    return int(b * 255), int(g * 255), int(r * 255)

def get_id_color_hex(track_id):
    """
    Generates a bright, highly saturated color based on track_id hash (Hex format for Tkinter UI).
    """
    b, g, r = get_id_color_bgr(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"
