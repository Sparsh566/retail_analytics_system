# Real-Time Object Detection, Tracking & Retail Analytics System

An enterprise-grade desktop computer vision system that performs real-time multi-object detection and comparative tracking using **YOLOv8**, **SORT**, and **DeepSORT** models. The application is packaged into a professional Python structure with a monospace-themed dark GUI featuring side-by-side stream displays, comparative stability benchmarks, unique customer footfall analysis, in-frame dwell times, and event tracking control logs.

---

## 📂 Project Architecture and Structure

The codebase is organized into a modular package architecture following software engineering best practices:

```
retail_analytics_system/
├── src/                      # Core application package
│   ├── __init__.py           # Package-level initializer
│   ├── app.py                # Main orchestrator (MainController)
│   ├── config.py             # Design system styling tokens & configuration variables
│   ├── trackers/             # Object tracking subdomain
│   │   ├── __init__.py       # Trackers package initializer
│   │   ├── sort.py           # Custom SORT tracker (Kalman Filter + scipy Hungarian)
│   │   └── deepsort.py       # deep-sort-realtime wrapper & spatial-SORT fallback
│   ├── vision/               # Computer vision & frame grabbing subdomain
│   │   ├── __init__.py       # Vision package initializer
│   │   └── detector.py       # YOLOv8 engine & 2D Top-down security CCTV shop simulator
│   └── ui/                   # Desktop User Interface subdomain
│       ├── __init__.py       # UI package initializer
│       └── dashboard.py      # Tkinter Splash Screen & Live multi-column Dashboard
├── main.py                   # Lightweight root-level bootstrap launcher
├── requirements.txt          # Python dependency manifest
├── run.bat                   # Automated Windows virtual environment setup & boot script
├── .gitignore                # Git configuration directory exclusions
└── tests/                    # Core unit and integration tests
    └── __init__.py           # Tests package initializer
```

---

## ⚙️ How It Works (Scientific Underpinnings)

### 1. Object Detection (YOLOv8)
The system leverages a pre-trained **YOLOv8-Nano (yolov8n.pt)** convolutional neural network at runtime to identify spatial targets on each video frame with a default confidence threshold of `0.5`. 

### 2. Simple Online and Realtime Tracking (SORT)
Our custom SORT tracker establishes temporal object tracks purely through spatial dynamics:
*   **Kalman Filtering**: Each track is modeled via a constant-velocity Kalman Filter. The state vector is defined as:
    $$x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$
    where $u, v$ represent the horizontal and vertical center of the bounding box, $s$ is the scale (bounding box area), $r$ is the constant aspect ratio, and the remaining values represent velocity components.
*   **Hungarian Association**: In subsequent frames, detections are associated with existing track predictions by forming an Intersection over Union (IoU) cost matrix and solving the linear sum assignment problem using `scipy.optimize.linear_sum_assignment`.

### 3. Deep Association Tracking (DeepSORT)
To mitigate tracking failures caused by rapid camera movement or occlusions (e.g. objects crossing paths), DeepSORT incorporates appearance descriptors:
*   **Deep Re-ID Feature Extraction**: Employs a pre-trained MobileNet neural network to extract a low-dimensional appearance descriptor vector from each detected bounding box.
*   **Cosine Similarity Matching**: Associates tracks based on both spatial Kalman predictions and deep cosine similarity distances. If a tracked object is temporarily blocked, its appearance vector allows the system to re-identify and preserve its unique ID when it reappears.

---

## 📈 Retail Intelligence Features

Designed specifically for commercial operations, boutique footfall tracking, and customer flow analysis:
*   **Total Customer Footfall**: Tracks and counts unique session IDs assigned specifically to the `"person"` category.
*   **Customer Dwell Duration**: Real-time ticker displaying active seconds for currently visible shoppers.
*   **Security Event Logs**: Terminal-style rolling logger detailing real-time entries, exits, and cumulative dwell durations:
    ```
    [14:10:02] ENTERED: Customer DS_3 (person)
    [14:10:45] EXITED:  Customer DS_3 (person) - Dwell: 43.2s
    ```

---

## 💻 Quickstart (Windows Deployment)

Deploy the system with a single command launcher. The startup script handles Python checks, initializes a virtual environment, checks and downloads pip dependencies, and runs the application.

1.  **Clone the Repository**:
    ```powershell
    git clone https://github.com/Sparsh566/retail_analytics_system.git
    cd retail_analytics_system
    ```
2.  **Launch the System**:
    Double-click `run.bat` in File Explorer, or execute it in PowerShell:
    ```powershell
    .\run.bat
    ```

---

## 🛡️ Fail-Safe Simulator Mode
If the host machine lacks a connected webcam or an internet connection to load the YOLO model weights on initial boot, the system automatically engages **CCTV Simulator Mode**. It renders a beautifully styled overhead boutique store layout, creating virtual shoppers walking between apparel shelves, food aisles, and checkout stations. It injects coordinate jitter and occlusions to showcase the tracking performance side-by-side in real-time.
