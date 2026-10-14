# 📷 AI ADAS Live Device Camera Detection

Real-time, interactive camera detection window using your computer's built-in webcam or connected external USB/virtual cameras.

---

## 🌟 Key Capabilities

1. **Hardware Camera Discovery & Dynamic Switching**
   - Automatically scans and enumerates connected cameras (e.g., `Camera 0`, `Camera 1`).
   - Seamlessly switch between laptop internal camera and external webcam with a single keypress or mouse click.
2. **YOLO11 Perception + Rider Disambiguation**
   - Powered by the fast Ultralytics **YOLO11 Nano** model.
   - Humans/pedestrians/animals are classified as **`ROADSIDE ENTITY`** (identical to the uploaded video pipeline).
   - Vehicles (cars, trucks, buses, motorcycles, bicycles) are recognized and tracked with persistent IDs.
3. **Calibrated Monocular Distance & Kinematics**
   - Fuses geometric flat-road projection and vehicle height priors to compute estimated metric distance (`~X.X m (est)`).
   - Evaluates closing speed ($\text{km/h}$) and Time-to-Collision ($\text{TTC}$).
4. **Spatial Ego-Corridor & Lane Reasoner**
   - Visualizes the ego driving path with a translucent green perspective trapezoid.
   - Monitors blind-spot intrusions and lateral cut-ins.
5. **Hybrid ML Safety Guardrails**
   - Color-coded targeting corner brackets and telemetry pills:
     - 🟢 **LOW / SAFE**: Normal driving envelope.
     - 🟡 **MEDIUM / CAUTION**: Approaching obstacles, adjacent lane activity.
     - 🔴 **CRITICAL / HAZARD**: Imminent collision buffer or extreme headway.

---

## 🚀 How to Run

### Option 1: Double-Click the Launcher (Windows)
Double-click [`run_camera.bat`](run_camera.bat) inside this folder.

### Option 2: Run via Terminal
From the project root:
```bash
python live_camera/live_detector.py
```

---

## ⌨️ Interactive Controls & Shortcuts

| Action | Control Shortcut | Description |
| :--- | :---: | :--- |
| **Switch Camera** | **`C`**, **`Space`**, or **Click `CAM #` Badge** | Cycles immediately to the next detected camera on your device. |
| **Fullscreen** | **`F`** | Toggles full-screen window display mode. |
| **HUD Mode** | **`M`** | Toggles between Full ADAS HUD and Minimalist Target Box mode. |
| **Reset Tracker** | **`R`** | Flushes active tracking history and reinitializes ByteTrack IDs. |
| **Exit** | **`Q`** or **`ESC`** | Closes the camera stream and exits the application. |
