# Installation & Setup Guide

This guide walks you through setting up **Block Blast AI** from scratch, connecting your Android device via ADB, and launching your first autonomous game session.

---

## 1. System Requirements

### Host Machine (PC / Mac / Linux)
- **Operating System**: Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+).
- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13.
- **Android Debug Bridge (ADB)**: Installed and accessible in your system `PATH`.
- **USB Port**: USB 3.0 port recommended for lowest capture latency.

### Android Device
- Any Android smartphone or tablet running Android 8.0 or newer.
- Block Blast game installed from Google Play Store or APK.
- Developer options enabled.

---

## 2. Setting Up Android Debug Bridge (ADB)

ADB allows your computer to capture the screen and send touch commands to your phone.

### Installing ADB

#### Windows
1. Download Android SDK Platform-Tools from Google:  
   [https://developer.android.com/tools/releases/platform-tools](https://developer.android.com/tools/releases/platform-tools)
2. Extract the ZIP folder (e.g. to `C:\platform-tools`).
3. Add the folder path to your System Environment Variable `PATH`:
   - Press `Win + R`, type `sysdm.cpl`, press Enter.
   - Go to **Advanced** > **Environment Variables**.
   - Under **System variables**, select `Path` > **Edit** > **New** > enter `C:\platform-tools`.
4. Open a new terminal and verify:
   ```bash
   adb version
   ```

#### macOS (Homebrew)
```bash
brew install android-platform-tools
adb version
```

#### Linux (Debian / Ubuntu)
```bash
sudo apt update && sudo apt install -y adb
adb version
```

---

## 3. Configuring Your Android Device

### Enable Developer Options
1. Open **Settings** on your Android device.
2. Go to **About Phone** > **Software Information**.
3. Tap **Build Number** 7 times until you see the message *"You are now a developer!"*.

### Enable USB Debugging
1. Go back to **Settings** > **Developer Options**.
2. Turn ON **USB Debugging**.
3. *(Xiaomi / MIUI / HyperOS only)*: Also enable **USB Debugging (Security settings)** to allow touch and gesture injection over ADB.

### Verify Connection
Connect your phone to your PC via a USB cable. On your phone screen, a prompt will appear:
> **"Allow USB debugging?"**  
> Check *Always allow from this computer* and tap **Allow**.

In your computer terminal, run:
```bash
adb devices
```
You should see your device listed as `device`:
```text
List of devices attached
1234567890ABCDEF    device
```

If it says `unauthorized`, unlock your phone screen and accept the dialog prompt.

---

## 4. Installing Block Blast AI

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/block-blast-ai.git
cd block-blast-ai
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the Unit Test Suite
Verify that all 246 tests pass in your environment:
```bash
python -m unittest discover tests
```
Expected output:
```text
Ran 246 tests in ~45s
OK
```

---

## 5. First Run & Preflight Checks

### Test Dry-Run Mode (Safe Verification)
Before letting the AI take physical control of your phone's touch screen, test the computer vision and perception loop in read-only **dry-run mode**:

1. Open **Block Blast** on your phone.
2. Start a fresh game so the 8x8 grid is visible.
3. Keep the phone screen ON and unlocked.
4. Run:
   ```bash
   python run_autonomous_game.py --dry-run
   ```

The AI will:
- Connect to your ADB device automatically.
- Capture the frame.
- Detect the 8x8 grid bounding box.
- Identify the 3 pieces in your tray.
- Print live telemetry without executing physical swipes.

Press `Ctrl+C` or press `Q` in the console to exit.

---

## 6. Live Autonomous Play

Once dry-run confirms reliable perception:

```bash
python run_autonomous_game.py
```

### Interactive Console Keybindings
While running, you can control execution in real time:
- **`S`**: Start / Resume autonomous play.
- **`P`**: Pause execution and immediately release any active touch pointer.
- **`Space`**: Execute a single step (1 turn) and wait.
- **`Q`**: Stop gracefully and restore device state.

---

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `adb: command not found` | ADB is not in system PATH | Follow Section 2 to add platform-tools to PATH. |
| `Device unauthorized` | Debugging permission not granted | Unlock device, reconnect USB cable, and tap **Allow**. |
| `Touch events not registering` | Device security restriction (Xiaomi/MIUI) | Enable **USB Debugging (Security settings)** in Developer Options. |
| `Board not detected` | Screen rotated or game not in foreground | Ensure phone is in portrait mode and Block Blast board is visible. |
| `High capture latency (>1000ms)` | USB 2.0 port or slow ADB shell | Connect to USB 3.0 port. Block Blast AI will automatically utilize optimized raw socket capture when available. |
