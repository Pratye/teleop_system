# MultiCamStreamer — iOS App

Streams synchronized wide + telephoto frames over TCP so the Mac pipeline can
compute stereo depth without any neural network.

## Requirements

| Item | Minimum |
|---|---|
| iPhone | iPhone X or later (multi-cam sessions need A12+ for simultaneous wide+tele) |
| iOS | 18.0+ (deployment target aligned with Xcode project at `/MultiCamStreamer`; e.g. 18.6.2 is fine) |
| Mac | macOS 13+ |
| Xcode | 15+ |

> iPhone 16 has a **1× wide + 2× telephoto** dual system. Both can run
> simultaneously via `AVCaptureMultiCamSession`.

---

## Xcode Setup (5 minutes)

1. **Open Xcode → File → New → Project → App**
   - Product Name: `MultiCamStreamer`
   - Interface: SwiftUI
   - Language: Swift
   - Bundle ID: `com.yourname.MultiCamStreamer`

2. **Replace generated files** with the four `.swift` files in this folder:
   - `MultiCamStreamerApp.swift`
   - `ContentView.swift`
   - `MultiCamSession.swift`

3. **Add Info.plist keys** — open your project's `Info.plist` and add:

   | Key | Type | Value |
   |---|---|---|
   | `NSCameraUsageDescription` | String | `"Stereo streaming"` |
   | `NSLocalNetworkUsageDescription` | String | `"Local TCP server"` |
   | `NSBonjourServices` | Array → String | `_stro._tcp` |

4. **Sign & deploy** — select your iPhone as target, press ▶.
   - Trust the developer certificate on iPhone: Settings → General → VPN & Device Management.

5. The app shows your **iPhone's local IP** — use that as `--iphone-ip`.

---

## Running the full pipeline

```bash
# Step 1 — stereo calibration (once, ~2 minutes)
# Hold a 9×6 checkerboard (25 mm squares) in view of both cameras
python scripts/calib_stereo_iphone.py \
    --iphone-ip 192.168.1.42 \
    --board-cols 9 --board-rows 6 --square-mm 25 \
    --show

# Step 2 — run teleop with stereo depth
mjpython -m src.main --mode sim \
    --stereo-iphone --iphone-ip 192.168.1.42 \
    --stereo-calib config/stereo_iphone_calib.npz \
    --show-cv
```

### Without calibration (coarse mode)
```bash
mjpython -m src.main --mode sim \
    --stereo-iphone --iphone-ip 192.168.1.42 \
    --show-cv
```
Coarse mode uses the Apple-provided intrinsics per frame but assumes
the cameras are roughly parallel and uses a hard-coded 12 mm baseline.
Depth accuracy is ~±15 cm at arm distance. Good enough for testing.

### Test the stereo stream independently
```bash
python -m src.stereo_iphone --iphone-ip 192.168.1.42 --show
```

---

## Depth accuracy vs distance

| Distance | Baseline 12 mm | After calibration (subpixel) |
|---|---|---|
| 0.4 m | ±4 cm | ±1.5 cm |
| 0.7 m | ±12 cm | ±4 cm |
| 1.2 m | ±35 cm | ±12 cm |

For arm tracking (0.4–0.8 m reach), calibrated stereo gives useful accuracy.
MediaPipe world 3D landmarks are better for joint **angles** but stereo wins
for absolute **wrist XYZ position** at close range.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Multi-cam not supported" | Check device: needs iPhone X + iOS 16 |
| No connection from Mac | Check same Wi-Fi network; disable iPhone firewall in iOS Settings |
| High latency | Use USB tethering instead of Wi-Fi |
| Disparity map all zeros | Run `calib_stereo_iphone.py` first; check `--sgbm-min-disp` sign |
| `cornerSubPix` fails | Use a flat, evenly lit checkerboard; avoid glare |
