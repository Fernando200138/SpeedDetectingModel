# MemePoseMatcher

A fun, real-time application that detects your upper-body pose and facial expressions using MediaPipe, classifies them into three meme-driven expressions, and displays matching 3-second video clips in a GUI overlay.

## Features

- **Real-time detection** using MediaPipe Pose and Face Mesh
- **Expression classification** into two meme categories:
  - 🗣️ **messi_yell**: Wide-open mouth (yelling/shouting)
  - 😊 **mom_homeless_laugh**: Eyes closed, contained laughter expression
- **Neutral state**: When no strong expression is detected, the app continues to loop the last matched clip
- **Live metrics display**: See real-time values for eye aspect ratio (EAR) and mouth aspect ratio (MAR)
- **Smooth classification** with temporal windowing and debouncing to avoid jitter

## Setup

Use Python 3.9–3.12 for MediaPipe 0.10.21. The application uses the legacy `mp.solutions` API, which is not available in MediaPipe 1.x.

### 1. Clone/Create the Project

```bash
cd /home/alex/projects/MachineLearningMeme
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Meme Video Clips

Place your two 3-second video files into the `data/videos/` directory:

- `speed_messi.mp4` - for the yelling/wide-mouth expression
- `speed_mom_homeless.mp4` - for the contained laughing expression

**Example**:
```bash
ls data/videos/
# Output:
# speed_bark_lilnasx.mp4
# speed_mom_homeless.mp4
# speed_messi.mp4
```

> **Note**: If videos are missing, the app will still run but won't display overlays. You'll see a warning at startup.

## Running the App

```bash
python -m src.app
```

You should see:
1. A large window showing your webcam feed with pose/face skeleton overlays
2. A small video panel in the top-left corner playing the matched expression clip
3. Real-time metrics (EAR, MAR, arm extension) displayed on screen
4. Console output with frame statistics and expression labels

## Controls

- **`q`**: Quit the application
- **`m`**: Toggle metrics display on/off

## Customization

### Tuning Classifier Thresholds

Edit the thresholds in `src/classifier.py`:

```python
self.MAR_YELL_THRESHOLD = 0.45       # increase for harder yelling detection
self.EAR_CLOSED_THRESHOLD = 0.18     # adjust eye-closed sensitivity
```

### Changing Video Panel Size

In `src/app.py`, modify the `VideoPlayer` initialization:

```python
video_player = VideoPlayer(label_to_path=label_to_path, panel_size=(560, 300))  
```

### Changing Webcam Resolution

In `src/app.py`:

```python
capture = MediaPipeCapture(cam_index=0, webcam_width=920, webcam_height=780)
```

## Troubleshooting

- **Webcam not detected**: Check that your camera is connected and not in use by another application. Try `cam_index=1` or `cam_index=2` in `src/app.py`.
- **Poor pose/face detection**: Ensure good lighting and that you're facing the camera.
- **Videos not playing**: Verify that video files are in `data/videos/` and have the correct filenames.
- **Expression not classifying correctly**: Adjust the thresholds in `src/classifier.py` and try exaggerating your expression.

## Project Structure

```
meme_pose_matcher/
├── data/
│   └── videos/                     # Your three video files go here
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── src/
    ├── __init__.py                 # Package marker
    ├── capture.py                  # MediaPipe webcam + landmark capture
    ├── pose_utils.py               # Landmark processing & metrics (EAR, MAR, angles)
    ├── classifier.py               # Expression classification logic (rule-based)
    ├── video_player.py             # Video clip playback manager
    ├── ui.py                       # OpenCV GUI and overlay rendering
    └── app.py                      # Main application entry point
```

## How It Works

1. **Capture**: MediaPipe Pose and Face Mesh detect landmarks from your webcam feed (33 pose points, 468 face points).
2. **Metrics**: Eye aspect ratio (EAR), mouth aspect ratio (MAR), and head movement are calculated.
3. **Classification**: A rule-based classifier examines these metrics over a 0.6-second temporal window and classifies into one of three states: `messi_yell`, `mom_homeless_laugh`, or `neutral`.
4. **Debouncing**: To avoid rapid label jitter, the classifier requires a 0.25-second hold before switching to a new expression.
5. **Playback**: The `VideoPlayer` displays the matched clip (or repeats the last active clip during neutral states).
6. **UI**: OpenCV window shows the camera feed with skeleton overlays, live metrics, and the video panel.

## Future Enhancements

- Train a small neural net classifier for more robust expression detection
- Add support for more expressions or custom meme clips
- Implement eye-gaze tracking for additional metrics
- Add sound (play audio tracks alongside video)
- Support multiple simultaneous detection (detect multiple faces)
- Add pose confidence/validity filtering

## License

This project is provided as-is for fun and educational purposes.

---

**Enjoy making memes with your own expressions!** 🎬😄
