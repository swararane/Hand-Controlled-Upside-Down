Open the Gate to the Upside Down

Quick start

1. Create a Python virtual environment (recommended).

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the app:

```powershell
python main.py
```

Controls / Gestures (visualized on-screen):
- Two-hand stretch: open the portal (move hands apart)
- Pinch + drag: pinch with thumb+index and move to reposition portal
- Rotate hand: rotate wrist to apply twist
- Palm push: push forward (toward camera) to trigger close/shockwave

Keyboard toggles:
- `u`: Toggle UpsideDown visual mode (color grade, CRT, chromatic aberration)
- `d`: Toggle demo mode (auto open/close every few seconds)
- `ESC`: Quit

Automatic mode switching:
- The app now auto-switches to Normal mode when no hands are detected for ~1.5s.
- Showing your hands will switch to the Upside Down visual mode automatically.

Created and developed by @tubakhxn

Inspired by the mood and visuals of the "Upside Down" concept from the Stranger Things universe — this project is a fan-made visual demo and is not affiliated with or endorsed by the rights holders.

Forking & Contributing
- To fork this project on GitHub: visit the repository page and click the "Fork" button (or clone then push to your own repo).
- Clone your fork locally and make changes. Create feature branches and submit pull requests for review.
- Please include descriptive commit messages and keep changes focused.

No license included
- This repository intentionally does not include a license file. If you plan to reuse or distribute this project, please consult the author (`@tubakhxn`) and add an appropriate open-source license to your fork if desired.

Credits & Dependencies
- Built with Python, OpenCV, Mediapipe, and NumPy.
- See `requirements.txt` for exact package versions.

Have fun — show your hands to open the gate.

Notes
- Requires a webcam and decent lighting.
- If Mediapipe fails to detect hands, adjust camera or lighting.
