# rPPG Heart Rate Monitor — Web Version

Real-time heart rate estimation from your webcam, running in the browser.

## How it works

- Your browser captures webcam frames and sends them to the Flask backend (~10fps)
- The backend runs MediaPipe face mesh + rPPG signal processing (CHROM / POS / ICA)
- Estimated BPM and overlay data are returned to the browser for display
- No video is stored — frames are processed in memory and discarded

## Deploy on Render (free tier)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USER/rppg-web.git
git push -u origin main
```

### 2. Create a Render Web Service

1. Go to https://render.com and sign in
2. Click **New → Web Service**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` — click **Deploy**

That's it. Render will install dependencies and start the app.
First deploy takes ~3–5 minutes (MediaPipe is large).

> **Note:** The free tier sleeps after 15 min of inactivity.
> The first request after sleep takes ~30 seconds to wake up.

## Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Project structure

```
rppg-web/
├── app.py              # Flask backend + all signal processing
├── templates/
│   └── index.html      # Frontend (webcam capture + canvas overlay)
├── requirements.txt
├── render.yaml         # Render deploy config
└── runtime.txt         # Python version pin
```
