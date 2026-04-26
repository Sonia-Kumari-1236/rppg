import os, cv2, uuid, time, threading, base64
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify
from scipy.signal import butter, filtfilt, find_peaks, welch

app = Flask(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BUFFER_SEC     = 10
TARGET_FPS     = 15
BPM_LOW        = 50
BPM_HIGH       = 150
WINDOW_SEC     = 8
UPDATE_EVERY   = 2.0
MOTION_THRESH  = 8.0
BPM_SMOOTH_N   = 8
MAX_BPM_JUMP   = 10
SESSION_TTL    = 90
MAX_SESSIONS   = 3

# ── Lightweight face detector (Haar cascade, ~1MB, no heavy ML runtime) ────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

# ── Session store ──────────────────────────────────────────────────────────────
_sessions: dict = {}
_store_lock = threading.Lock()


def _new_session():
    max_buf = int(TARGET_FPS * BUFFER_SEC)
    win_buf = int(TARGET_FPS * WINDOW_SEC)
    return {
        "lock":        threading.Lock(),
        "r_buf":       deque(maxlen=max_buf),
        "g_buf":       deque(maxlen=max_buf),
        "b_buf":       deque(maxlen=max_buf),
        "time_buf":    deque(maxlen=max_buf),
        "fps_est":     float(TARGET_FPS),
        "bpm_str":     "--",
        "bpm_col":     [200, 200, 200],
        "best_algo":   "--",
        "snr_vals":    {},
        "bpm_history": deque(maxlen=BPM_SMOOTH_N),
        "last_update": 0.0,
        "prev_gray":   None,
        "wave_buf":    deque(maxlen=150),
        "win_buf":     win_buf,
        "last_seen":   time.time(),
        "last_face":   None,   # cached face bbox
    }


def _get_session(sid: str):
    with _store_lock:
        now = time.time()
        stale = [k for k, v in _sessions.items() if now - v["last_seen"] > SESSION_TTL]
        for k in stale:
            del _sessions[k]
        if sid not in _sessions and len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k]["last_seen"])
            del _sessions[oldest]
        if sid not in _sessions:
            _sessions[sid] = _new_session()
        _sessions[sid]["last_seen"] = time.time()
        return _sessions[sid]


# ── Face + forehead ROI extraction ────────────────────────────────────────────

def detect_face(gray):
    """Returns (x,y,w,h) of best face or None."""
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick largest face
    return max(faces, key=lambda f: f[2]*f[3])


def forehead_roi(frame, face):
    x, y, w, h = face
    # top 20% of face bbox = forehead
    fy1 = y + int(h * 0.05)
    fy2 = y + int(h * 0.22)
    fx1 = x + int(w * 0.20)
    fx2 = x + int(w * 0.80)
    fy1, fy2 = max(fy1,0), min(fy2, frame.shape[0])
    fx1, fx2 = max(fx1,0), min(fx2, frame.shape[1])
    roi = frame[fy1:fy2, fx1:fx2]
    if roi.size == 0:
        return None, None
    return roi, (fx1, fy1, fx2-fx1, fy2-fy1)


def left_cheek_roi(frame, face):
    x, y, w, h = face
    cy1 = y + int(h * 0.45)
    cy2 = y + int(h * 0.70)
    cx1 = x + int(w * 0.05)
    cx2 = x + int(w * 0.30)
    cy1, cy2 = max(cy1,0), min(cy2, frame.shape[0])
    cx1, cx2 = max(cx1,0), min(cx2, frame.shape[1])
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None, None
    return roi, (cx1, cy1, cx2-cx1, cy2-cy1)


def right_cheek_roi(frame, face):
    x, y, w, h = face
    cy1 = y + int(h * 0.45)
    cy2 = y + int(h * 0.70)
    cx1 = x + int(w * 0.70)
    cx2 = x + int(w * 0.95)
    cy1, cy2 = max(cy1,0), min(cy2, frame.shape[0])
    cx1, cx2 = max(cx1,0), min(cx2, frame.shape[1])
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None, None
    return roi, (cx1, cy1, cx2-cx1, cy2-cy1)


# ── Signal processing ──────────────────────────────────────────────────────────

def _norm(s):
    std = s.std()
    return (s - s.mean()) / std if std > 1e-8 else s - s.mean()

def _detrend(s):
    t = np.arange(len(s))
    return s - np.polyval(np.polyfit(t, s, 1), t)

def _bp(s, fps):
    if len(s) < 15: return s
    nyq = fps / 2.0
    lo  = float(np.clip(BPM_LOW /60/nyq, 1e-4, 0.99))
    hi  = float(np.clip(BPM_HIGH/60/nyq, 1e-4, 0.99))
    if lo >= hi: return s
    b, a = butter(3, [lo, hi], btype="band")
    return filtfilt(b, a, s)

def chrom(R, G, B):
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    sx, sy = X.std() or 1., Y.std() or 1.
    return X - (sx/sy)*Y

def pos(R, G, B):
    rn = R / (R.mean() or 1e-8)
    gn = G / (G.mean() or 1e-8)
    bn = B / (B.mean() or 1e-8)
    S1 = rn - gn
    S2 = rn + gn - 2*bn
    s1, s2 = S1.std() or 1., S2.std() or 1.
    return S1 + (s1/s2)*S2

def snr_score(s, fps):
    if len(s) < 8: return float("-inf")
    fv = np.abs(np.fft.rfft(s))**2
    fr = np.fft.rfftfreq(len(s), 1/fps)
    mask = (fr >= BPM_LOW/60) & (fr <= BPM_HIGH/60)
    if not mask.any() or mask.all(): return float("-inf")
    n = fv[~mask].sum()
    return float(fv[mask].sum()/n) if n > 1e-12 else float("inf")

def bpm_welch(s, fps):
    nperseg = min(len(s), int(fps*8))
    fr, psd = welch(s, fps, nperseg=nperseg)
    mask    = (fr >= BPM_LOW/60) & (fr <= BPM_HIGH/60)
    if not mask.any(): return None
    return float(fr[mask][np.argmax(psd[mask])] * 60)

def bpm_peaks(s, fps):
    md = max(int(fps*60/BPM_HIGH), 1)
    pro = 0.15*(s.max()-s.min()) if s.max()!=s.min() else 0.01
    pk, _ = find_peaks(s, distance=md, prominence=pro)
    if len(pk) < 3: return None
    ibi = np.median(np.diff(pk)/fps)
    return float(60/ibi) if ibi > 0 else None

def estimate_bpm(s, fps):
    a = bpm_welch(s, fps)
    b = bpm_peaks(s, fps)
    valid = [x for x in [a, b] if x and BPM_LOW <= x <= BPM_HIGH]
    if not valid: return None
    if len(valid)==2 and abs(valid[0]-valid[1]) < 10:
        return float(np.mean(valid))
    return b or a

def bpm_color(bpm):
    if 55 <= bpm <= 100: return [0, 230, 60]
    if 45 <= bpm <= 120: return [0, 200, 255]
    return [0, 80, 255]


# ── Core frame processor ───────────────────────────────────────────────────────

def _process(frame, sid, sess):
    now  = time.time()
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out = dict(
        session_id=sid, face_found=False, landmarks=[],
        rois={}, bpm=sess["bpm_str"], bpm_color=sess["bpm_col"],
        motion=False, any_covered=False, pose="forward",
        best_algo=sess["best_algo"], snr_vals=sess["snr_vals"],
        wave=list(sess["wave_buf"]), buffer_pct=0,
        fps=round(sess["fps_est"]), frame_w=W, frame_h=H,
    )

    def reset():
        sess["bpm_str"] = "--"; sess["best_algo"] = "--"
        sess["bpm_col"] = [200,200,200]; sess["snr_vals"] = {}
        for q in ("bpm_history","wave_buf","time_buf","r_buf","g_buf","b_buf"):
            sess[q].clear()
        sess["prev_gray"] = None; sess["last_face"] = None

    # Detect face every 5 frames, cache otherwise
    face = None
    if sess["last_face"] is not None:
        face = sess["last_face"]
    detected = detect_face(gray)
    if detected is not None:
        sess["last_face"] = detected
        face = detected
    elif face is None:
        reset()
        return {**out, "bpm":"--", "bpm_color":[200,200,200]}

    # Motion check
    motion = False
    if sess["prev_gray"] is not None:
        diff   = np.abs(gray.astype(np.float32) - sess["prev_gray"].astype(np.float32))
        motion = float(diff.mean()) > MOTION_THRESH
    sess["prev_gray"] = gray.copy()

    # Extract ROIs
    rois_out = {}
    combined_r, combined_g, combined_b = [], [], []

    for name, roi_fn in [("forehead", forehead_roi),
                          ("left_cheek", left_cheek_roi),
                          ("right_cheek", right_cheek_roi)]:
        roi, bbox = roi_fn(frame, face)
        if roi is None: continue
        r_mean = roi[:,:,2].astype(float).mean()
        g_mean = roi[:,:,1].astype(float).mean()
        b_mean = roi[:,:,0].astype(float).mean()
        combined_r.append(r_mean)
        combined_g.append(g_mean)
        combined_b.append(b_mean)
        bx, by, bw, bh = bbox
        rois_out[name] = {
            "poly": [[bx/W, by/H],[( bx+bw)/W, by/H],
                     [(bx+bw)/W,(by+bh)/H],[bx/W,(by+bh)/H]],
            "covered": False,
            "color": [0, 220, 255] if name=="forehead" else [0,255,80],
        }

    if not combined_r:
        reset()
        return {**out, "bpm":"--", "bpm_color":[200,200,200]}

    r_val = float(np.mean(combined_r))
    g_val = float(np.mean(combined_g))
    b_val = float(np.mean(combined_b))

    if not motion:
        sess["r_buf"].append(r_val)
        sess["g_buf"].append(g_val)
        sess["b_buf"].append(b_val)
        sess["time_buf"].append(now)
        n = len(sess["time_buf"])
        if n > 10:
            el = sess["time_buf"][-1] - sess["time_buf"][-10]
            if el > 0: sess["fps_est"] = 9.0/el

    n    = len(sess["time_buf"])
    need = max(int(sess["fps_est"]*8), 30)
    buf_pct = min(int(n/need*100), 100)

    # BPM estimation
    if (now - sess["last_update"]) >= UPDATE_EVERY and n >= need:
        sess["last_update"] = now
        win = min(sess["win_buf"], n)
        R = np.array(list(sess["r_buf"])[-win:])
        G = np.array(list(sess["g_buf"])[-win:])
        B = np.array(list(sess["b_buf"])[-win:])
        Rn = _detrend(_norm(R))
        Gn = _detrend(_norm(G))
        Bn = _detrend(_norm(B))
        cands = {
            "CHROM": _bp(chrom(Rn, Gn, Bn), sess["fps_est"]),
            "POS":   _bp(pos(Rn, Gn, Bn),   sess["fps_est"]),
            "GREEN": _bp(Gn,                  sess["fps_est"]),
        }
        snrs = {k: snr_score(v, sess["fps_est"]) for k,v in cands.items()}
        best = max(snrs, key=snrs.get)
        final = cands[best]
        sess["wave_buf"].clear()
        for v in final[-150:]: sess["wave_buf"].append(float(v))
        bpm = estimate_bpm(final, sess["fps_est"])
        if bpm:
            hist = sess["bpm_history"]
            if not hist or abs(bpm - np.mean(hist)) < MAX_BPM_JUMP:
                hist.append(bpm)
            sm = float(np.mean(hist))
            sess["bpm_str"] = f"{sm:.0f}"
            sess["bpm_col"] = bpm_color(sm)
        sess["best_algo"] = best
        sess["snr_vals"]  = {k: round(float(v),1) for k,v in snrs.items()}

    fx, fy, fw, fh = face
    return {**out,
        "face_found":  True,
        "landmarks":   [[fx/W, fy/H], [(fx+fw)/W, (fy+fh)/H]],
        "rois":        rois_out,
        "bpm":         sess["bpm_str"],
        "bpm_color":   sess["bpm_col"],
        "motion":      motion,
        "any_covered": False,
        "pose":        "forward",
        "best_algo":   sess["best_algo"],
        "snr_vals":    sess["snr_vals"],
        "wave":        list(sess["wave_buf"]),
        "buffer_pct":  buf_pct,
        "fps":         round(sess["fps_est"]),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    body = request.get_json(force=True)
    sid  = body.get("session_id") or str(uuid.uuid4())
    raw  = body.get("frame", "")
    if not raw:
        return jsonify({"error": "no frame"}), 400
    try:
        b64   = raw.split(",")[1] if "," in raw else raw
        arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("decode failed")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    sess = _get_session(sid)
    with sess["lock"]:
        return jsonify(_process(frame, sid, sess))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
