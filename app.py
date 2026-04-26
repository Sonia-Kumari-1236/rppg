import os, uuid, time, threading, base64, math
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BUFFER_SEC   = 10
TARGET_FPS   = 15
BPM_LOW      = 50
BPM_HIGH     = 150
UPDATE_EVERY = 2.0
MOTION_THRESH= 8.0
BPM_SMOOTH_N = 8
MAX_BPM_JUMP = 10
SESSION_TTL  = 90
MAX_SESSIONS = 4

_sessions: dict = {}
_store_lock = threading.Lock()

def _new_session():
    max_buf = int(TARGET_FPS * BUFFER_SEC)
    return {
        "lock":        threading.Lock(),
        "g_buf":       deque(maxlen=max_buf),
        "r_buf":       deque(maxlen=max_buf),
        "b_buf":       deque(maxlen=max_buf),
        "time_buf":    deque(maxlen=max_buf),
        "fps_est":     float(TARGET_FPS),
        "bpm_str":     "--",
        "bpm_col":     [200, 200, 200],
        "best_algo":   "--",
        "snr_vals":    {},
        "bpm_history": deque(maxlen=BPM_SMOOTH_N),
        "last_update": 0.0,
        "wave_buf":    deque(maxlen=120),
        "last_seen":   time.time(),
        "prev_pixel":  None,
    }

def _get_session(sid):
    with _store_lock:
        now = time.time()
        for k in [k for k,v in _sessions.items() if now-v["last_seen"]>SESSION_TTL]:
            del _sessions[k]
        if sid not in _sessions and len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k]["last_seen"])
            del _sessions[oldest]
        if sid not in _sessions:
            _sessions[sid] = _new_session()
        _sessions[sid]["last_seen"] = time.time()
        return _sessions[sid]

def _detrend(s):
    n = len(s)
    t = np.arange(n)
    m = np.polyfit(t, s, 1)
    return s - (m[0]*t + m[1])

def _norm(s):
    std = s.std()
    return (s - s.mean()) / std if std > 1e-8 else s*0

def butter_bp_simple(s, fps):
    if len(s) < 6:
        return s
    long_win  = max(3, min(int(fps * 1.5), len(s)//2))
    short_win = max(2, min(int(fps * 0.4), len(s)//4))
    def ma(x, w):
        kernel = np.ones(w)/w
        return np.convolve(x, kernel, mode='same')
    hp = s - ma(s, long_win)
    lp = ma(hp, short_win)
    return lp

def chrom_signal(R, G, B):
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    sx = X.std() or 1.
    sy = Y.std() or 1.
    return X - (sx/sy)*Y

def pos_signal(R, G, B):
    rm = R.mean() or 1e-8
    gm = G.mean() or 1e-8
    bm = B.mean() or 1e-8
    S1 = R/rm - G/gm
    S2 = R/rm + G/gm - 2*B/bm
    s1 = S1.std() or 1.
    s2 = S2.std() or 1.
    return S1 + (s1/s2)*S2

def fft_bpm(s, fps):
    n = len(s)
    if n < 8:
        return None, float('-inf')
    fv = np.abs(np.fft.rfft(s)) ** 2
    fr = np.fft.rfftfreq(n, 1.0/fps)
    mask = (fr >= BPM_LOW/60.0) & (fr <= BPM_HIGH/60.0)
    if not mask.any():
        return None, float('-inf')
    peak_hz = fr[mask][np.argmax(fv[mask])]
    signal_power = fv[mask].sum()
    noise_power  = fv[~mask].sum()
    snr = signal_power / noise_power if noise_power > 1e-12 else float('inf')
    return float(peak_hz * 60), float(snr)

def peaks_bpm(s, fps):
    if len(s) < 10:
        return None
    threshold = s.mean() + 0.3 * s.std()
    min_dist  = max(int(fps * 60 / BPM_HIGH), 1)
    peaks = []
    i = 1
    while i < len(s) - 1:
        if s[i] > threshold and s[i] >= s[i-1] and s[i] >= s[i+1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
        i += 1
    if len(peaks) < 3:
        return None
    intervals = np.diff(peaks) / fps
    ibi = float(np.median(intervals))
    return float(60 / ibi) if ibi > 0 else None

def bpm_color(bpm):
    if 55 <= bpm <= 100: return [0, 230, 60]
    if 45 <= bpm <= 120: return [0, 200, 255]
    return [0, 80, 255]

def decode_frame_rgb(b64_data):
    raw = base64.b64decode(b64_data)
    import cv2
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("decode failed")
    H, W = frame.shape[:2]
    small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    f = small[0:sh//4, sw//5:4*sw//5]
    l = small[sh*3//10:sh*13//20, 0:sw//5]
    r = small[sh*3//10:sh*13//20, 4*sw//5:]
    regions = [x for x in [f, l, r] if x.size > 0]
    if not regions:
        raise ValueError("no roi")
    R = float(np.mean([reg[:,:,2].mean() for reg in regions]))
    G = float(np.mean([reg[:,:,1].mean() for reg in regions]))
    B = float(np.mean([reg[:,:,0].mean() for reg in regions]))
    gray = small[:,:,1].astype(np.float32)
    return R, G, B, gray, (H, W)

def _process(b64_data, sid, sess):
    now = time.time()
    out = dict(
        session_id=sid, face_found=False, landmarks=[],
        rois={}, bpm=sess["bpm_str"], bpm_color=sess["bpm_col"],
        motion=False, any_covered=False, pose="forward",
        best_algo=sess["best_algo"], snr_vals=sess["snr_vals"],
        wave=list(sess["wave_buf"]), buffer_pct=0,
        fps=round(sess["fps_est"]), frame_w=640, frame_h=480,
    )
    def reset():
        sess["bpm_str"]="--"; sess["best_algo"]="--"
        sess["bpm_col"]=[200,200,200]; sess["snr_vals"]={}
        for q in ("bpm_history","wave_buf","time_buf","r_buf","g_buf","b_buf"):
            sess[q].clear()
        sess["prev_pixel"]=None
    try:
        R, G, B, gray_small, (fH, fW) = decode_frame_rgb(b64_data)
        out["frame_w"] = fW; out["frame_h"] = fH
    except Exception:
        reset()
        return {**out, "bpm":"--", "bpm_color":[200,200,200]}
    motion = False
    if sess["prev_pixel"] is not None:
        diff = np.abs(gray_small - sess["prev_pixel"])
        motion = float(diff.mean()) > MOTION_THRESH
    sess["prev_pixel"] = gray_small
    out["face_found"] = True
    out["motion"] = motion
    out["rois"] = {
        "forehead":    {"poly":[[0.17,0.0],[0.83,0.0],[0.83,0.20],[0.17,0.20]],"covered":False,"color":[0,220,255]},
        "left_cheek":  {"poly":[[0.0,0.45],[0.25,0.45],[0.25,0.70],[0.0,0.70]],"covered":False,"color":[0,255,80]},
        "right_cheek": {"poly":[[0.75,0.45],[1.0,0.45],[1.0,0.70],[0.75,0.70]],"covered":False,"color":[0,255,80]},
    }
    if not motion:
        sess["r_buf"].append(R)
        sess["g_buf"].append(G)
        sess["b_buf"].append(B)
        sess["time_buf"].append(now)
        n = len(sess["time_buf"])
        if n > 10:
            el = sess["time_buf"][-1] - sess["time_buf"][-10]
            if el > 0:
                sess["fps_est"] = 9.0 / el
    n = len(sess["time_buf"])
    need = max(int(sess["fps_est"] * 8), 30)
    buf_pct = min(int(n / need * 100), 100)
    out["buffer_pct"] = buf_pct
    if (now - sess["last_update"]) >= UPDATE_EVERY and n >= need:
        sess["last_update"] = now
        R_arr = np.array(list(sess["r_buf"])[-n:])
        G_arr = np.array(list(sess["g_buf"])[-n:])
        B_arr = np.array(list(sess["b_buf"])[-n:])
        Rn = _detrend(_norm(R_arr))
        Gn = _detrend(_norm(G_arr))
        Bn = _detrend(_norm(B_arr))
        fps = sess["fps_est"]
        cands = {
            "CHROM": butter_bp_simple(chrom_signal(Rn, Gn, Bn), fps),
            "POS":   butter_bp_simple(pos_signal(Rn, Gn, Bn),   fps),
            "GREEN": butter_bp_simple(Gn, fps),
        }
        best_snr = float('-inf')
        best_name = "GREEN"
        snr_vals = {}
        for name, sig in cands.items():
            _, snr = fft_bpm(sig, fps)
            snr_vals[name] = round(snr, 1)
            if snr > best_snr:
                best_snr = snr
                best_name = name
        final = cands[best_name]
        sess["wave_buf"].clear()
        for v in final[-120:]:
            sess["wave_buf"].append(float(v))
        bpm_fft, _ = fft_bpm(final, fps)
        bpm_pk = peaks_bpm(final, fps)
        valid = [x for x in [bpm_fft, bpm_pk] if x and BPM_LOW <= x <= BPM_HIGH]
        bpm = None
        if len(valid) == 2 and abs(valid[0]-valid[1]) < 10:
            bpm = float(np.mean(valid))
        elif valid:
            bpm = valid[0]
        if bpm:
            hist = sess["bpm_history"]
            if not hist or abs(bpm - np.mean(list(hist))) < MAX_BPM_JUMP:
                hist.append(bpm)
            sm = float(np.mean(list(hist)))
            sess["bpm_str"] = f"{sm:.0f}"
            sess["bpm_col"] = bpm_color(sm)
        sess["best_algo"] = best_name
        sess["snr_vals"]  = snr_vals
    return {**out,
        "bpm":        sess["bpm_str"],
        "bpm_color":  sess["bpm_col"],
        "best_algo":  sess["best_algo"],
        "snr_vals":   sess["snr_vals"],
        "wave":       list(sess["wave_buf"]),
        "buffer_pct": buf_pct,
        "fps":        round(sess["fps_est"]),
    }

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
    b64 = raw.split(",")[1] if "," in raw else raw
    sess = _get_session(sid)
    with sess["lock"]:
        return jsonify(_process(b64, sid, sess))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
