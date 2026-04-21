import os, cv2, uuid, time, threading, base64
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.linalg import eigh

app = Flask(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BUFFER_SEC     = 30
TARGET_FPS     = 15        # expected client send rate
BPM_LOW        = 65
BPM_HIGH       = 120
WINDOW_SEC     = 15
UPDATE_EVERY   = 2.0
MOTION_THRESH  = 8.0
BPM_SMOOTH_N   = 10
MAX_BPM_JUMP   = 8
COVER_DROP_PCT = 0.30
SESSION_TTL    = 300       # 5 min idle → evict

REGIONS = {
    "Glabella": [108, 151, 337, 336, 285, 8, 55, 107],
    "cheek1":   [116, 117, 118, 119, 120, 100, 142, 36, 50, 123],
    "cheek2":   [349, 348, 347, 346, 345, 352, 280, 266, 371, 329],
    "temple1":  [54, 68, 70, 156, 143, 34, 127, 162, 21],
    "temple2":  [298, 284, 251, 389, 356, 264, 372, 383, 300],
}
COLORS = {
    "Glabella": [0, 220, 255],
    "cheek1":   [0, 255, 80],
    "cheek2":   [0, 255, 80],
    "temple1":  [255, 100, 0],
    "temple2":  [255, 100, 0],
}
SKIP_ON_POSE = {
    "left":  {"cheek2", "temple2"},
    "right": {"cheek1", "temple1"},
}

# ── Session store ──────────────────────────────────────────────────────────────
_sessions: dict = {}
_store_lock = threading.Lock()


def _new_session():
    max_buf = int(TARGET_FPS * BUFFER_SEC)
    win_buf = int(TARGET_FPS * WINDOW_SEC)
    return {
        "face_mesh":   mp.solutions.face_mesh.FaceMesh(
                           static_image_mode=False, max_num_faces=1,
                           refine_landmarks=False,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5),
        "lock":        threading.Lock(),
        "rgb_bufs":    {k: deque(maxlen=max_buf) for k in REGIONS},
        "time_buf":    deque(maxlen=max_buf),
        "baselines":   {k: deque(maxlen=90) for k in REGIONS},
        "fps_est":     float(TARGET_FPS),
        "bpm_str":     "--",
        "bpm_col":     [200, 200, 200],
        "best_algo":   "--",
        "snr_vals":    {},
        "bpm_history": deque(maxlen=BPM_SMOOTH_N),
        "last_update": 0.0,
        "prev_gray":   None,
        "wave_buf":    deque(maxlen=300),
        "win_buf":     win_buf,
        "last_seen":   time.time(),
    }


def _get_session(sid: str):
    with _store_lock:
        now = time.time()
        # Evict stale sessions
        for k in [k for k, v in _sessions.items() if now - v["last_seen"] > SESSION_TTL]:
            try: _sessions[k]["face_mesh"].close()
            except Exception: pass
            del _sessions[k]
        if sid not in _sessions:
            _sessions[sid] = _new_session()
        _sessions[sid]["last_seen"] = time.time()
        return _sessions[sid]


# ── Signal processing ──────────────────────────────────────────────────────────

def get_head_pose(lm, w, h):
    nose_x  = lm[1].x   * w
    left_x  = lm[234].x * w
    right_x = lm[454].x * w
    face_w  = right_x - left_x
    if face_w < 1: return "forward"
    ratio = (nose_x - left_x) / face_w
    if   ratio < 0.44: return "right"
    elif ratio > 0.56: return "left"
    return "forward"


def lm_to_pixels(lm, w, h):
    return np.array([
        [int(min(max(p.x*w, 0), w-1)), int(min(max(p.y*h, 0), h-1))]
        for p in lm], dtype=np.int32)


def mean_rgb_mask(frame, mask):
    ys, xs = np.where(mask == 255)
    if not len(xs): return None
    b = frame[ys, xs, 0].astype(float)
    g = frame[ys, xs, 1].astype(float)
    r = frame[ys, xs, 2].astype(float)
    bright = (r+g+b)/3
    return r.mean(), g.mean(), b.mean(), bright.mean()


def extract_rois(frame, pts, baselines, skip):
    h, w = frame.shape[:2]
    result, any_cov = {}, False
    for name, idxs in REGIONS.items():
        if name in skip: continue
        valid = [i for i in idxs if 0 <= i < len(pts)]
        if len(valid) < 3: continue
        poly = pts[np.array(valid)]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        vals = mean_rgb_mask(frame, mask)
        if vals is None: continue
        r, g, b, bright = vals
        covered = False
        if len(baselines[name]) >= 10:
            base = float(np.mean(baselines[name]))
            if base > 0 and (base - bright) / base > COVER_DROP_PCT:
                covered = True
        if not covered: baselines[name].append(bright)
        if covered: any_cov = True
        result[name] = {"rgb": (r, g, b), "poly": poly.tolist(), "covered": covered}
    return result, any_cov


def _norm(s):
    std = s.std()
    return (s - s.mean()) / std if std > 1e-8 else s - s.mean()

def _detrend(s):
    t = np.arange(len(s))
    return s - np.polyval(np.polyfit(t, s, 1), t)

def _bp(s, fps):
    if len(s) < 15: return s
    nyq  = fps / 2.0
    lo   = float(np.clip(BPM_LOW /60/nyq, 1e-4, 0.99))
    hi   = float(np.clip(BPM_HIGH/60/nyq, 1e-4, 0.99))
    if lo >= hi: return s
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, s)

def chrom(rgb):
    R,G,B = rgb[:,0], rgb[:,1], rgb[:,2]
    X,Y   = 3*R-2*G, 1.5*R+G-1.5*B
    sx,sy = X.std() or 1., Y.std() or 1.
    return X - (sx/sy)*Y

def pos(rgb):
    m = rgb.mean(axis=0); m[m==0] = 1e-8
    C = rgb/m
    S1,S2 = C[:,0]-C[:,1], C[:,0]+C[:,1]-2*C[:,2]
    s1,s2 = S1.std() or 1., S2.std() or 1.
    return S1 + (s1/s2)*S2

def ica(rgb):
    X = rgb - rgb.mean(axis=0)
    _, vecs = eigh(np.cov(X.T))
    return (X @ vecs)[:, 1]

def snr(s, fps):
    if len(s) < 8: return float("-inf")
    fv = np.abs(np.fft.rfft(s))**2
    fr = np.fft.rfftfreq(len(s), 1/fps)
    mask = (fr >= BPM_LOW/60) & (fr <= BPM_HIGH/60)
    if not mask.any() or mask.all(): return float("-inf")
    n = fv[~mask].sum()
    return float(fv[mask].sum()/n) if n > 1e-12 else float("inf")

def bpm_welch(s, fps):
    nperseg    = min(len(s), int(fps*8))
    fr, psd    = welch(s, fps, nperseg=nperseg)
    mask       = (fr >= BPM_LOW/60) & (fr <= BPM_HIGH/60)
    if not mask.any(): return None
    pf  = fr[mask][np.argmax(psd[mask])]
    bpm = pf * 60
    hf  = pf / 2
    if hf*60 >= BPM_LOW:
        hm = np.abs(fr-hf)<0.04; pm = np.abs(fr-pf)<0.04
        if hm.any() and pm.any() and psd[hm].max() > 0.25*psd[pm].max():
            bpm = hf*60
    return float(bpm)

def bpm_peaks(s, fps):
    md  = max(int(fps*60/BPM_HIGH), 1)
    pro = 0.15*(s.max()-s.min())
    pk, _ = find_peaks(s, distance=md, prominence=pro)
    if len(pk) < 3: return None
    ibi = np.median(np.diff(pk)/fps)
    return float(60/ibi) if ibi > 0 else None

def estimate_bpm(s, fps):
    a = bpm_welch(s, fps); b = bpm_peaks(s, fps)
    valid = [x for x in [a, b] if x and BPM_LOW <= x <= BPM_HIGH]
    if not valid: return None
    if len(valid)==2 and abs(valid[0]-valid[1]) < 8: return float(np.mean(valid))
    return b or a

def bpm_color(bpm):
    if 55 <= bpm <= 100: return [0, 230, 60]
    if 45 <= bpm <= 120: return [0, 200, 255]
    return [0, 80, 255]


# ── Core frame processor ───────────────────────────────────────────────────────

def _process(frame, sid, sess):
    now  = time.time()
    H, W = frame.shape[:2]

    rgb_in  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result  = sess["face_mesh"].process(rgb_in)

    out = dict(
        session_id=sid, face_found=False, landmarks=[],
        rois={}, bpm=sess["bpm_str"], bpm_color=sess["bpm_col"],
        motion=False, any_covered=False, pose="forward",
        best_algo=sess["best_algo"], snr_vals=sess["snr_vals"],
        wave=list(sess["wave_buf"]), buffer_pct=0,
        fps=round(sess["fps_est"]), frame_w=W, frame_h=H,
    )

    def reset():
        for k in ("bpm_str","best_algo"): sess[k] = "--"
        sess["bpm_col"]=[200,200,200]; sess["snr_vals"]={}
        for q in ("bpm_history","wave_buf","time_buf"): sess[q].clear()
        sess["prev_gray"] = None
        for q in sess["rgb_bufs"].values():  q.clear()
        for q in sess["baselines"].values(): q.clear()

    if not result.multi_face_landmarks:
        reset()
        return {**out, "bpm":"--", "bpm_color":[200,200,200]}

    lm   = result.multi_face_landmarks[0].landmark
    pts  = lm_to_pixels(lm, W, H)
    pose = get_head_pose(lm, W, H)
    skip = SKIP_ON_POSE.get(pose, set())
    roi_data, any_cov = extract_rois(frame, pts, sess["baselines"], skip)

    # Motion check
    motion = False
    if sess["prev_gray"] is not None:
        diff   = np.abs(gray.astype(np.float32) - sess["prev_gray"].astype(np.float32))
        motion = float(diff.mean()) > MOTION_THRESH
    sess["prev_gray"] = gray.copy()

    # Buffer signal
    if not motion and not any_cov:
        for name, d in roi_data.items():
            sess["rgb_bufs"][name].append(d["rgb"])
        sess["time_buf"].append(now)
        n = len(sess["time_buf"])
        if n > 10:
            el = sess["time_buf"][-1] - sess["time_buf"][-10]
            if el > 0: sess["fps_est"] = 9.0/el

    if any_cov:
        sess["bpm_str"]="--"; sess["bpm_col"]=[200,200,200]
        sess["bpm_history"].clear(); sess["wave_buf"].clear()
        sess["time_buf"].clear()
        for q in sess["rgb_bufs"].values(): q.clear()

    n       = len(sess["time_buf"])
    win_buf = sess["win_buf"]
    need    = max(int(sess["fps_est"]*10), 30)
    buf_pct = min(int(n/need*100), 100)

    # BPM estimation
    if (now - sess["last_update"]) >= UPDATE_EVERY and n >= need:
        sess["last_update"] = now
        win = min(win_buf, n)
        R=G=B=np.zeros(win); cnt=0
        for buf in sess["rgb_bufs"].values():
            if len(buf) >= win:
                a = np.array(list(buf)[-win:])
                R+=a[:,0]; G+=a[:,1]; B+=a[:,2]; cnt+=1
        if cnt > 0:
            R/=cnt; G/=cnt; B/=cnt
            rgb_n = np.stack([_detrend(_norm(R)), _detrend(_norm(G)), _detrend(_norm(B))], axis=1)
            cands = {
                "CHROM": _bp(chrom(rgb_n), sess["fps_est"]),
                "POS":   _bp(pos(rgb_n),   sess["fps_est"]),
                "ICA":   _bp(ica(rgb_n),   sess["fps_est"]),
            }
            snrs    = {k: snr(v, sess["fps_est"]) for k,v in cands.items()}
            best    = max(snrs, key=snrs.get)
            final   = _bp(cands[best], sess["fps_est"])
            sess["wave_buf"].clear()
            for v in final[-200:]: sess["wave_buf"].append(float(v))
            bpm = estimate_bpm(final, sess["fps_est"])
            if bpm:
                hist = sess["bpm_history"]
                if not hist or abs(bpm - np.mean(hist)) < MAX_BPM_JUMP:
                    hist.append(bpm)
                sm = float(np.mean(hist))
                sess["bpm_str"] = f"{sm:.0f}"; sess["bpm_col"] = bpm_color(sm)
            sess["best_algo"] = best
            sess["snr_vals"]  = {k: round(float(v),1) for k,v in snrs.items()}

    rois_out = {
        name: {
            "poly":    [[p[0]/W, p[1]/H] for p in d["poly"]],
            "covered": d["covered"],
            "color":   [0,0,200] if d["covered"] else COLORS[name],
        } for name, d in roi_data.items()
    }

    return {**out,
        "face_found":  True,
        "landmarks":   [[lm_p.x, lm_p.y] for lm_p in lm],
        "rois":        rois_out,
        "bpm":         sess["bpm_str"],
        "bpm_color":   sess["bpm_col"],
        "motion":      motion,
        "any_covered": any_cov,
        "pose":        pose,
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
