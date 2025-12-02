from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
import os

# --- Section 1: Lightweight Kalman filter tracker for the ball ---
class BallKF:
    def __init__(self, dt=1.0):
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1,  0],
                                             [0, 0, 0,  1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        # Tunable noise covariances (trust detections more; keep model flexible)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e-1
        self.initialized = False

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, cx, cy):
        m = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
        self.kf.correct(m)

    def state(self):
        s = self.kf.statePost
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])

# Global variables for script mode (will be replaced by class when used programmatically)
model = None
video_name = None
video_path = None
cap = None
output_path = None
frame_width = None
frame_height = None
fps = None
out = None
dt = None
kf = None

# --- Homography from 4 stump-bottom points → top-down pitch ---
L_PITCH_M = 20.12  # length in meters
WICKET_WIDTH_M = 0.2286  # meters (≈ 9 inches across outer stumps)
W_PITCH_M = 3.05   # width in meters

# Initialize global variables (will be set in script mode or by BallAnalyzer)
CALIB_JSON = None
CALIB_OK = False
H = None
pitch_quad_px = None



def px_to_ground_m(x, y):
    """Map image pixel (x,y) → (X across, Y along) meters on pitch. Returns None if not calibrated."""
    if not CALIB_OK or H is None:
        return None
    p = np.array([[[float(x), float(y)]]], dtype=np.float32)
    g = cv2.perspectiveTransform(p, H)[0, 0]
    return float(g[0]), float(g[1])

# --- Project a pixel to the ground quad before homography (fixes z sticking at 0/20) ---

def px_to_ground_via_foot(x, y):
    """Project image point (x,y) onto the pitch quadrilateral, then map to meters via homography.
    This approximates the ground-foot of the airborne ball.
    Returns (X_m, Y_m) or None if calibration missing.
    """
    if not CALIB_OK or H is None or pitch_quad_px is None or len(pitch_quad_px) != 4:
        return None
    P = np.array([float(x), float(y)], dtype=float)

    # Quad corners (bowBL, bowBR, batBR, batBL) as float vectors
    bow_bl, bow_br, bat_br, bat_bl = pitch_quad_px.astype(float)

    # Centerline mids at bowler/batsman ends
    M_bow = 0.5 * (bow_bl + bow_br)
    M_bat = 0.5 * (bat_bl + bat_br)

    # Parametric position along pitch axis (t in [0,1]) via projection onto centerline
    axis = M_bat - M_bow
    denom = float(np.dot(axis, axis))
    if denom < 1e-9:
        return None
    t = float(np.dot(P - M_bow, axis) / denom)
    t = max(0.0, min(1.0, t))

    # Interpolate current left/right edge points at this along-pitch t
    L_t = bow_bl + t * (bat_bl - bow_bl)
    R_t = bow_br + t * (bat_br - bow_br)

    edge = R_t - L_t
    edge_denom = float(np.dot(edge, edge))
    if edge_denom < 1e-9:
        return None

    # Project P onto the across edge segment to get its ground foot (u in [0,1])
    u = float(np.dot(P - L_t, edge) / edge_denom)
    u = max(0.0, min(1.0, u))
    foot = L_t + u * edge

    # Now map this ground foot through the homography
    p = np.array([[[float(foot[0]), float(foot[1])]]], dtype=np.float32)
    g = cv2.perspectiveTransform(p, H)[0, 0]
    return float(g[0]), float(g[1])

# --- Fusion controls ---
USE_MEASUREMENT_WHEN_AVAILABLE = True  # draw raw detection when we have it
CONF_GATE = 0.40                       # only correct KF if conf >= this
# Bootstrap velocity for KF using first two detections
last_det = None
last_det_frame = None
_kf_reset_done_for_bounce = False

# --- Section 2: Release & Bounce detection ---
# Thresholds are in pixels per frame (ppf). Tune for your footage/FPS.
MIN_SPEED_PX_PER_F = 3.0      # when crossed the first time => release
SPEED_SMOOTH_N = 3            # moving average window to reduce jitter
BOUNCE_MIN_GAP_FRAMES = 10     # must be this many frames after release to count bounce
VY_MIN_MAG = 0.5              # ignore tiny sign flips around 0
# Kalman handling around bounce
KF_FREEZE_AFTER_BOUNCE_FRAMES = 3   # use raw detections for N frames after bounce
KF_RESET_ON_BOUNCE = True           # drop and re-bootstrap the KF after the freeze window

# Realtime position overlay
SHOW_REALTIME_POS = True

# Contact detection (post-bounce) thresholds
CONTACT_MIN_Z_M = 17.5         # consider contact only near batsman end (meters along pitch)
CONTACT_ANGLE_DEG = 40.0       # deflection angle threshold between velocity vectors
CONTACT_SPEED_DROP = 0.50      # drop ratio: |v_after| <= CONTACT_SPEED_DROP * |v_before|
CONTACT_MIN_GAP_FRAMES = 6     # wait a few frames after bounce before checking contact
STOP_ON_CONTACT = False         # end processing once contact is detected

# Z-axis anchor targets (meters)
RELEASE_TARGET_Z_M = 0.0
CONTACT_TARGET_Z_M = 17.5
Z_MIN_SPAN_M = 3.0   # require at least this much measured span between release and contact to trust mapping

# State
release_idx = None
bounce_idx = None
contact_idx = None

release_pt = None
bounce_pt = None
contact_pt = None

# Affine mapping for z (Y along pitch)
z_affine_ok = False
zA, zB = 1.0, 0.0  # z_refined = zA * z_meas + zB

prev_est = None
vy_hist = []                  # store recent vy to smooth sign flips

# Live metrics (computed when homography is available)
release_speed_kmph_val = None
length_m_val = None
line_m_val = None

manual_id = 0

track_history = defaultdict(lambda: [])
trajectory_data = []
# Cache of Kalman-filtered positions per frame for analytics
filtered_by_frame = {}

# For JSON export of ground-projected positions
ball_positions = []  # to collect positions for JSON export

# --- Helpers to access trajectory by frame index (since we don't push every frame) ---

def traj_rows_between(start_frame, end_frame):
    """Return list of rows (x, y, frame_idx, t_s) whose frame index lies within [start_frame, end_frame]."""
    return [row for row in trajectory_data if start_frame <= row[2] <= end_frame]


def traj_row_at(frame_index):
    """Return row with exact frame_index if present; otherwise the nearest by abs(frame_idx - target)."""
    if not trajectory_data:
        return None
    exact = [row for row in trajectory_data if row[2] == frame_index]
    if exact:
        return exact[0]
    # nearest
    return min(trajectory_data, key=lambda r: abs(r[2] - frame_index))


# --- Z mapping helpers ---
def _compute_z_affine_mapping():
    """Compute affine mapping for along-pitch z using release/contact anchors.
    Sets globals: z_affine_ok, zA, zB. Returns True if mapping is valid.
    """
    global z_affine_ok, zA, zB
    if release_idx is None or contact_idx is None:
        return False
    row_r = traj_row_at(release_idx)
    row_c = traj_row_at(contact_idx)
    if row_r is None or row_c is None:
        return False
    xr, yr, _, _ = row_r
    xc, yc, _, _ = row_c
    gm_r = px_to_ground_via_foot(xr, yr)
    gm_c = px_to_ground_via_foot(xc, yc)
    if gm_r is None or gm_c is None:
        return False
    _, zr_hat = gm_r
    _, zc_hat = gm_c
    # Require usable span
    if abs(zc_hat - zr_hat) < 1e-6:
        return False
    # Build affine map so that z(release)=0, z(contact)=CONTACT_TARGET_Z_M
    zA = (CONTACT_TARGET_Z_M - RELEASE_TARGET_Z_M) / (zc_hat - zr_hat)
    zB = RELEASE_TARGET_Z_M - zA * zr_hat
    # Basic sanity: measured span should be at least Z_MIN_SPAN_M after mapping
    span = zA * (zc_hat - zr_hat)
    z_affine_ok = abs(span) >= Z_MIN_SPAN_M * 0.5
    if z_affine_ok:
        print(f"[zmap] z = {zA:.4f} * z_hat + {zB:.4f} (span~{span:.2f} m)")
    else:
        print("[zmap] WARNING: z span too small; not applying mapping")
    return z_affine_ok


def _map_z(z_meas):
    """Apply affine mapping if available; else return input."""
    return float(zA * z_meas + zB) if z_affine_ok else float(z_meas)

def _angle_deg(v1x, v1y, v2x, v2y):
    a = np.array([v1x, v1y], dtype=float)
    b = np.array([v2x, v2y], dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def _rows_to_ground(rows, fps):
    """Map rows [(x,y,fidx,t)] to [(fidx, t_s, Xg, Yg)] using px_to_ground_via_foot, dropping Nones."""
    out = []
    for xpx, ypx, fidx, ts in rows:
        gm = px_to_ground_via_foot(xpx, ypx)
        if gm is None:
            continue
        Xg, Yg = gm
        Yg = _map_z(Yg)
        out.append((int(fidx), float(ts), float(Xg), float(Yg)))
    # sort by frame idx for monotonicity
    out.sort(key=lambda r: r[0])
    return out

# --- Robust speed estimator using KF-filtered points and trimmed median ---

def robust_speed_kmph(window_rows, fps):
    """window_rows: list of (x_px, y_px, frame_idx, t_s) sorted by frame_idx.
    Returns horizontal (along-pitch) speed in km/h using ground Y meters.
    Uses KF-filtered positions when available.
    """
    if not window_rows or len(window_rows) < 2 or fps <= 0:
        return None
    rows = sorted(window_rows, key=lambda r: r[2])

    # Build ground Y series using filtered positions when possible
    series = []  # list of (frame_idx, Y_m)
    for xpx, ypx, fidx, ts in rows:
        fx_fy = filtered_by_frame.get(fidx)
        if fx_fy is not None:
            px, py = fx_fy
        else:
            px, py = xpx, ypx
        gm = px_to_ground_via_foot(px, py)
        if gm is None:
            continue
        _, Yg = gm
        Yg = _map_z(Yg)
        series.append((fidx, float(Yg)))
    if len(series) < 2:
        return None

    # Determine dominant direction (sign of median delta)
    deltas = [series[i+1][1] - series[i][1] for i in range(len(series)-1)]
    if not deltas:
        return None
    med_delta = float(np.median(deltas))
    direction = 1.0 if med_delta >= 0 else -1.0

    # Compute per-step speeds (m/s) using frame gaps; keep only steps matching direction
    step_speeds = []
    Y0 = series[0][1]
    YN = series[-1][1]
    total_travel = abs(YN - Y0)
    for i in range(len(series)-1):
        f0, y0 = series[i]
        f1, y1 = series[i+1]
        df = f1 - f0
        if df <= 0:
            continue
        vy = (y1 - y0) / (df / fps)  # m/s
        if vy * direction <= 0:
            continue  # discard sign-inconsistent steps
        # Discard obviously impossible steps > 65 m/s (~234 km/h)
        if abs(vy) > 65:
            continue
        step_speeds.append(abs(vy))

    if not step_speeds:
        return None

    # Require a minimum total travel to reduce noise (e.g., at least 0.8 m)
    if total_travel < 0.8:
        return None

    # Trim extremes (10% on each side) and take median
    step_speeds_sorted = sorted(step_speeds)
    n = len(step_speeds_sorted)
    k = max(0, int(0.1 * n))
    trimmed = step_speeds_sorted[k: n - k] if n - 2*k >= 1 else step_speeds_sorted
    mps_med = float(np.median(trimmed))
    return mps_med * 3.6

if __name__ == "__main__":
    # Script mode: Load model and video from test directory
    model = YOLO("runs/detect/train/weights/best.pt")
    video_name = "test-video2"
    video_path = f"test/input/{video_name}.mp4"
    cap = cv2.VideoCapture(video_path)

    # Output settings
    output_path = f"test/output/{video_name}.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    dt = 1.0 / max(fps, 1.0)
    kf = BallKF(dt=dt)

    # Load calibration for script mode
    CALIB_JSON = "calibration_stumps.json"
    try:
        with open(CALIB_JSON, "r") as f:
            _cal = json.load(f)
        pts = _cal["image_points"]
        bow_bl = pts["bowler_bottom_left"]
        bow_br = pts["bowler_bottom_right"]
        bat_bl = pts["batsman_bottom_left"]
        bat_br = pts["batsman_bottom_right"]

        src = np.array([bow_bl, bow_br, bat_br, bat_bl], dtype=np.float32)
        dst = np.array([[0.0, 0.0],
                        [WICKET_WIDTH_M, 0.0],
                        [WICKET_WIDTH_M, L_PITCH_M],
                        [0.0, L_PITCH_M]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        pitch_quad_px = src.astype(np.int32)
        CALIB_OK = True
        print("[calib] 4-point homography loaded for script mode.")
    except Exception as e:
        print(f"[calib] WARNING: could not build homography from {CALIB_JSON}: {e}")

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

            # Robust timestamp for this frame (fallback to frame_idx/fps if unavailable)
            t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_s = float(t_msec) / 1000.0 if t_msec and t_msec > 0 else (frame_idx / max(fps, 1.0))
        
            # Detect ball (YOLO) + fuse with Kalman filter
            results = model.track(frame, persist=True, conf=0.5)
            res = results[0]
            boxes = res.boxes.xywh.cpu() if res.boxes is not None else []
            confs = res.boxes.conf.cpu() if res.boxes is not None else []
        
            annotated_frame = res.plot() if res is not None else frame.copy()
        
            # Draw proper pitch overlay (homography quad) if available
            if CALIB_OK and pitch_quad_px is not None and len(pitch_quad_px) == 4:
                overlay = annotated_frame.copy()
                cv2.polylines(overlay, [pitch_quad_px.reshape((-1, 1, 2))], isClosed=True, color=(255, 255, 255), thickness=3)
                cv2.fillPoly(overlay, [pitch_quad_px.reshape((-1, 1, 2))], color=(255, 255, 255))
                cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)
        
            # Extract best detection (if any) and its confidence
            cx = cy = None
            best_conf = None
            if boxes is not None and len(boxes) > 0:
                best_idx = confs.argmax()
                x, y, w, h = boxes[best_idx]
                cx, cy = float(x), float(y)
                best_conf = float(confs[best_idx])
                # Draw detection box and label (explicit)
                x1, y1 = int(cx - float(w) / 2), int(cy - float(h) / 2)
                x2, y2 = int(cx + float(w) / 2), int(cy + float(h) / 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Ball {best_conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
            # Are we within the post-bounce KF freeze window?
            in_kf_freeze = (bounce_idx is not None) and ((frame_idx - bounce_idx) <= KF_FREEZE_AFTER_BOUNCE_FRAMES)
            # Adaptive bootstrap + gated fusion (with bounce freeze)
            est_x = est_y = None
        
            if in_kf_freeze:
                # During freeze: do NOT use KF predict/correct; use raw detection if available
                if cx is not None and cy is not None:
                    est_x, est_y = cx, cy
                    # For analytics, store the measurement so speed/metrics use it
                    filtered_by_frame[frame_idx] = (float(cx), float(cy))
                # Mark that we still need to reset KF after freeze ends
                if KF_RESET_ON_BOUNCE:
                    _kf_reset_done_for_bounce = False
            else:
                # Optionally reset/rebootstrap the KF the first frame after freeze ends
                if KF_RESET_ON_BOUNCE and bounce_idx is not None and not _kf_reset_done_for_bounce and (frame_idx - bounce_idx) == (KF_FREEZE_AFTER_BOUNCE_FRAMES + 1):
                    # Reset KF but keep trajectory continuity; seed bootstrap with current detection if available
                    kf.initialized = False
                    if cx is not None and cy is not None:
                        last_det = (cx, cy)
                        last_det_frame = frame_idx
                        # Ensure we keep drawing and recording using the measurement on this frame
                        est_x, est_y = cx, cy
                        filtered_by_frame[frame_idx] = (float(cx), float(cy))
                    _kf_reset_done_for_bounce = True
        
                if not kf.initialized:
                    if cx is not None and cy is not None:
                        if last_det is None:
                            # First-ever detection: store and draw raw; wait for second to bootstrap velocity
                            last_det = (cx, cy)
                            last_det_frame = frame_idx
                            est_x, est_y = cx, cy
                            filtered_by_frame[frame_idx] = None
                        else:
                            # Second detection: initialize KF with velocity from two detections
                            dt_frames = max(1, frame_idx - (last_det_frame or frame_idx))
                            vx = (cx - last_det[0]) / dt_frames
                            vy = (cy - last_det[1]) / dt_frames
                            kf.kf.statePost = np.array([[cx], [cy], [vx], [vy]], dtype=np.float32)
                            kf.initialized = True
                            # Use the current detection as the first estimate
                            est_x, est_y = cx, cy
                            filtered_by_frame[frame_idx] = None
                else:
                    # Predict first
                    px, py = kf.predict()
                    # Correct only if detection is strong enough
                    if cx is not None and cy is not None and (best_conf is None or best_conf >= CONF_GATE):
                        kf.correct(cx, cy)
                    sx, sy, _, _ = kf.state()
                    # For visualization, prefer the measurement when available to remove KF "lag"
                    if USE_MEASUREMENT_WHEN_AVAILABLE and cx is not None and cy is not None:
                        est_x, est_y = cx, cy
                    else:
                        est_x, est_y = sx, sy
                    # Cache the filtered state for analytics (even if we draw the measurement)
                    filtered_by_frame[frame_idx] = (float(sx), float(sy))
        
            # Draw/store trajectory if we have an estimate
            if est_x is not None and est_y is not None:
                track = track_history[manual_id]
                track.append((est_x, est_y))
                trajectory_data.append((est_x, est_y, frame_idx, t_s))
                # Collect ground-projected positions for JSON
                if CALIB_OK:
                    gm = px_to_ground_via_foot(est_x, est_y)
                    if gm is not None:
                        Xg, Yg = gm
                        Yg = _map_z(Yg)
                        ball_positions.append({
                            "frame": int(frame_idx),
                            "t_s": float(t_s),
                            "x_m": float(Xg),
                            "z_m": float(Yg)
                        })
                # (Optional) keep filtered estimate separately if you want smoother analytics later
                # filtered_x, filtered_y, _, _ = kf.state() if kf.initialized else (est_x, est_y, 0, 0)
                if len(track) > 1:
                    pts = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=False, color=(255, 255, 0), thickness=6)
                cv2.circle(annotated_frame, (int(est_x), int(est_y)), 6, (0, 215, 255), -1)
        
                # Realtime (x,z) meters near the ball
                if CALIB_OK and SHOW_REALTIME_POS:
                    gm = px_to_ground_via_foot(est_x, est_y)
                    if gm is not None:
                        Xg, Yg = gm  # X across wicket (m), Y along pitch (m)
                        # Clamp to physical pitch bounds for display
                        Xg_c = max(0.0, min(WICKET_WIDTH_M, Xg))
                        Yg_m = _map_z(Yg)
                        Yg_c = max(0.0, min(L_PITCH_M, Yg_m))
                        # Show lateral relative to wicket center for readability
                        Xrel = Xg_c - (WICKET_WIDTH_M / 2.0)
                        label = f"x={Xrel:+.2f}m  z={Yg_c:.2f}m"
                        cv2.putText(annotated_frame, label, (int(est_x) + 10, int(est_y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        
                # --- Section 2 logic: compute release & bounce ---
                # Velocity in pixels/frame from current and previous estimates
                if prev_est is not None:
                    vx_ppf = float(est_x - prev_est[0])
                    vy_ppf = float(est_y - prev_est[1])
                    speed_ppf = (vx_ppf**2 + vy_ppf**2) ** 0.5
        
                    # Smooth speed a bit with a short window to avoid triggering on noise
                    vy_hist.append(vy_ppf)
                    if len(vy_hist) > SPEED_SMOOTH_N:
                        vy_hist.pop(0)
                    avg_speed_ppf = speed_ppf
                    if len(vy_hist) >= 2:
                        # simple average over last N speeds approximated using vy history length
                        # (we keep full speed but use vy_hist length as a small, quick buffer)
                        avg_speed_ppf = (avg_speed_ppf + sum(abs(v) for v in vy_hist) / max(1, len(vy_hist))) / 2.0
        
                    # RELEASE: first time speed crosses threshold
                    if release_idx is None and avg_speed_ppf >= MIN_SPEED_PX_PER_F:
                        release_idx = frame_idx
                        release_pt = (int(est_x), int(est_y))
        
                    # BOUNCE: detect a sign flip in vertical velocity with minimum magnitude, and only after a small gap from release
                    if release_idx is not None and bounce_idx is None and (frame_idx - release_idx) >= BOUNCE_MIN_GAP_FRAMES:
                        if len(vy_hist) >= 2:
                            prev_vy = vy_hist[-2]
                            curr_vy = vy_hist[-1]
                            if abs(prev_vy) > VY_MIN_MAG and abs(curr_vy) > VY_MIN_MAG:
                                if (prev_vy > 0 and curr_vy < 0) or (prev_vy < 0 and curr_vy > 0):
                                    bounce_idx = frame_idx
                                    bounce_pt = (int(est_x), int(est_y))
        
                    # CONTACT: after bounce, detect bat contact by deflection or speed drop near batsman end
                    if bounce_idx is not None and contact_idx is None and (frame_idx - bounce_idx) >= CONTACT_MIN_GAP_FRAMES and CALIB_OK:
                        # Consider last 6-8 trajectory rows after bounce
                        post_rows = [row for row in trajectory_data if row[2] > bounce_idx]
                        if len(post_rows) >= 4:
                            # Map to ground and compute recent velocities
                            g = _rows_to_ground(post_rows[-8:], fps)
                            if len(g) >= 4:
                                # Build two vectors: pre (before last), post (last step)
                                f0, t0, X0, Y0 = g[-4]
                                f1, t1, X1, Y1 = g[-3]
                                f2, t2, X2, Y2 = g[-2]
                                f3, t3, X3, Y3 = g[-1]
                                dt_pre = max(1e-6, (f2 - f1) / fps)
                                dt_post = max(1e-6, (f3 - f2) / fps)
                                vpre = ((X2 - X1) / dt_pre, (Y2 - Y1) / dt_pre)
                                vpost = ((X3 - X2) / dt_post, (Y3 - Y2) / dt_post)
                                ang = _angle_deg(vpre[0], vpre[1], vpost[0], vpost[1])
                                sp_pre = (vpre[0]**2 + vpre[1]**2) ** 0.5
                                sp_post = (vpost[0]**2 + vpost[1]**2) ** 0.5
                                near_batsman = (Y3 >= CONTACT_MIN_Z_M)
                                big_deflect = (ang >= CONTACT_ANGLE_DEG)
                                big_slow = (sp_post <= CONTACT_SPEED_DROP * sp_pre)
                                if near_batsman and (big_deflect or big_slow):
                                    contact_idx = frame_idx
                                    contact_pt = (int(est_x), int(est_y))
                                    # Build z-affine mapping once we have release & contact
                                    _compute_z_affine_mapping()
        
                    # Draw markers if we have events
                    if release_pt is not None:
                        cv2.circle(annotated_frame, release_pt, 7, (255, 0, 0), -1)   # blue = release
                        cv2.putText(annotated_frame, "RELEASE", (release_pt[0]+8, release_pt[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    if bounce_pt is not None:
                        cv2.circle(annotated_frame, bounce_pt, 7, (0, 0, 255), -1)   # red = bounce
                        cv2.putText(annotated_frame, "BOUNCE", (bounce_pt[0]+8, bounce_pt[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if contact_pt is not None:
                        cv2.circle(annotated_frame, contact_pt, 7, (0, 165, 255), -1)  # orange = contact
                        cv2.putText(annotated_frame, "CONTACT", (contact_pt[0]+8, contact_pt[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
                    # --- Ground metrics with homography ---
                    if CALIB_OK:
                        # Compute speed from release → bounce/window using ground coords (more permissive)
                        if release_idx is not None:
                            start_f = release_idx
                            end_f = bounce_idx if bounce_idx is not None else (release_idx + 10)
                            window_rows = traj_rows_between(start_f, end_f)
                            if len(window_rows) >= 2:
                                spd = robust_speed_kmph(window_rows, fps)
                                if spd is not None:
                                    release_speed_kmph_val = float(spd)
                        # Fallback: live speed from the last few samples if release not locked or window too small
                        if release_speed_kmph_val is None:
                            rows_sorted = sorted(trajectory_data, key=lambda r: r[2])
                            tail = rows_sorted[-8:] if len(rows_sorted) >= 3 else []
                            if len(tail) >= 2:
                                spd2 = robust_speed_kmph(tail, fps)
                                if spd2 is not None:
                                    release_speed_kmph_val = float(spd2)
        
                        # Compute length & line at bounce once
                        if bounce_idx is not None and length_m_val is None and line_m_val is None:
                            row_b = traj_row_at(bounce_idx)
                            if row_b is not None:
                                bx, by, _, _ = row_b
                                gmb = px_to_ground_via_foot(bx, by)
                                if gmb is not None:
                                    Xb, Yb_hat = gmb
                                    Yb = _map_z(Yb_hat)
                                    length_m_val = max(0.0, min(L_PITCH_M, L_PITCH_M - Yb))
                                    line_m_val = Xb - (WICKET_WIDTH_M / 2.0)
                                    # Optional sanity print for extreme numbers
                                    if (release_speed_kmph_val is not None and (release_speed_kmph_val < 10 or release_speed_kmph_val > 170)):
                                        print(f"[metrics] WARN speed {release_speed_kmph_val:.1f} km/h; check calibration & frame timing")
                                    if abs(line_m_val) > 1.0 or length_m_val < 0 or length_m_val > L_PITCH_M:
                                        print(f"[metrics] WARN line/length out of typical range: line={line_m_val:.2f} m, length={length_m_val:.2f} m")
        
                        # Overlays
                        y_base = 40
                        if z_affine_ok:
                            cv2.putText(annotated_frame, "z-calib: release→0m, contact→17.5m", (20, 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 180), 2)
                        # Show a placeholder while speed is still estimating
                        if release_speed_kmph_val is None and release_idx is not None:
                            cv2.putText(annotated_frame, "Speed ~ … (estimating)", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 80), 2)
                        if release_speed_kmph_val is not None:
                            cv2.putText(annotated_frame, f"Speed ~ {release_speed_kmph_val:.1f} km/h", (20, y_base),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 50), 2)
                            y_base += 36
                        if length_m_val is not None:
                            cv2.putText(annotated_frame, f"Length ~ {length_m_val:.2f} m", (20, y_base),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 200, 255), 2)
                            y_base += 36
                        if line_m_val is not None:
                            cv2.putText(annotated_frame, f"Line ~ {line_m_val:+.2f} m", (20, y_base),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 50), 2)
                    # Optionally end processing once contact is detected
                    if STOP_ON_CONTACT and contact_idx is not None:
                        print(f"[events] Contact detected at frame {contact_idx}. Stopping.")
                        # Write current frame and break after saving
                        out.write(annotated_frame)
                        cv2.imshow("Ball Tracking", annotated_frame)
                        break
        
                prev_est = (est_x, est_y)
        
            # Save frame
            cv2.imshow("Ball Tracking", annotated_frame)
            out.write(annotated_frame)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
                frame_idx += 1

    # Cleanup
    cap.release()
    out.release()

    # Export ball positions to JSON
    session_id = video_name
    out_json_path = f"test/output/{session_id}_positions.json"
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(ball_positions, f, indent=2)
    print(f"[export] Ball positions written to {out_json_path} ({len(ball_positions)} points)")

    cv2.destroyAllWindows()


class BallAnalyzer:
    """Wrapper class for programmatic ball tracking and analysis"""

    def __init__(self, model_path: str):
        """
        Initialize ball analyzer

        Args:
            model_path: Path to YOLO ball detection model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path

    def analyze(self, video_path: str, stump_json_path: str, output_dir: str, file_id: str) -> dict:
        """
        Analyze ball trajectory in video

        Args:
            video_path: Path to input video
            stump_json_path: Path to stump calibration JSON
            output_dir: Directory for output files
            file_id: Unique identifier for this analysis

        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(stump_json_path):
            raise FileNotFoundError(f"Stump JSON not found: {stump_json_path}")

        # Load model
        model = YOLO(self.model_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default fallback

        # Output video path
        output_video_path = os.path.join(output_dir, f"{file_id}_annotated.mp4")
        output_json_path = os.path.join(output_dir, f"{file_id}_results.json")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Initialize Kalman filter
        dt = 1.0 / max(fps, 1.0)
        kf = BallKF(dt=dt)

        # Load calibration
        with open(stump_json_path, "r") as f:
            calib_data = json.load(f)

        pts = calib_data.get("image_points", {})
        bow_bl = pts.get("bowler_bottom_left")
        bow_br = pts.get("bowler_bottom_right")
        bat_bl = pts.get("batsman_bottom_left")
        bat_br = pts.get("batsman_bottom_right")

        CALIB_OK = False
        H = None
        pitch_quad_px = None

        if bow_bl and bow_br and bat_bl and bat_br:
            src = np.array([bow_bl, bow_br, bat_br, bat_bl], dtype=np.float32)
            dst = np.array([[0.0, 0.0],
                            [WICKET_WIDTH_M, 0.0],
                            [WICKET_WIDTH_M, L_PITCH_M],
                            [0.0, L_PITCH_M]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            pitch_quad_px = src.astype(np.int32)
            CALIB_OK = True

        # Analysis state
        release_idx = None
        bounce_idx = None
        release_pt = None
        bounce_pt = None
        prev_est = None
        vy_hist = []
        trajectory_data = []
        ball_positions = []
        filtered_by_frame = {}
        release_speed_kmph_val = None
        length_m_val = None
        line_m_val = None

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_s = float(t_msec) / 1000.0 if t_msec and t_msec > 0 else (frame_idx / max(fps, 1.0))

            # Detect ball with lower confidence threshold
            results = model.track(frame, persist=True, conf=0.25, verbose=False)  # Lowered from 0.5 to 0.25
            res = results[0]
            boxes = res.boxes.xywh.cpu() if res.boxes is not None else []
            confs = res.boxes.conf.cpu() if res.boxes is not None else []

            # Debug logging every 30 frames
            if frame_idx % 30 == 0:
                print(f"[Frame {frame_idx}] Detections: {len(boxes)}, CALIB_OK: {CALIB_OK}")

            annotated_frame = frame.copy()

            # Draw pitch overlay
            if CALIB_OK and pitch_quad_px is not None:
                overlay = annotated_frame.copy()
                cv2.polylines(overlay, [pitch_quad_px.reshape((-1, 1, 2))], isClosed=True, color=(255, 255, 255), thickness=3)
                cv2.fillPoly(overlay, [pitch_quad_px.reshape((-1, 1, 2))], color=(255, 255, 255))
                cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)

            # Extract detection
            cx = cy = None
            best_conf = None
            if boxes is not None and len(boxes) > 0:
                best_idx = confs.argmax()
                x, y, w, h = boxes[best_idx]
                cx, cy = float(x), float(y)
                best_conf = float(confs[best_idx])

                # Draw detection box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Ball {best_conf:.2f}", (x1, max(y1-10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Kalman filtering (simplified for production)
            est_x = est_y = None
            if cx is not None and cy is not None:
                if not kf.initialized:
                    kf.correct(cx, cy)
                    est_x, est_y = cx, cy
                else:
                    kf.predict()
                    kf.correct(cx, cy)
                    sx, sy, _, _ = kf.state()
                    est_x, est_y = cx, cy  # Use measurement for visualization
                    filtered_by_frame[frame_idx] = (float(sx), float(sy))

            # Track trajectory
            if est_x is not None and est_y is not None:
                trajectory_data.append((est_x, est_y, frame_idx, t_s))

                # Draw trajectory line (connect previous points)
                if len(trajectory_data) > 1:
                    pts = np.array([(int(x), int(y)) for x, y, _, _ in trajectory_data[-10:]], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=3)

                # Draw current ball position
                cv2.circle(annotated_frame, (int(est_x), int(est_y)), 8, (0, 215, 255), -1)
                cv2.circle(annotated_frame, (int(est_x), int(est_y)), 10, (255, 255, 255), 2)

                # Event detection (simplified)
                if prev_est is not None:
                    vx_ppf = float(est_x - prev_est[0])
                    vy_ppf = float(est_y - prev_est[1])
                    speed_ppf = (vx_ppf**2 + vy_ppf**2) ** 0.5

                    vy_hist.append(vy_ppf)
                    if len(vy_hist) > 3:
                        vy_hist.pop(0)

                    # Release detection
                    if release_idx is None and speed_ppf >= MIN_SPEED_PX_PER_F:
                        release_idx = frame_idx
                        release_pt = (int(est_x), int(est_y))

                    # Bounce detection
                    if release_idx is not None and bounce_idx is None and (frame_idx - release_idx) >= 10:
                        if len(vy_hist) >= 2:
                            prev_vy = vy_hist[-2]
                            curr_vy = vy_hist[-1]
                            if abs(prev_vy) > 0.5 and abs(curr_vy) > 0.5:
                                if (prev_vy > 0 and curr_vy < 0) or (prev_vy < 0 and curr_vy > 0):
                                    bounce_idx = frame_idx
                                    bounce_pt = (int(est_x), int(est_y))

                # Draw markers
                if release_pt is not None:
                    cv2.circle(annotated_frame, release_pt, 7, (255, 0, 0), -1)
                    cv2.putText(annotated_frame, "RELEASE", (release_pt[0]+8, release_pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if bounce_pt is not None:
                    cv2.circle(annotated_frame, bounce_pt, 7, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "BOUNCE", (bounce_pt[0]+8, bounce_pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Compute metrics
                if CALIB_OK:
                    # Compute speed if we have trajectory data
                    if release_speed_kmph_val is None and len(trajectory_data) >= 5:
                        if release_idx is not None:
                            # Use release to bounce window if both detected
                            if bounce_idx is not None:
                                window = trajectory_data[release_idx:bounce_idx]
                            else:
                                # Otherwise use first 10-15 frames of trajectory
                                window = trajectory_data[release_idx:min(release_idx+15, len(trajectory_data))]
                        else:
                            # Fallback: use any available trajectory window
                            window = trajectory_data[:min(15, len(trajectory_data))]

                        if len(window) >= 2:
                            spd = self._compute_speed(window, fps, H)
                            if spd:
                                release_speed_kmph_val = spd

                    # Compute length and line at bounce
                    if bounce_idx is not None and length_m_val is None and line_m_val is None:
                        bx, by = bounce_pt
                        gm = px_to_ground_via_foot(bx, by) if H is not None else None
                        if gm:
                            Xb, Yb = gm
                            length_m_val = max(0.0, min(L_PITCH_M, L_PITCH_M - Yb))
                            line_m_val = Xb - (WICKET_WIDTH_M / 2.0)

                prev_est = (est_x, est_y)

            # Write annotated frame
            out.write(annotated_frame)
            frame_idx += 1

        # Cleanup
        cap.release()
        out.release()

        # Determine line and length categories
        line_category = "Middle"
        if line_m_val is not None:
            if line_m_val < -0.05:
                line_category = "Leg"
            elif line_m_val > 0.05:
                line_category = "Off"

        length_category = "Good"
        if length_m_val is not None:
            if length_m_val < 4.0:
                length_category = "Yorker"
            elif length_m_val < 8.0:
                length_category = "Full"
            elif length_m_val > 12.0:
                length_category = "Short"

        # Prepare results
        results = {
            "speed": round(release_speed_kmph_val, 1) if release_speed_kmph_val else None,
            "line": line_category,
            "length": length_category,
            "bounce": {
                "x": bounce_pt[0] if bounce_pt else None,
                "y": bounce_pt[1] if bounce_pt else None,
                "angle": 0.0
            },
            "trajectory": [{"x": float(x), "y": float(y)} for x, y, _, _ in trajectory_data],
            "annotated_video_path": output_video_path
        }

        # Debug logging
        print(f"[BallAnalyzer] Analysis complete:")
        print(f"  - Trajectory points: {len(trajectory_data)}")
        print(f"  - Release detected: {release_idx is not None} (frame {release_idx})")
        print(f"  - Bounce detected: {bounce_idx is not None} (frame {bounce_idx})")
        print(f"  - Speed: {results['speed']} km/h")
        print(f"  - Line: {results['line']}, Length: {results['length']}")
        print(f"  - Calibration OK: {CALIB_OK}")

        # Save results JSON
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _compute_speed(self, trajectory_window, fps, H):
        """Simplified speed calculation"""
        if not trajectory_window or len(trajectory_window) < 2 or H is None:
            return None

        speeds = []
        for i in range(len(trajectory_window) - 1):
            x1, y1, f1, t1 = trajectory_window[i]
            x2, y2, f2, t2 = trajectory_window[i + 1]

            gm1 = px_to_ground_via_foot(x1, y1) if H is not None else None
            gm2 = px_to_ground_via_foot(x2, y2) if H is not None else None

            if gm1 and gm2:
                _, z1 = gm1
                _, z2 = gm2
                dz = abs(z2 - z1)
                dt = (f2 - f1) / fps
                if dt > 0:
                    speed_mps = dz / dt
                    speeds.append(speed_mps * 3.6)  # Convert to km/h

        if speeds:
            return float(np.median(speeds))
        return None