import time
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras.models import load_model

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gestura · Hand Sign Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  /* ── Root Variables ── */
  :root {
    --bg:       #0b0d11;
    --surface:  #13161e;
    --border:   #1f2330;
    --accent:   #5bffc8;
    --accent2:  #a78bfa;
    --danger:   #ff6b6b;
    --text:     #e8eaf2;
    --muted:    #6b7280;
  }

  /* ── Global Reset ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
  }

  [data-testid="stHeader"] { display: none; }
  [data-testid="stSidebar"] { display: none; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  footer { display: none; }

  /* ── Top Nav Bar ── */
  .nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .nav-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: -0.02em;
    color: var(--accent);
  }
  .nav-logo span { color: var(--text); }
  .nav-badge {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 20px;
  }

  /* ── Main Grid ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 0;
    height: calc(100vh - 65px);
  }

  /* ── Camera Panel ── */
  .camera-panel {
    padding: 32px 32px 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }
  .section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* Camera feed wrapper */
  .camera-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    position: relative;
  }
  .camera-wrap img { border-radius: 0; display: block; }

  /* Live indicator */
  .live-dot {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--accent);
    text-transform: uppercase;
  }
  .live-dot::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.4s ease infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.75); }
  }

  /* ── Right Panel ── */
  .right-panel {
    padding: 32px 28px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    overflow-y: auto;
    background: var(--surface);
  }

  /* Gesture Card */
  .gesture-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .gesture-card::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(91,255,200,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .gesture-label-sm {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }
  .gesture-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.03em;
    color: var(--accent);
    line-height: 1;
    min-height: 52px;
  }
  .gesture-value.idle { color: var(--border); }

  /* Confidence Bar */
  .conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 14px;
  }
  .conf-bar-bg {
    height: 4px;
    background: var(--border);
    border-radius: 4px;
    margin-top: 6px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  /* Stats Row */
  .stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .stat-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }
  .stat-card .val {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--text);
  }
  .stat-card .lbl {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 2px;
  }

  /* History Log */
  .history-wrap {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    flex: 1;
  }
  .history-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
  }
  .history-item:last-child { border-bottom: none; }
  .history-item .h-sign {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: var(--text);
  }
  .history-item .h-time { color: var(--muted); font-size: 0.68rem; }
  .history-item .h-conf {
    font-size: 0.68rem;
    color: var(--accent);
    background: rgba(91,255,200,0.08);
    border: 1px solid rgba(91,255,200,0.18);
    padding: 2px 8px;
    border-radius: 20px;
  }

  /* Toggle Switch */
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 0.78rem;
    color: var(--text);
    margin-bottom: 8px;
  }

  /* Streamlit checkbox overrides */
  [data-testid="stCheckbox"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
  }
  [data-testid="stCheckbox"] label {
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
  }
  [data-testid="stCheckbox"] span[data-baseweb="checkbox"] > div {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
  }

  /* Warning & alerts */
  .stWarning { background: rgba(255,107,107,0.08) !important; border-color: var(--danger) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* Empty history */
  .empty-hist {
    text-align: center;
    color: var(--muted);
    font-size: 0.74rem;
    padding: 20px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "last_prediction": None,
    "confidence": 0.0,
    "history": [],           # list of {"sign": str, "conf": float, "time": str}
    "total_detected": 0,
    "session_start": time.time(),
    "voice_enabled": True,
    "pending_speech": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def speak_text(text):
    """Speak using the browser's built-in Web Speech API."""
    if not st.session_state.voice_enabled:
        return
    st.session_state.pending_speech = text


# ─────────────────────────────────────────────
#  NORMALIZATION  (must match train_model.py)
# ─────────────────────────────────────────────
def normalize_landmarks(landmarks):
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    lm -= lm[0].copy()                          # translate wrist to origin
    scale = np.max(np.abs(lm)) + 1e-6
    lm /= scale                                 # scale to [-1, 1]
    return lm.flatten()


@st.cache_resource
def load_assets():
    model = load_model("bim_hand_sign_model.keras")
    label_encoder = joblib.load("label_encoder.pkl")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return model, label_encoder, mp_hands, hands


def elapsed_str():
    secs = int(time.time() - st.session_state.session_start)
    m, s = divmod(secs, 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
#  NAVBAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">Gestura<span>.</span></div>
  <div style="display:flex;align-items:center;gap:16px;">
    <span class="live-dot">Live</span>
    <span class="nav-badge">Hand Sign Translator</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
col_cam, col_panel = st.columns([3, 1.2], gap="small")

# Hidden speech injector — sits outside columns so it always renders
speech_ph = st.empty()

# ───── LEFT: Camera ─────
with col_cam:
    st.markdown('<div class="section-label">Camera Feed</div>', unsafe_allow_html=True)

    run = st.checkbox("▶  Start Camera", value=False)
    voice_toggle = st.checkbox("🔊  Voice Output", value=True)
    st.session_state.voice_enabled = voice_toggle

    frame_window = st.empty()
    warn_placeholder = st.empty()

# ───── RIGHT: Panel ─────
with col_panel:
    st.markdown('<div class="section-label">Detected Gesture</div>', unsafe_allow_html=True)

    gesture_ph = st.empty()
    conf_ph = st.empty()

    st.markdown('<div class="section-label" style="margin-top:8px;">Session Stats</div>', unsafe_allow_html=True)
    stats_ph = st.empty()

    st.markdown('<div class="section-label" style="margin-top:8px;">Detection History</div>', unsafe_allow_html=True)
    history_ph = st.empty()


# ─────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────
def render_speech():
    import streamlit.components.v1 as components
    word = st.session_state.get("pending_speech")
    if not word:
        speech_ph.empty()
        return
    safe = word.replace("'", "\\'")
    html = f"""
    <script>
    (function() {{
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        function doSpeak() {{
            var u = new SpeechSynthesisUtterance('{safe}');
            u.rate = 0.95; u.pitch = 1.0; u.volume = 1.0;
            var voices = window.speechSynthesis.getVoices();
            var eng = voices.find(v => v.lang.startsWith('en') && v.localService);
            if (eng) u.voice = eng;
            window.speechSynthesis.speak(u);
        }}
        if (window.speechSynthesis.getVoices().length === 0) {{
            window.speechSynthesis.addEventListener('voiceschanged', doSpeak, {{once:true}});
        }} else {{
            doSpeak();
        }}
    }})();
    </script>
    """
    with speech_ph:
        components.html(html, height=0, width=0)
    st.session_state.pending_speech = None


def render_gesture(label, conf):
    idle = label is None
    val_class = "gesture-value idle" if idle else "gesture-value"
    val_text = "——" if idle else label
    conf_pct = int(conf * 100)
    gesture_ph.markdown(f"""
    <div class="gesture-card">
      <div class="gesture-label-sm">Current Sign</div>
      <div class="{val_class}">{val_text}</div>
      <div class="conf-row">
        <span>Confidence</span>
        <span>{conf_pct}%</span>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats():
    elapsed = elapsed_str()
    total = st.session_state.total_detected
    unique = len({h["sign"] for h in st.session_state.history})
    stats_ph.markdown(f"""
    <div class="stats-row">
      <div class="stat-card"><div class="val">{total}</div><div class="lbl">Detected</div></div>
      <div class="stat-card"><div class="val">{unique}</div><div class="lbl">Unique</div></div>
      <div class="stat-card"><div class="val">{elapsed}</div><div class="lbl">Elapsed</div></div>
      <div class="stat-card"><div class="val">{'ON' if st.session_state.voice_enabled else 'OFF'}</div><div class="lbl">Voice</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_history():
    hist = st.session_state.history[-8:][::-1]
    if not hist:
        history_ph.markdown(
            '<div class="history-wrap"><div class="empty-hist">No gestures detected yet.</div></div>',
            unsafe_allow_html=True,
        )
        return
    items_html = ""
    for h in hist:
        items_html += f"""
        <div class="history-item">
          <span class="h-sign">{h['sign']}</span>
          <span class="h-conf">{int(h['conf']*100)}%</span>
          <span class="h-time">{h['time']}</span>
        </div>"""
    history_ph.markdown(f'<div class="history-wrap">{items_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INITIAL RENDER (idle state)
# ─────────────────────────────────────────────
render_gesture(None, 0.0)
render_stats()
render_history()

# ─────────────────────────────────────────────
#  CAMERA LOOP
# ─────────────────────────────────────────────
if run:
    model, label_encoder, mp_hands, hands_detector = load_assets()
    mp_draw = mp.solutions.drawing_utils

    draw_spec_hand = mp_draw.DrawingSpec(color=(91, 255, 200), thickness=2, circle_radius=3)
    draw_spec_conn = mp_draw.DrawingSpec(color=(167, 139, 250), thickness=1)

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            warn_placeholder.warning("⚠  Camera not detected. Check your webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        current_label = None
        current_conf = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    landmarks = normalize_landmarks(landmarks)   # ← normalise
                    X_input = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(X_input, verbose=0)
                    confidence = float(np.max(prediction))
                    predicted_class = np.argmax(prediction)
                    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                    current_label = predicted_label
                    current_conf = confidence

                    if confidence > 0.8 and predicted_label != st.session_state.last_prediction:
                        speak_text(predicted_label)
                        st.session_state.last_prediction = predicted_label
                        st.session_state.total_detected += 1
                        st.session_state.history.append({
                            "sign": predicted_label,
                            "conf": confidence,
                            "time": time.strftime("%H:%M:%S"),
                        })

                mp_draw.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    draw_spec_hand, draw_spec_conn,
                )

        # Overlay text on frame
        if current_label:
            cv2.putText(
                frame, f"{current_label}  {int(current_conf*100)}%",
                (16, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                (91, 255, 200), 2, cv2.LINE_AA,
            )

        frame_window.image(frame, channels="BGR", use_container_width=True)
        render_gesture(current_label, current_conf)
        render_stats()
        render_history()
        render_speech()

    cap.release()
else:
    # Placeholder when camera is off
    frame_window.markdown("""
    <div style="
      background:#13161e;
      border:1px solid #1f2330;
      border-radius:12px;
      height:420px;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      color:#6b7280;
      font-size:0.82rem;
      letter-spacing:0.08em;
    ">
      <div style="font-size:2.5rem;margin-bottom:14px;">📷</div>
      <div>Enable camera to begin translation</div>
    </div>
    """, unsafe_allow_html=True)