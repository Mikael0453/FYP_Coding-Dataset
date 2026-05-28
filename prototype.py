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

  :root {
    --bg:      #0b0d11;
    --surface: #13161e;
    --border:  #1f2330;
    --accent:  #5bffc8;
    --accent2: #a78bfa;
    --danger:  #ff6b6b;
    --text:    #e8eaf2;
    --muted:   #6b7280;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
  }
  [data-testid="stHeader"]  { display: none; }
  [data-testid="stSidebar"] { display: none; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  footer { display: none; }

  /* ── Navbar ── */
  .nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 40px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .nav-logo {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.5rem;
    letter-spacing: -0.02em; color: var(--accent);
  }
  .nav-logo span { color: var(--text); }
  .nav-badge {
    font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 20px;
  }
  .live-dot {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.7rem; letter-spacing: 0.1em; color: var(--accent); text-transform: uppercase;
  }
  .live-dot::before {
    content: ''; width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); animation: pulse 1.4s ease infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.75); }
  }

  /* ── Tab bar ── */
  [data-testid="stTabs"] > div:first-child {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 36px !important;
    gap: 0 !important;
  }
  [data-testid="stTabs"] button[role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 14px 22px !important;
    margin: 0 !important;
    transition: color 0.2s !important;
  }
  [data-testid="stTabs"] button[role="tab"]:hover { color: var(--text) !important; }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
  [data-testid="stTabs"] > div:last-child { padding: 0 !important; }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 12px;
  }

  /* ── Gesture card ── */
  .gesture-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 14px;
    padding: 28px 24px; text-align: center; position: relative; overflow: hidden;
  }
  .gesture-card::before {
    content: ''; position: absolute; top:-60px; left:-60px; width:200px; height:200px;
    background: radial-gradient(circle, rgba(91,255,200,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .gesture-label-sm {
    font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 16px;
  }
  .gesture-value {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
    letter-spacing: -0.03em; color: var(--accent); line-height: 1; min-height: 52px;
  }
  .gesture-value.idle { color: var(--border); }
  .conf-row {
    display: flex; justify-content: space-between;
    font-size: 0.68rem; color: var(--muted); margin-top: 14px;
  }
  .conf-bar-bg {
    height: 4px; background: var(--border); border-radius: 4px; margin-top: 6px; overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 4px; transition: width 0.3s ease;
  }

  /* ── Stats ── */
  .stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px;
  }
  .stat-card .val {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.5rem; color: var(--text);
  }
  .stat-card .lbl {
    font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); margin-top: 2px;
  }

  /* ── History log ── */
  .history-wrap {
    background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
  }
  .history-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 0.8rem;
  }
  .history-item:last-child { border-bottom: none; }
  .history-item .h-sign { font-family: 'Syne', sans-serif; font-weight: 700; color: var(--text); }
  .history-item .h-time { color: var(--muted); font-size: 0.68rem; }
  .history-item .h-conf {
    font-size: 0.68rem; color: var(--accent);
    background: rgba(91,255,200,0.08); border: 1px solid rgba(91,255,200,0.18);
    padding: 2px 8px; border-radius: 20px;
  }
  .empty-hist { text-align: center; color: var(--muted); font-size: 0.74rem; padding: 20px 0; }

  /* ── Streamlit checkbox ── */
  [data-testid="stCheckbox"] {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 10px 14px !important;
  }
  [data-testid="stCheckbox"] label {
    color: var(--text) !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
  }

  /* ── Chat bubbles ── */
  .chat-wrap {
    background: var(--bg); border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden; display: flex; flex-direction: column;
  }
  .chat-messages {
    height: 400px; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 14px;
  }
  .chat-messages::-webkit-scrollbar { width: 3px; }
  .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .bubble-nonsigner {
    align-self: flex-end; max-width: 72%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff; border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; font-size: 0.82rem; line-height: 1.55;
    box-shadow: 0 4px 14px rgba(79,70,229,0.25);
  }
  .bubble-signer {
    align-self: flex-start; max-width: 72%;
    background: rgba(91,255,200,0.08); border: 1px solid rgba(91,255,200,0.18);
    color: var(--accent); border-radius: 16px 16px 16px 4px;
    padding: 10px 16px; font-size: 0.82rem; line-height: 1.55;
  }
  .bubble-meta {
    font-size: 0.6rem; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 4px; opacity: 0.6;
  }
  .bubble-nonsigner .bubble-meta { text-align: right; }

  .chat-empty {
    height: 400px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.75rem; gap: 8px;
  }
  .chat-empty .icon { font-size: 2.2rem; }

  /* Info box */
  .info-box {
    background: rgba(91,255,200,0.05); border: 1px solid rgba(91,255,200,0.15);
    border-radius: 10px; padding: 14px 18px; font-size: 0.74rem; color: var(--muted);
    line-height: 1.7; margin-bottom: 16px;
  }
  .info-box span { color: var(--accent); }

  /* Streamlit text input in chat */
  [data-testid="stTextInput"] input {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important;
    padding: 12px 16px !important;
  }
  [data-testid="stTextInput"] input:focus {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
    outline: none !important;
  }

  /* Streamlit buttons */
  [data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.78rem !important; letter-spacing: 0.04em !important;
    background: var(--accent2) !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 22px !important; transition: opacity 0.2s !important;
    width: 100% !important;
  }
  [data-testid="stButton"] > button:hover { opacity: 0.82 !important; }

  .h-divider { height: 1px; background: var(--border); margin: 18px 0; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "last_prediction":  None,
    "confidence":       0.0,
    "history":          [],
    "total_detected":   0,
    "session_start":    time.time(),
    "voice_enabled":    True,
    "pending_speech":   None,
    "chat_messages":    [],       # {role, text, ts}
    "chat_speak":       None,   # kept for compatibility (unused in chat tab)
    "last_sign_added":  None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def speak_text(text, channel="sign"):
    if channel == "sign":
        if st.session_state.voice_enabled:
            st.session_state.pending_speech = text
    else:
        st.session_state.chat_speak = text


def normalize_landmarks(landmarks):
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    lm -= lm[0].copy()
    scale = np.max(np.abs(lm)) + 1e-6
    lm /= scale
    return lm.flatten()


@st.cache_resource
def load_assets():
    model     = load_model("bim_hand_sign_model.keras")
    label_enc = joblib.load("label_encoder.pkl")
    mp_h      = mp.solutions.hands
    hands_obj = mp_h.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    return model, label_enc, mp_h, hands_obj


def elapsed_str():
    s = int(time.time() - st.session_state.session_start)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def tts_inject(ph, text):
    """Render a zero-height TTS iframe into placeholder ph."""
    import streamlit.components.v1 as components
    if not text:
        return
    safe = text.replace("'", "\\'").replace('"', '\\"')
    html = f"""<script>
    (function(){{
      if(!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      function go(){{
        var u=new SpeechSynthesisUtterance('{safe}');
        u.rate=0.95;u.pitch=1.0;u.volume=1.0;
        var vs=window.speechSynthesis.getVoices();
        var e=vs.find(v=>v.lang.startsWith('en')&&v.localService);
        if(e) u.voice=e;
        window.speechSynthesis.speak(u);
      }}
      if(window.speechSynthesis.getVoices().length===0)
        window.speechSynthesis.addEventListener('voiceschanged',go,{{once:true}});
      else go();
    }})();
    </script>"""
    with ph:
        components.html(html, height=0, width=0)


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
#  TABS
# ─────────────────────────────────────────────
tab_trans, tab_chat = st.tabs(["🤟  Translator", "💬  Chat with Signer"])

# ══════════════════════════════════════════════
#  TAB 1 — TRANSLATOR
# ══════════════════════════════════════════════
with tab_trans:
    speech_ph = st.empty()

    col_cam, col_panel = st.columns([3, 1.2], gap="small")

    with col_cam:
        st.markdown('<div class="section-label" style="padding:24px 0 4px 24px;">Camera Feed</div>',
                    unsafe_allow_html=True)
        run          = st.checkbox("▶  Start Camera",  value=False, key="run_cam")
        voice_toggle = st.checkbox("🔊  Voice Output", value=True,  key="voice_tog")
        st.session_state.voice_enabled = voice_toggle
        frame_window     = st.empty()
        warn_placeholder = st.empty()

    with col_panel:
        st.markdown('<div class="section-label" style="padding:24px 0 4px 0;">Detected Gesture</div>',
                    unsafe_allow_html=True)
        gesture_ph = st.empty()
        st.markdown('<div class="section-label" style="margin-top:16px;">Session Stats</div>',
                    unsafe_allow_html=True)
        stats_ph = st.empty()
        st.markdown('<div class="section-label" style="margin-top:16px;">Detection History</div>',
                    unsafe_allow_html=True)
        history_ph = st.empty()

    # render helpers
    def render_gesture(label, conf):
        idle      = label is None
        val_cls   = "gesture-value idle" if idle else "gesture-value"
        val_txt   = "——" if idle else label
        pct       = int(conf * 100)
        gesture_ph.markdown(f"""
        <div class="gesture-card">
          <div class="gesture-label-sm">Current Sign</div>
          <div class="{val_cls}">{val_txt}</div>
          <div class="conf-row"><span>Confidence</span><span>{pct}%</span></div>
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;"></div></div>
        </div>""", unsafe_allow_html=True)

    def render_stats():
        elapsed = elapsed_str()
        total   = st.session_state.total_detected
        unique  = len({h["sign"] for h in st.session_state.history})
        stats_ph.markdown(f"""
        <div class="stats-row">
          <div class="stat-card"><div class="val">{total}</div><div class="lbl">Detected</div></div>
          <div class="stat-card"><div class="val">{unique}</div><div class="lbl">Unique</div></div>
          <div class="stat-card"><div class="val">{elapsed}</div><div class="lbl">Elapsed</div></div>
          <div class="stat-card"><div class="val">{'ON' if st.session_state.voice_enabled else 'OFF'}</div><div class="lbl">Voice</div></div>
        </div>""", unsafe_allow_html=True)

    def render_history():
        hist = st.session_state.history[-8:][::-1]
        if not hist:
            history_ph.markdown(
                '<div class="history-wrap"><div class="empty-hist">No gestures detected yet.</div></div>',
                unsafe_allow_html=True)
            return
        items = "".join(f"""
        <div class="history-item">
          <span class="h-sign">{h['sign']}</span>
          <span class="h-conf">{int(h['conf']*100)}%</span>
          <span class="h-time">{h['time']}</span>
        </div>""" for h in hist)
        history_ph.markdown(f'<div class="history-wrap">{items}</div>', unsafe_allow_html=True)

    render_gesture(None, 0.0)
    render_stats()
    render_history()

    if run:
        model, label_encoder, mp_hands, hands_det = load_assets()
        mp_draw   = mp.solutions.drawing_utils
        d_hand    = mp_draw.DrawingSpec(color=(91, 255, 200), thickness=2, circle_radius=3)
        d_conn    = mp_draw.DrawingSpec(color=(167, 139, 250), thickness=1)
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                warn_placeholder.warning("⚠  Camera not detected. Check your webcam.")
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_det.process(rgb)

            cur_label = None
            cur_conf  = 0.0

            if result.multi_hand_landmarks:
                for hl in result.multi_hand_landmarks:
                    lms = []
                    for lm in hl.landmark:
                        lms.extend([lm.x, lm.y, lm.z])
                    if len(lms) == 63:
                        lms_n    = normalize_landmarks(lms)
                        pred     = model.predict(np.array(lms_n).reshape(1,-1), verbose=0)
                        conf     = float(np.max(pred))
                        cls      = np.argmax(pred)
                        lbl      = label_encoder.inverse_transform([cls])[0]
                        cur_label, cur_conf = lbl, conf

                        if conf > 0.8 and lbl != st.session_state.last_prediction:
                            ts = time.strftime("%H:%M:%S")
                            speak_text(lbl, "sign")
                            st.session_state.last_prediction = lbl
                            st.session_state.total_detected += 1
                            st.session_state.history.append({"sign": lbl, "conf": conf, "time": ts})
                            # push to chat as signer bubble
                            if lbl != st.session_state.last_sign_added:
                                st.session_state.chat_messages.append(
                                    {"role": "signer", "text": lbl, "ts": ts})
                                st.session_state.last_sign_added = lbl

                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS, d_hand, d_conn)

            if cur_label:
                cv2.putText(frame, f"{cur_label}  {int(cur_conf*100)}%",
                            (16, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (91, 255, 200), 2, cv2.LINE_AA)

            frame_window.image(frame, channels="BGR", use_container_width=True)
            render_gesture(cur_label, cur_conf)
            render_stats()
            render_history()

            word = st.session_state.get("pending_speech")
            if word:
                tts_inject(speech_ph, word)
                st.session_state.pending_speech = None

        cap.release()
    else:
        frame_window.markdown("""
        <div style="background:#13161e;border:1px solid #1f2330;border-radius:12px;height:420px;
             display:flex;flex-direction:column;align-items:center;justify-content:center;
             color:#6b7280;font-size:0.82rem;letter-spacing:0.08em;">
          <div style="font-size:2.5rem;margin-bottom:14px;">📷</div>
          <div>Enable camera to begin translation</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — CHAT WITH SIGNER  (BIM sign display)
# ══════════════════════════════════════════════

# ── BIM SignBank page URLs — one per letter/digit ───────────────────────────
# bimsignbank.org is the official MFD resource. URL pattern: /alphabets/<x>/<x>
BIM_LETTER_PAGE = {
    ch: f"https://www.bimsignbank.org/alphabets/{ch.lower()}/{ch.lower()}"
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
}
BIM_DIGIT_PAGE = {
    str(d): f"https://www.bimsignbank.org/numbers/{d}/{d}"
    for d in range(10)
}


def build_bim_sign_component(text: str, height_px: int = 200) -> str:
    """
    Build a self-contained HTML document that renders each letter of `text`
    as a card containing an iframe pointed at bimsignbank.org.
    The browser executes the React app inside each iframe and shows the
    real BIM hand-sign image from the official MFD database.
    Returns the full HTML string for use with streamlit.components.v1.html().
    """
    import html as _html
    text_upper = text.upper().strip()
    words = text_upper.split()
    if not words:
        return ""

    word_blocks = []
    for word in words:
        cards = []
        for ch in word:
            if ch.isalpha():
                page_url = BIM_LETTER_PAGE[ch]
            elif ch.isdigit():
                page_url = BIM_DIGIT_PAGE[ch]
            else:
                continue
            safe = _html.escape(ch)
            cards.append(f"""
            <div class="card">
              <div class="lbl">{safe}</div>
              <div class="fw">
                <iframe src="{page_url}" title="BIM {safe}"
                        scrolling="no" loading="lazy"
                        sandbox="allow-scripts allow-same-origin allow-popups">
                </iframe>
              </div>
            </div>""")
        if cards:
            word_blocks.append(
                f'<div class="word">{"".join(cards)}</div>'
            )

    if not word_blocks:
        return ""

    # Viewport height for the component — grows with message length
    n_letters = sum(len(w) for w in text_upper.split())
    comp_h = max(height_px, 160 + (n_letters // 8) * 20)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'DM Mono',monospace;overflow-x:auto;padding:4px 0;}}
.row{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;}}
.word{{
  display:flex;flex-wrap:wrap;gap:6px;align-items:flex-end;
  padding-right:16px;
  border-right:2px solid rgba(91,255,200,0.25);
  margin-right:4px;
}}
.word:last-child{{border-right:none;padding-right:0;}}
.card{{
  display:flex;flex-direction:column;align-items:center;gap:4px;
  background:#13161e;border:1px solid #1f2330;border-radius:8px;
  padding:5px 4px 4px;width:96px;
}}
.lbl{{
  font-size:11px;font-weight:700;color:#5bffc8;
  letter-spacing:0.12em;text-transform:uppercase;
  font-family:'Syne',sans-serif;
}}
/* The iframe wrapper clips the BIM page to show only the sign image area */
.fw{{
  width:88px;height:100px;overflow:hidden;border-radius:6px;
  background:#0b0d11;position:relative;
}}
.fw iframe{{
  width:900px;height:700px;border:none;
  /* Zoom/offset to crop to the hand-sign image region of bimsignbank.org */
  transform:scale(0.155) translate(-310px,-180px);
  transform-origin:top left;
  pointer-events:none;
}}
.note{{
  margin-top:10px;font-size:10px;color:#6b7280;
  background:rgba(167,139,250,0.06);
  border:1px solid rgba(167,139,250,0.15);
  border-radius:6px;padding:7px 10px;line-height:1.6;
}}
</style>
</head>
<body>
  <div class="row">{chr(10).join(word_blocks)}</div>
  <div class="note">
    🇲🇾 Live BIM signs from <b>bimsignbank.org</b> — official Bahasa Isyarat Malaysia resource by MFD.
    Each card shows the real BIM fingerspelling sign. Word boundaries marked with green dividers.
  </div>
</body>
</html>""", comp_h


with tab_chat:
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # Extra CSS for BIM sign display
    st.markdown("""
    <style>
      /* ── BIM sign display ── */
      .bim-display-box {
        background: var(--bg);
        border: 1px solid rgba(91,255,200,0.25);
        border-radius: 14px;
        padding: 20px 18px 16px;
        margin-top: 16px;
      }
      .bim-display-title {
        font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 14px; display: flex;
        align-items: center; gap: 8px;
      }
      .bim-display-title::before {
        content: ''; display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: var(--accent);
      }
      .bim-message-text {
        font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
        color: var(--text); margin-bottom: 16px; letter-spacing: -0.01em;
      }
      .bim-signs-row {
        display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start;
      }
      .bim-word-group {
        display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-end;
        padding-right: 14px; border-right: 1px solid var(--border);
        margin-right: 4px;
      }
      .bim-word-group:last-child { border-right: none; }
      .bim-letter-card {
        display: flex; flex-direction: column; align-items: center; gap: 4px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 6px 4px 4px;
        min-width: 52px;
      }
      .bim-letter-label {
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.68rem;
        color: var(--accent); letter-spacing: 0.1em;
      }
      .bim-letter-img {
        width: 48px; height: 48px; object-fit: contain;
        border-radius: 4px; background: rgba(255,255,255,0.04);
      }
      .bim-letter-fallback {
        width: 48px; height: 48px;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem;
        color: var(--text); background: rgba(91,255,200,0.06);
        border-radius: 6px; border: 1px dashed rgba(91,255,200,0.2);
      }
      .bim-empty-state {
        text-align: center; padding: 24px 0; color: var(--muted); font-size: 0.75rem; line-height: 1.8;
      }
      .bim-empty-icon { font-size: 2.2rem; margin-bottom: 8px; }
      .bim-note {
        margin-top: 12px; font-size: 0.66rem; color: var(--muted);
        background: rgba(167,139,250,0.06); border: 1px solid rgba(167,139,250,0.15);
        border-radius: 8px; padding: 8px 12px; line-height: 1.7;
      }
    </style>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1], gap="medium")

    with col_main:
        # Info banner — corrected for deaf signer
        st.markdown("""
        <div class="info-box">
          <span>💬 Chat with the Signer (Deaf)</span><br/>
          Type a message below. The system will display the corresponding
          <span>BIM (Bahasa Isyarat Malaysia)</span> hand signs so the
          <span>deaf signer can read your message</span> through fingerspelling.
        </div>""", unsafe_allow_html=True)

        # ── BIM Sign Display Panel ──────────────────────────────────────────
        st.markdown('<div class="section-label">BIM Hand Sign Display for Signer</div>',
                    unsafe_allow_html=True)

        last_nonsigner = next(
            (m for m in reversed(st.session_state.chat_messages) if m["role"] == "nonsigner"),
            None
        )

        if last_nonsigner is None:
            st.markdown("""
            <div class="bim-display-box">
              <div class="bim-display-title">Awaiting Message</div>
              <div class="bim-empty-state">
                <div class="bim-empty-icon">🤟</div>
                <div>Type a message below and send it.<br/>
                The real BIM hand sign from <b>bimsignbank.org</b> will appear<br/>
                for each letter so the signer can follow along.</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            import streamlit.components.v1 as components
            msg_text = last_nonsigner["text"]
            bim_result = build_bim_sign_component(msg_text)
            if bim_result:
                comp_html, comp_h = bim_result
                st.markdown(f"""
                <div class="bim-display-box">
                  <div class="bim-display-title">BIM Fingerspelling — Show this to the signer</div>
                  <div class="bim-message-text">"{msg_text}"</div>
                </div>""", unsafe_allow_html=True)
                components.html(comp_html, height=comp_h, scrolling=False)
            else:
                st.markdown("""
                <div class="bim-display-box">
                  <div class="bim-empty-state">
                    <div>No signable characters found in message.</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)

        # ── Chat message history ────────────────────────────────────────────
        st.markdown('<div class="section-label">Message History</div>', unsafe_allow_html=True)
        msgs = st.session_state.chat_messages
        if not msgs:
            st.markdown("""
            <div class="chat-wrap">
              <div class="chat-empty">
                <div class="icon">🤟</div>
                <div>No messages yet</div>
                <div style="font-size:0.65rem;margin-top:4px;opacity:0.55;">
                  Type below — signer replies appear from the Translator tab.
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            bubbles = ""
            for m in msgs[-30:]:
                if m["role"] == "nonsigner":
                    bubbles += f"""
                    <div class="bubble-nonsigner">
                      <div class="bubble-meta">You · {m['ts']}</div>
                      {m['text']}
                    </div>"""
                else:
                    bubbles += f"""
                    <div class="bubble-signer">
                      <div class="bubble-meta">🤟 Signer · {m['ts']}</div>
                      {m['text']}
                    </div>"""
            st.markdown(f"""
            <div class="chat-wrap">
              <div class="chat-messages" id="chatBox">{bubbles}</div>
            </div>
            <script>
              var b=document.getElementById('chatBox');
              if(b) b.scrollTop=b.scrollHeight;
            </script>""", unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        # ── Quick replies ──
        st.markdown('<div class="section-label">Quick Replies</div>', unsafe_allow_html=True)
        quick = ["Hello!", "Please repeat", "Yes", "No",
                 "Thank you", "Can you slow down?", "I understand", "Nice to meet you"]
        qcols = st.columns(4)
        for i, q in enumerate(quick):
            with qcols[i % 4]:
                if st.button(q, key=f"qr_{i}"):
                    ts = time.strftime("%H:%M:%S")
                    st.session_state.chat_messages.append({"role": "nonsigner", "text": q, "ts": ts})
                    st.rerun()

        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)

        # ── Text input ──
        st.markdown('<div class="section-label">Type a Message</div>', unsafe_allow_html=True)
        in_col, btn_col = st.columns([5, 1])
        with in_col:
            typed = st.text_input("msg", label_visibility="collapsed",
                                  placeholder="Type here and press Send…",
                                  key="chat_typed")
        with btn_col:
            send = st.button("Send →", key="send_btn")

        if send and typed.strip():
            ts = time.strftime("%H:%M:%S")
            st.session_state.chat_messages.append({"role": "nonsigner", "text": typed.strip(), "ts": ts})
            st.rerun()

        # Clear button
        if st.button("🗑  Clear Chat", key="clear_chat"):
            st.session_state.chat_messages   = []
            st.session_state.last_sign_added = None
            st.rerun()

    # ── Right sidebar ──
    with col_side:
        st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
             padding:20px 18px;font-size:0.74rem;line-height:1.9;color:var(--muted);">
          <div style="margin-bottom:10px;">
            <span style="color:var(--accent2);font-weight:600;">You (Non-Signer)</span><br/>
            Type a message or tap a Quick Reply.
            The system will show the <b style="color:var(--text);">BIM hand sign</b>
            for each letter so the <b style="color:var(--text);">deaf signer</b> can read it.
          </div>
          <div style="height:1px;background:var(--border);margin:12px 0;"></div>
          <div>
            <span style="color:var(--accent);font-weight:600;">🤟 Signer (Deaf)</span><br/>
            Reads the BIM signs displayed on screen, then
            replies by performing signs in the
            <b style="color:var(--text);">Translator tab</b>.
            Detected signs appear as reply bubbles here.
          </div>
          <div style="height:1px;background:var(--border);margin:12px 0;"></div>
          <div style="font-size:0.67rem;color:#4b5563;">
            🇲🇾 Uses <b>Bahasa Isyarat Malaysia (BIM)</b> fingerspelling —
            the official Malaysian deaf sign system.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Signer Replies</div>', unsafe_allow_html=True)

        signer_msgs = [m for m in st.session_state.chat_messages
                       if m["role"] == "signer"][-6:][::-1]
        if not signer_msgs:
            st.markdown("""
            <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                 padding:16px 18px;font-size:0.72rem;color:var(--muted);text-align:center;line-height:1.7;">
              No signs detected yet.<br/>Start the camera in the Translator tab.
            </div>""", unsafe_allow_html=True)
        else:
            rows = "".join(f"""
            <div class="history-item">
              <span class="h-sign">{m['text']}</span>
              <span class="h-time">{m['ts']}</span>
            </div>""" for m in signer_msgs)
            st.markdown(f'<div class="history-wrap">{rows}</div>', unsafe_allow_html=True)

        # ── BIM Fingerspelling Reference Chart ──
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">BIM Fingerspelling Reference</div>',
                    unsafe_allow_html=True)
        ref_cards = ""
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            page_url = BIM_LETTER_PAGE[ch]
            ref_cards += f"""
            <a href="{page_url}" target="_blank" rel="noopener"
               title="View BIM sign for {ch} on bimsignbank.org"
               style="display:inline-flex;flex-direction:column;align-items:center;
                      gap:3px;background:var(--bg);border:1px solid var(--border);
                      border-radius:6px;padding:5px 4px;margin:2px;
                      text-decoration:none;transition:border-color 0.2s;"
               onmouseover="this.style.borderColor='#5bffc8'"
               onmouseout="this.style.borderColor='#1f2330'">
              <span style="font-size:0.6rem;color:var(--accent);font-family:'Syne',sans-serif;
                    font-weight:700;">{ch}</span>
              <span style="font-size:0.55rem;color:#4b5563;">↗</span>
            </a>"""
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:8px;">{ref_cards}</div>
        <div style="font-size:0.62rem;color:#4b5563;line-height:1.6;">
          Tap any letter to view its full BIM sign on
          <a href="https://www.bimsignbank.org/groups/general/alphabets"
             target="_blank" style="color:#5bffc8;text-decoration:none;">bimsignbank.org</a>
        </div>""", unsafe_allow_html=True)

