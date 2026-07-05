# Custom CSS to apply Stitch "QualiVision AI" theme to Streamlit
STITCH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

/* Lock the dashboard to 100vh and hide scrollbars */
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100vh !important;
    overflow: hidden !important;
    background-color: #111318 !important;
    color: #e2e2e8 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Hide Streamlit header/footer/sidebar */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Main Block padding reduction */
.block-container {
    padding: 24px !important;
    max-width: 1440px !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* Panels (Glassmorphism / Cards) */
.glass-panel {
    background: #060e20; /* surface-container-lowest */
    border: 1px solid #3c494c; /* outline-variant */
    border-radius: 8px; /* rounded-lg */
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); /* shadow-lg */
}

/* Typography Custom Classes for st.markdown */
.stitch-brand {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
}
.stitch-brand span {
    color: #00f5ff;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
}

.label-caps {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #849495;
    margin-bottom: 8px;
    display: block;
}

.data-lg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
}
.data-lg.success { color: #a2eeff; text-shadow: 0 0 10px rgba(162, 238, 255, 0.3); } /* primary-fixed */
.data-lg.error { color: #ffb4ab; text-shadow: 0 0 10px rgba(255, 180, 171, 0.3); } /* error */

/* Status Dot */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    padding: 4px 12px;
    border-radius: 9999px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
}
.glow-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.glow-dot.cyan { background: #2fd9f4; box-shadow: 0 0 8px #2fd9f4; animation: pulse-cyan 2s infinite; } /* primary-fixed-dim */
.glow-dot.success { background: #a2eeff; box-shadow: 0 0 8px #a2eeff; } /* primary-fixed */
.glow-dot.error { background: #ffb4ab; box-shadow: 0 0 8px #ffb4ab; } /* error */

@keyframes pulse-cyan {
    0% { opacity: 0.6; box-shadow: 0 0 4px #2fd9f4; }
    50% { opacity: 1; box-shadow: 0 0 12px #2fd9f4; }
    100% { opacity: 0.6; box-shadow: 0 0 4px #2fd9f4; }
}

/* History Table Fixed Height */
.history-container {
    max-height: 320px;
    overflow-y: auto;
    width: 100%;
}
.history-container table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.history-container th {
    position: sticky;
    top: 0;
    background: #111318;
    color: #849495;
    text-align: left;
    padding: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.history-container td {
    padding: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

/* Make Streamlit buttons match the primary button style */
.stButton > button {
    background: transparent !important;
    color: #00f5ff !important;
    border: 1px solid #00f5ff !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover {
    background: rgba(0, 245, 255, 0.1) !important;
    border-color: #00f5ff !important;
    color: #00f5ff !important;
}
</style>
"""

def render_html_component(html_content):
    import streamlit as st
    st.markdown(html_content, unsafe_allow_html=True)
