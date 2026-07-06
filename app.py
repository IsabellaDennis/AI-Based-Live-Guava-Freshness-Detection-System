import os
import cv2
import time
import datetime
import threading
import numpy as np
from urllib.parse import urlencode
import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from app.camera import Camera
from app.predictor import predict
from app.audio import speak

@st.cache_resource
def load_mobilenet_model():
    try:
        return MobileNetV2(weights='imagenet')
    except Exception as e:
        print(f"Warning: MobileNetV2 could not be loaded: {e}")
        return None

mobilenet_model = load_mobilenet_model()

# -----------------------------------------------------------------------------
# Streamlit Configuration (Hide all chrome to allow pixel-perfect iframe)
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .stApp { padding: 0 !important; margin: 0 !important; overflow: hidden !important; background: #0b1326 !important; }
.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; height: 100vh !important; overflow: hidden !important; }
header { display: none !important; }
footer { display: none !important; }
iframe { border: none !important; margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Shared State & Backend FSM Logic
# -----------------------------------------------------------------------------
class FSMState:
    WAITING = "WAITING"
    OBJECT_ENTERING = "OBJECT_ENTERING"
    VERIFYING_OBJECT = "VERIFYING_OBJECT"
    INSPECTING = "INSPECTING"
    RESULT = "RESULT"
    FROZEN_RESULT = "FROZEN_RESULT"
    OBJECT_REMOVED = "OBJECT_REMOVED"

class SystemState:
    def __init__(self):
        self.camera = None
        
        # Counters
        self.fresh_count = 0
        self.rotten_count = 0
        self.unknown_count = 0
        self.total_scans = 0
        self.history = []
        
        # Background subtractor for ROI detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        # FSM variables
        self.state = FSMState.WAITING
        self.state_enter_time = time.time()
        self.current_id = None
        self.cnn_buffer = []
        self.feed_active = False  # Changed from True for default stopped state
        self.camera_requested = False
        self.camera_stop_requested = False
        
        # Display variables
        self.current_frame = None
        self.current_stats = self.get_waiting_stats()
        self.start_time = time.time()

    def get_waiting_stats(self):
        return {
            "status": "WAITING FOR INSPECTION",
            "confidence": "0",
            "fresh_count": str(self.fresh_count),
            "rotten_count": str(self.rotten_count),
            "unknown_count": str(self.unknown_count),
            "total_scans": str(self.total_scans),
            "history_html": self.generate_history_html(),
            "is_waiting": True,
            "feed_active": self.feed_active,
            "color": "#2fd9f4",

            "runtime": format_runtime(time.time() - self.start_time) if hasattr(self, 'start_time') else "00:00:00"
        }

    def generate_history_html(self):
        if len(self.history) == 0:
            return """<!-- Empty State -->
<div class="flex-1 flex flex-col items-center justify-center text-center p-md overflow-y-auto bg-surface-container-low/30 h-full min-h-[300px]">
<span class="material-symbols-outlined text-4xl text-outline-variant mb-sm">inbox</span>
<div class="font-label-md text-label-md text-on-surface mb-xs">No inspections available.</div>
<div class="font-label-sm text-label-sm text-on-surface-variant">Start the camera feed and show Guava to begin.</div>
</div>"""
        
        rows = "<div class=\"flex flex-col\">"
        for i, item in enumerate(self.history):
            # Format from screen2.html
            
            # The span styling for result
            if item['result'] == "FRESH":
                result_span = '<span class="bg-[#003e1e] text-[#a0f0c0] border border-[#34c759] px-sm py-xs text-[10px] uppercase font-bold tracking-widest whitespace-nowrap">Accepted - Fresh</span>'
                conf_class = "text-[#34c759]"
            elif item['result'] == "ROTTEN":
                result_span = '<span class="bg-error-container text-on-error-container border border-error px-sm py-xs text-[10px] uppercase font-bold tracking-widest whitespace-nowrap">Rejected - Rotten</span>'
                conf_class = "text-error"
            else:
                result_span = '<span class="bg-[#5c3e00] text-[#ffdd88] border border-[#ffb300] px-sm py-xs text-[10px] uppercase font-bold tracking-widest whitespace-nowrap">Unknown</span>'
                conf_class = "text-[#ffb300]"

            rows += f"""<!-- Row {i+1} -->
<div class="grid grid-cols-[60px_2fr_2fr_2fr_1.5fr_1.5fr_120px] gap-sm px-lg py-md items-center table-row-industrial font-label-md text-label-md text-on-surface">
<div class="text-center text-on-surface-variant">{i+1}</div>
<div class="text-on-surface">{item['id']}</div>
<div class="text-on-surface-variant">{datetime.datetime.now().strftime("%Y-%m-%d")} {item['time']}</div>
<div class="flex justify-center">{result_span}</div>
<div class="text-center {conf_class} font-bold">{item['confidence']}</div>
<div class="text-center text-on-surface-variant">5.12s</div>
<div class="flex justify-center gap-xs">
<button class="w-8 h-8 flex items-center justify-center rounded-DEFAULT border border-outline-variant bg-surface-container-low hover:border-primary hover:text-primary transition-colors text-on-surface-variant"><span class="material-symbols-outlined text-[18px]">photo_camera</span></button>
<button class="w-8 h-8 flex items-center justify-center rounded-DEFAULT border border-outline-variant bg-surface-container-low hover:border-primary hover:text-primary transition-colors text-on-surface-variant"><span class="material-symbols-outlined text-[18px]">download</span></button>
</div>
</div>"""
        rows += "</div>"
        return rows

if 'sys_state' not in st.session_state:
    st.session_state.sys_state = SystemState()

state = st.session_state.sys_state

def format_runtime(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

import uuid
def generate_id():
    return f"GUA-{uuid.uuid4().hex[:6].upper()}"

def is_guava(image_roi):
    """Hybrid verification pipeline: Color -> Contour -> MobileNetV2"""
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([20, 30, 30])
    upper_bound = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    color_ratio = cv2.countNonZero(mask) / (image_roi.shape[0] * image_roi.shape[1])
    
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_valid_contour = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.3 < circularity < 1.2:
                    has_valid_contour = True
                    break
                    
    if color_ratio < 0.05 and not has_valid_contour:
        return False
        
    if not mobilenet_model:
        return True 
        
    resized = cv2.resize(image_roi, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(rgb, axis=0)
    img_array = preprocess_input(img_array)
    preds = mobilenet_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=3)[0]
    
    rejected_keywords = [
        'plastic bag', 'toy', 'phone', 'bottle', 'cup', 'book', 'paper', 'laptop', 
        'keyboard', 'mouse', 'face', 'hand', 'pear', 'avocado', 'passion fruit', 
        'apple', 'orange', 'lemon', 'mango', 'tennis ball', 'green ball', 
        'artificial', 'cellular', 'screen', 'monitor', 'chair', 'table', 'person', 'bag',
        'sunglasses', 'mask', 'jersey', 'suit', 'wig', 'seat belt', 'lab coat',
        'neck brace', 'sweatshirt', 'cardigan', 'abaya', 'academic gown', 't-shirt',
        'bow tie', 'bulletproof vest', 'jean', 'miniskirt', 'poncho', 'sarong',
        'sombrero', 'cowboy hat', 'hair spray', 'lotion', 'band aid', 'goggles',
        'stole', 'bath towel', 'swimming trunks', 'brassiere', 'maillot', 'pajama',
        'apron', 'lipstick', 'glasses', 'man', 'woman', 'human'
    ]
    for _, label, _ in decoded:
        label_lower = label.lower()
        if any(kw in label_lower for kw in rejected_keywords):
            return False
            
    return True

# -----------------------------------------------------------------------------
# Processing Loop (FSM)
# -----------------------------------------------------------------------------
def process_loop():
    while True:
        try:
            if state.camera_stop_requested:
                if state.camera is not None:
                    state.camera.release()
                    state.camera = None
                state.feed_active = False
                state.camera_stop_requested = False
                
            if state.camera_requested and not state.feed_active:
                if state.camera is None:
                    state.camera = Camera()
                state.feed_active = True
                state.camera_requested = False
                
            if not state.feed_active:
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 16
                state.current_frame = frame
                state.current_stats = state.get_waiting_stats()
                state.current_stats["feed_active"] = False
                time.sleep(0.1)
                continue
                
            # Process camera frame
            frame = state.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
                
            h, w = frame.shape[:2]
            size = min(h, w)
            sy, sx = (h - size) // 2, (w - size) // 2
            cropped = frame[sy:sy+size, sx:sx+size]
            
            fg_mask = state.bg_subtractor.apply(cropped)
            motion_ratio = cv2.countNonZero(fg_mask) / (size * size)
            object_present = motion_ratio > 0.03
            
            now = time.time()
            time_in_state = now - state.state_enter_time
            
            # State Machine Logic
            if state.state == FSMState.WAITING:
                if object_present:
                    if is_guava(cropped):
                        state.state = FSMState.INSPECTING
                        state.state_enter_time = now
                        state.current_id = generate_id()
                        state.cnn_buffer = []
                        
            elif state.state == FSMState.INSPECTING:
                # Draw Scanning Box
                cv2.rectangle(frame, (sx, sy), (sx+size, sy+size), (255, 255, 0), 2)
                cv2.putText(frame, "SCANNING...", (sx, max(30, sy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                
                # Tight crop using the foreground mask to remove background
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    pad = 10
                    bx = max(0, bx - pad)
                    by = max(0, by - pad)
                    bw = min(size - bx, bw + 2*pad)
                    bh = min(size - by, bh + 2*pad)
                    inference_img = cropped[by:by+bh, bx:bx+bw]
                else:
                    inference_img = cropped
                    
                score = predict(inference_img)
                state.cnn_buffer.append(score)
                
                if len(state.cnn_buffer) >= 6:
                    avg_score = sum(state.cnn_buffer) / 6.0
                    is_rotten = avg_score > 0.5
                    label = "ROTTEN" if is_rotten else "FRESH"
                    conf = avg_score if is_rotten else (1.0 - avg_score)
                    conf_pct = int(conf * 100)
                    
                    if label == "FRESH":
                        state.fresh_count += 1
                        speak("Fresh Guava Detected")
                        color_hex = "#10b981"
                    else:
                        state.rotten_count += 1
                        speak("Rotten Guava Detected")
                        color_hex = "#ef4444"
                        
                    state.history.append({
                        "id": state.current_id,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "result": label,
                        "confidence": f"{conf_pct}%",
                        "color": color_hex
                    })
                    state.history = state.history[:100]
                    state.total_scans += 1
                    
                    state.current_stats = {
                        "status": label,
                        "confidence": str(conf_pct),
                        "fresh_count": str(state.fresh_count),
                        "rotten_count": str(state.rotten_count),
                        "unknown_count": str(state.unknown_count),
                        "total_scans": str(state.total_scans),
                        "history_html": state.generate_history_html(),
                        "is_waiting": False,
                        "feed_active": state.feed_active,
                        "color": color_hex,
                        "runtime": format_runtime(time.time() - state.start_time)
                    }
                    state.state = FSMState.FROZEN_RESULT
                    state.state_enter_time = now
                    
            elif state.state == FSMState.FROZEN_RESULT:
                color_bgr = (0, 0, 255) if state.current_stats["status"] == "ROTTEN" else ((0, 255, 0) if state.current_stats["status"] == "FRESH" else (0, 165, 255))
                cv2.rectangle(frame, (sx, sy), (sx+size, sy+size), color_bgr, 2)
                cv2.putText(frame, f"{state.current_stats['status']} ({state.current_stats['confidence']}%)", (sx, max(30, sy - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
                
                state.current_stats["runtime"] = format_runtime(time.time() - state.start_time)
                
                if not object_present:
                    state.state = FSMState.OBJECT_REMOVED
                    state.state_enter_time = now
                    
            elif state.state == FSMState.OBJECT_REMOVED:
                color_bgr = (0, 0, 255) if state.current_stats["status"] == "ROTTEN" else ((0, 255, 0) if state.current_stats["status"] == "FRESH" else (0, 165, 255))
                cv2.rectangle(frame, (sx, sy), (sx+size, sy+size), color_bgr, 2)
                
                state.current_stats["runtime"] = format_runtime(time.time() - state.start_time)
                if object_present:
                    state.state = FSMState.FROZEN_RESULT
                    state.state_enter_time = now
                elif time_in_state >= 1.0:
                    state.state = FSMState.WAITING
                    state.state_enter_time = now
                    state.current_id = None
                    
            state.current_frame = frame
            time.sleep(0.03)
        except Exception as e:
            import traceback
            print(f"[PROCESS LOOP ERROR]: {e}")
            traceback.print_exc()
            time.sleep(0.5)

if 'fsm_started' not in st.session_state:
    st.session_state.fsm_started = True
    threading.Thread(target=process_loop, daemon=True).start()

import base64
import time

# Declare the custom component
import os
ui_component_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ui_component")
ui_component = components.declare_component("guava_ui", path=ui_component_path)

def render_dashboard():
    # Prepare base64 frame
    b64_frame = ""
    if state.current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', state.current_frame)
        if ret:
            b64_frame = base64.b64encode(buffer).decode('utf-8')
    else:
        # Send empty frame if none available
        b64_frame = ""

    # Render the custom component and get actions
    # We pass key="main_ui" to ensure stable mounting
    action = ui_component(stats=state.current_stats, frame=b64_frame, key="main_ui", default=None)
    print(f"[DEBUG] Action from UI: {action}")

    if action:
        act = action.get('action')
        if act == 'start' and not state.feed_active and not state.camera_requested:
            print("[APP] Received START FEED action - Requesting Camera")
            state.camera_requested = True
            state.state = FSMState.WAITING
            state.current_id = None
            state.start_time = time.time()
        elif act == 'stop' and state.feed_active and not state.camera_stop_requested:
            print("[APP] Received STOP FEED action - Requesting Stop")
            state.camera_stop_requested = True

render_dashboard()

# Hide Streamlit's visual "running" indicators to prevent flashing/blurring
st.markdown("""
    <style>
        [data-testid="stStatusWidget"] {
            visibility: hidden;
        }
        [data-testid="stAppViewContainer"] > .main {
            opacity: 1 !important;
        }
    </style>
""", unsafe_allow_html=True)

if state.feed_active or state.camera_requested or state.camera_stop_requested:
    time.sleep(0.1)  # slightly faster, but without blurring it will look smoother
    st.rerun()
