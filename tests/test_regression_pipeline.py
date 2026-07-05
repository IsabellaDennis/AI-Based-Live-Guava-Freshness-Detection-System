import os
import sys
import numpy as np
import pytest

# Ensure cv2 is available for tests
import cv2

class SimulatedQCSystem:
    def __init__(self, limit_fresh=0.25, limit_rotten=0.75, presence_sensitivity=10.0, enable_filters=True):
        self.limit_fresh = limit_fresh
        self.limit_rotten = limit_rotten
        self.presence_sensitivity = presence_sensitivity
        self.enable_filters = enable_filters
        
        # State variables
        self.count_accepted = 0
        self.count_rejected = 0
        self.count_unknown = 0
        self.total_frames = 0
        self.history = []
        self.score_filter = []
        self.score_filter_maxlen = 6
        
        self.qc_state = "WAITING_FOR_OBJECT"
        self.current_inspection_id = None
        self.inspection_logged = False
        self.inspection_voice_spoken = False
        self.empty_frames_count = 0
        self.present_frames_count = 0
        self.non_guava_frames = 0
        self.warmup_frames = 0
        
        # Freezing & movement state variables
        self.inspection_frozen = False
        self.frozen_classification = "Empty"
        self.frozen_confidence = 0.0
        self.frozen_state_action = "Waiting for Object"
        self.frozen_rotten_sigmoid = 0.5
        self.frozen_smooth_score = 0.5
        self.prev_gray_crop = None
        
        # Debug logs enqueued
        self.debug_logs = []
        # CNN call counter for verifying inference-per-inspection
        self.cnn_call_count = 0
        # Speech announcements enqueued
        self.spoken_texts = []
        # Siren calls enqueued
        self.siren_triggered = False

    def speak(self, text):
        self.spoken_texts.append(text)

    def trigger_rotten_siren(self):
        self.siren_triggered = True

    def log_debug(self, message):
        self.debug_logs.append(message)

    def process_frame(self, change_ratio, is_guava_color, is_guava_imagenet, cnn_score, blemish_ratio, crop_roi=None):
        self.total_frames += 1
        self.warmup_frames += 1
        
        is_any_obj = (change_ratio > (self.presence_sensitivity / 100.0)) and (self.warmup_frames > 25)
        is_guava_present = is_any_obj and (not self.enable_filters or is_guava_color)
        
        # Simulate gray crop
        if crop_roi is not None:
            gray_crop = crop_roi
        else:
            gray_crop = np.zeros((10, 10), dtype=np.uint8)
            
        # Movement tracking (lightweight frame-to-frame diff)
        movement_score = 0.0
        if is_any_obj and self.prev_gray_crop is not None:
            prev_crop = self.prev_gray_crop
            if prev_crop.shape != gray_crop.shape:
                prev_crop = cv2.resize(prev_crop, (gray_crop.shape[1], gray_crop.shape[0]))
            diff_roi = cv2.absdiff(gray_crop, prev_crop)
            _, thresh_roi = cv2.threshold(diff_roi, 15, 255, cv2.THRESH_BINARY)
            movement_ratio = float(np.sum(thresh_roi > 0) / thresh_roi.size)
            movement_score = movement_ratio * 100.0
            
        self.prev_gray_crop = gray_crop.copy()
        
        # Unfreeze on significant movement
        if is_any_obj and self.inspection_frozen:
            if movement_score > 8.0:
                self.log_debug("Object Moved")
                self.inspection_frozen = False
                self.qc_state = "OBJECT_DETECTED"
                self.current_inspection_id = f"INSP-SIM-{self.total_frames}"
                self.inspection_logged = False
                self.inspection_voice_spoken = False
                self.score_filter.clear()
                self.non_guava_frames = 0
                self.log_debug("Inspection Started")
        
        if not is_any_obj:
            self.empty_frames_count += 1
            self.present_frames_count = 0
            
            if self.empty_frames_count == 1:
                self.log_debug("Object Removed")
                
            if self.empty_frames_count >= 15:
                if self.qc_state != "WAITING_FOR_OBJECT":
                    self.log_debug("Waiting For Object")
                self.qc_state = "WAITING_FOR_OBJECT"
                self.current_inspection_id = None
                self.inspection_logged = False
                self.inspection_voice_spoken = False
                self.inspection_frozen = False
                self.score_filter.clear()
                self.frozen_classification = "Empty"
                self.frozen_confidence = 0.0
                self.frozen_state_action = "Waiting for Object"
                self.frozen_rotten_sigmoid = 0.5
                self.frozen_smooth_score = 0.5
        else:
            self.empty_frames_count = 0
            self.present_frames_count += 1
            
            if self.qc_state == "WAITING_FOR_OBJECT" and self.present_frames_count >= 3:
                self.qc_state = "OBJECT_DETECTED"
                self.current_inspection_id = f"INSP-SIM-{self.total_frames}"
                self.inspection_logged = False
                self.inspection_voice_spoken = False
                self.inspection_frozen = False
                self.score_filter.clear()
                self.log_debug("Inspection Started")

        # Initialize defaults for this frame's diagnostic display values
        classification = "Empty"
        confidence_pct = 0.0
        state_action = "Waiting for Object"
        rotten_sigmoid = 0.5
        smooth_score = 0.5

        if self.inspection_frozen:
            classification = self.frozen_classification
            confidence_pct = self.frozen_confidence
            state_action = self.frozen_state_action
            rotten_sigmoid = self.frozen_rotten_sigmoid
            smooth_score = self.frozen_smooth_score
            
        elif self.qc_state == "WAITING_FOR_OBJECT":
            classification = "Empty"
            confidence_pct = 0.0
            state_action = "Waiting for Object"
            rotten_sigmoid = 0.5
            smooth_score = 0.5
            self.score_filter.clear()
            self.non_guava_frames = 0
            
        elif self.qc_state in ["OBJECT_DETECTED", "INSPECTING", "ACCEPTED", "REJECTED", "NOT_A_GUAVA"]:
            if is_guava_present:
                is_verified = True
                if self.enable_filters and not is_guava_imagenet:
                    is_verified = False
                    
                if is_verified:
                    self.non_guava_frames = 0
                    if self.qc_state in ["OBJECT_DETECTED", "INSPECTING", "NOT_A_GUAVA"]:
                        if self.qc_state == "NOT_A_GUAVA":
                            self.inspection_logged = False
                            self.inspection_voice_spoken = False
                        self.qc_state = "INSPECTING"
                        
                        rotten_sigmoid = cnn_score
                        self.cnn_call_count += 1
                        self.log_debug(f"CNN inference executed (score={rotten_sigmoid:.4f}, buffer={len(self.score_filter)+1}/{self.score_filter_maxlen})")
                        self.score_filter.append(rotten_sigmoid)
                        if len(self.score_filter) > self.score_filter_maxlen:
                            self.score_filter.pop(0)
                        
                        smooth_score = sum(self.score_filter) / len(self.score_filter)
                        
                        # Apply Blemish Ratio Overrides
                        if blemish_ratio >= 0.08:
                            smooth_score = 0.85
                            
                        classification = "Unknown"
                        confidence_pct = 50.0 + abs(smooth_score - 0.5) * 100
                        state_action = "Inspecting"
                        
                        if len(self.score_filter) >= self.score_filter_maxlen:
                            if smooth_score > self.limit_rotten:
                                self.qc_state = "REJECTED"
                            elif smooth_score < self.limit_fresh:
                                self.qc_state = "ACCEPTED"
                                
                            # Set display values for the decision frame (no additional CNN call)
                            if self.qc_state == "ACCEPTED":
                                classification = "Fresh"
                                confidence_pct = (1.0 - smooth_score) * 100
                                state_action = "Accepted"
                            elif self.qc_state == "REJECTED":
                                classification = "Rotten"
                                confidence_pct = smooth_score * 100
                                state_action = "Rejected"
                else:
                    self.non_guava_frames += 1
                    if self.non_guava_frames >= 5:
                        self.qc_state = "NOT_A_GUAVA"
                    else:
                        state_action = "Inspecting"
            else:
                self.non_guava_frames += 1
                if self.non_guava_frames >= 5:
                    self.qc_state = "NOT_A_GUAVA"
                else:
                    state_action = "Inspecting"
                
            if self.qc_state == "NOT_A_GUAVA":
                classification = "Unknown"
                confidence_pct = 0.0
                state_action = "Not a Guava"
                rotten_sigmoid = 0.5
                smooth_score = 0.5
                self.score_filter.clear()

        # Freeze logic
        if not self.inspection_frozen and self.qc_state in ["ACCEPTED", "REJECTED", "NOT_A_GUAVA"]:
            self.frozen_classification = classification
            self.frozen_confidence = confidence_pct
            self.frozen_state_action = state_action
            self.frozen_rotten_sigmoid = rotten_sigmoid
            self.frozen_smooth_score = smooth_score
            self.inspection_frozen = True
            self.log_debug("Inspection Finished")
            self.log_debug("Inspection Frozen")

        # Log results and update statistics ONCE per inspection cycle
        if not self.inspection_logged and self.qc_state in ["ACCEPTED", "REJECTED", "NOT_A_GUAVA"]:
            if self.qc_state == "ACCEPTED":
                self.count_accepted += 1
            elif self.qc_state == "REJECTED":
                self.count_rejected += 1
                self.trigger_rotten_siren()
            elif self.qc_state == "NOT_A_GUAVA":
                self.count_unknown += 1
                
            self.history.append({
                "id": self.current_inspection_id,
                "prediction": classification if self.qc_state in ["ACCEPTED", "REJECTED"] else "Not a Guava",
                "confidence": confidence_pct,
                "status": state_action
            })
            self.inspection_logged = True
            
        # Speech triggers ONCE per inspection cycle
        if not self.inspection_voice_spoken and self.qc_state in ["ACCEPTED", "REJECTED", "NOT_A_GUAVA"]:
            if self.qc_state == "ACCEPTED":
                self.speak("Fresh Guava Detected")
            elif self.qc_state == "REJECTED":
                self.speak("Rotten Guava Detected")
            elif self.qc_state == "NOT_A_GUAVA":
                self.speak("Unknown fruit detected")
            self.inspection_voice_spoken = True

        return state_action, classification, confidence_pct

# =====================================================================
# Regression Tests
# =====================================================================

def test_pipeline_fresh_guava_accepted():
    """Verify that a clean fresh guava is enqueued, inspected, and consistently accepted with one announcement."""
    sys = SimulatedQCSystem()
    
    # 1. Warm up the system (25 frames)
    for _ in range(25):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    assert sys.qc_state == "WAITING_FOR_OBJECT"
    
    # 2. Object enters ROI (change_ratio > threshold 10%)
    for _ in range(2):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    assert sys.qc_state == "WAITING_FOR_OBJECT"
    
    sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    assert sys.qc_state == "INSPECTING"
    assert "Inspection Started" in sys.debug_logs
    
    # 3. Inspections buffers (needs 5 more frames to complete buffer of size 6)
    for i in range(4):
        state_action, classification, confidence = sys.process_frame(
            change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01
        )
        assert sys.qc_state == "INSPECTING"
        assert state_action == "Inspecting"
        assert classification == "Unknown"
        
    # 6th frame completes buffer (mean score = 0.05 < 0.25 limit_fresh) -> Accepted & Frozen
    state_action, classification, confidence = sys.process_frame(
        change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01
    )
    assert sys.qc_state == "ACCEPTED"
    assert state_action == "Accepted"
    assert classification == "Fresh"
    assert confidence > 90.0
    assert sys.inspection_frozen is True
    assert "Inspection Finished" in sys.debug_logs
    assert "Inspection Frozen" in sys.debug_logs
    
    # 4. Check outputs
    assert sys.count_accepted == 1
    assert sys.count_rejected == 0
    assert len(sys.spoken_texts) == 1
    assert sys.spoken_texts[0] == "Fresh Guava Detected"

def test_pipeline_static_object_freezes_inference():
    """Verify that if the object remains static in the ROI, inference is bypassed and counters do not re-increment."""
    sys = SimulatedQCSystem()
    for _ in range(25):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    for _ in range(5):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    
    # Completed & Frozen
    assert sys.inspection_frozen is True
    assert sys.count_accepted == 1
    assert len(sys.spoken_texts) == 1
    
    # CNN was called exactly 6 times during the inspection buffer fill phase
    cnn_at_freeze = sys.cnn_call_count
    assert cnn_at_freeze == 6
    
    # Feed 10 more static frames — CNN must NOT be called again
    for _ in range(10):
        state_action, classification, confidence = sys.process_frame(
            change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01
        )
        assert sys.inspection_frozen is True
        assert state_action == "Accepted"
        assert classification == "Fresh"
        
    assert sys.count_accepted == 1
    assert len(sys.spoken_texts) == 1
    assert sys.cnn_call_count == cnn_at_freeze  # No additional CNN calls

def test_pipeline_movement_triggers_new_inspection():
    """Verify that moving or rotating the object significantly unfreezes the state and triggers a new inspection."""
    sys = SimulatedQCSystem()
    
    # Static base ROI
    roi_static = np.zeros((10, 10), dtype=np.uint8)
    
    # Complete an inspection
    for _ in range(25):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0, crop_roi=roi_static)
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01, crop_roi=roi_static)
    for _ in range(5):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01, crop_roi=roi_static)
        
    assert sys.inspection_frozen is True
    assert sys.count_accepted == 1
    assert len(sys.spoken_texts) == 1
    
    # Now simulate a significant movement by providing a different crop_roi
    roi_moved = np.ones((10, 10), dtype=np.uint8) * 255
    
    # This frame detects a diff (100% changed > 8% threshold) -> triggers unfreeze & transitions to INSPECTING
    sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05, crop_roi=roi_moved)
    
    assert sys.inspection_frozen is False
    assert sys.qc_state == "INSPECTING"
    assert "Object Moved" in sys.debug_logs
    
    # Complete this new inspection (Rotten guava this time)
    for _ in range(5):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05, crop_roi=roi_moved)
        
    assert sys.qc_state == "REJECTED"
    assert sys.count_accepted == 1
    assert sys.count_rejected == 1
    assert len(sys.spoken_texts) == 2
    assert sys.spoken_texts == ["Fresh Guava Detected", "Rotten Guava Detected"]

def test_pipeline_rotten_guava_rejected():
    """Verify that a rotten guava is enqueued, inspected, and consistently rejected with one announcement."""
    sys = SimulatedQCSystem()
    
    # Warm up (26 frames)
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        
    # Object presence trigger (3 frames to transition WAITING -> OBJECT_DETECTED -> INSPECTING)
    for _ in range(2):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05)
    sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05)
    assert sys.qc_state == "INSPECTING"
    
    # Buffering (5 more frames)
    for _ in range(4):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05)
    
    state_action, classification, confidence = sys.process_frame(
        change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05
    )
    
    assert sys.qc_state == "REJECTED"
    assert state_action == "Rejected"
    assert classification == "Rotten"
    assert confidence > 90.0
    assert sys.count_rejected == 1
    assert sys.siren_triggered is True
    assert sys.spoken_texts == ["Rotten Guava Detected"]

def test_pipeline_rotten_blemish_override():
    """Verify that blemish ratio >= 0.08 forces a rotten classification override."""
    sys = SimulatedQCSystem()
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    
    # 3 frames to transition
    for _ in range(2):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.10, blemish_ratio=0.10)
    sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.10, blemish_ratio=0.10)
    assert sys.qc_state == "INSPECTING"
        
    for _ in range(4):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.10, blemish_ratio=0.10)
        
    sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.10, blemish_ratio=0.10)
    
    assert sys.qc_state == "REJECTED"
    assert sys.count_rejected == 1

def test_pipeline_mobile_phone_rejected():
    """Verify that a mobile phone (non-guava object) is flagged as NOT_A_GUAVA."""
    sys = SimulatedQCSystem()
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        
    # Phone enqueued: triggers presence but fails ImageNet verification (is_guava_imagenet = False)
    for _ in range(2):
        sys.process_frame(change_ratio=20.0, is_guava_color=True, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        
    # 3rd frame transitions to OBJECT_DETECTED and non_guava_frames becomes 1
    sys.process_frame(change_ratio=20.0, is_guava_color=True, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    assert sys.qc_state == "OBJECT_DETECTED"
    assert sys.non_guava_frames == 1
    
    # 3 more frames (non-guava count becomes 2, 3, 4)
    for _ in range(3):
        sys.process_frame(change_ratio=20.0, is_guava_color=True, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        assert sys.qc_state == "OBJECT_DETECTED"
        
    # 7th frame (non-guava count = 5) transitions to NOT_A_GUAVA & Frozen
    state_action, classification, confidence = sys.process_frame(
        change_ratio=20.0, is_guava_color=True, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0
    )
    
    assert sys.qc_state == "NOT_A_GUAVA"
    assert sys.inspection_frozen is True
    assert state_action == "Not a Guava"
    assert classification == "Unknown"
    assert sys.count_unknown == 1
    assert sys.spoken_texts == ["Unknown fruit detected"]

def test_pipeline_empty_roi_resets_to_waiting():
    """Verify that removing an object resets the state to WAITING_FOR_OBJECT after 15 empty frames."""
    sys = SimulatedQCSystem()
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        
    # Trigger object accepted state
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    for _ in range(6):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.01)
    assert sys.qc_state == "ACCEPTED"
    
    # Empty conveyor (15 frames)
    # The empty frames (change_ratio = 0.0) keep the state frozen in ACCEPTED
    for _ in range(14):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        assert sys.qc_state == "ACCEPTED"
        
    # The 15th empty frame triggers the overall timeout and resets state to WAITING_FOR_OBJECT
    sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    assert sys.qc_state == "WAITING_FOR_OBJECT"
    assert sys.current_inspection_id is None
    assert sys.inspection_logged is False
    assert len(sys.score_filter) == 0
    assert "Object Removed" in sys.debug_logs
    assert "Waiting For Object" in sys.debug_logs

def test_pipeline_consecutive_inspections():
    """Verify that consecutive items are detected, logged separately in history, and counted correctly."""
    sys = SimulatedQCSystem()
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
        
    # --- Item 1: Fresh Guava ---
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.0)
    for _ in range(6):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.0)
    assert sys.qc_state == "ACCEPTED"
    id_1 = sys.current_inspection_id
    
    # --- Empty conveyor (15 frames) ---
    for _ in range(15):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0)
    assert sys.qc_state == "WAITING_FOR_OBJECT"
    
    # --- Item 2: Rotten Guava (use blemish_ratio=0.05 so we do not trigger fresh override) ---
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05)
    for _ in range(6):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.95, blemish_ratio=0.05)
    assert sys.qc_state == "REJECTED"
    id_2 = sys.current_inspection_id
    
    assert id_1 != id_2
    assert sys.count_accepted == 1
    assert sys.count_rejected == 1
    assert len(sys.history) == 2
    assert sys.history[0]["prediction"] == "Fresh"
    assert sys.history[1]["prediction"] == "Rotten"
    assert sys.spoken_texts == ["Fresh Guava Detected", "Rotten Guava Detected"]

def test_pipeline_no_state_lockups():
    """Verify that multiple random state transitions do not cause the FSM to lock up."""
    sys = SimulatedQCSystem()
    roi_1 = np.zeros((10, 10), dtype=np.uint8)
    for _ in range(26):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0, crop_roi=roi_1)
        
    # Cycle 1: Fresh Guava
    for _ in range(3):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.0, crop_roi=roi_1)
    for _ in range(6):
        sys.process_frame(change_ratio=15.0, is_guava_color=True, is_guava_imagenet=True, cnn_score=0.05, blemish_ratio=0.0, crop_roi=roi_1)
    assert sys.qc_state == "ACCEPTED"
    
    # Cycle 2: Non-guava enters directly (swapped directly, so ROI changes, triggering unfreeze)
    roi_2 = np.ones((10, 10), dtype=np.uint8) * 255
    # First frame unfreezes
    sys.process_frame(change_ratio=20.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0, crop_roi=roi_2)
    assert sys.qc_state == "OBJECT_DETECTED"
    assert sys.non_guava_frames == 1
    
    for _ in range(4):
        sys.process_frame(change_ratio=20.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0, crop_roi=roi_2)
    assert sys.qc_state == "NOT_A_GUAVA"
    
    # Cycle 3: Conveyor cleared (15 frames of empty)
    for _ in range(15):
        sys.process_frame(change_ratio=0.0, is_guava_color=False, is_guava_imagenet=False, cnn_score=0.5, blemish_ratio=0.0, crop_roi=roi_2)
    assert sys.qc_state == "WAITING_FOR_OBJECT"
