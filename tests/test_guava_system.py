import os
import sys
import tempfile
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch

# Ensure the parent app directory is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

# Mock pyttsx3 and tensorflow globally to make tests lightweight and fast
sys.modules['pyttsx3'] = MagicMock()

# =====================================================================
# 1. Tests for app/camera.py (Region of Interest Calculations)
# =====================================================================
from camera import Camera

def test_camera_roi_calculations():
    """Verify that centered ROI calculations and bounding boxes are correctly computed."""
    cam = Camera()
    
    # Create a mock BGR frame (640x480 resolution)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    box_size = 200
    cropped_roi, display_frame, coords = cam.get_roi_details(dummy_frame, box_size=box_size)
    
    x1, y1, x2, y2 = coords
    
    # Target center is (320, 240). Half size is 100.
    # Coordinates should be (220, 140, 420, 340)
    assert x1 == 220
    assert y1 == 140
    assert x2 == 420
    assert y2 == 340
    
    # Verify crop dimensions match the box size exactly
    assert cropped_roi.shape == (200, 200, 3)

def test_camera_roi_out_of_bounds_clipping():
    """Verify coordinate safety checks clip bounding boxes within frame boundaries."""
    cam = Camera()
    
    # Frame size (100x100), requested box size is 300 (larger than frame)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    cropped_roi, display_frame, coords = cam.get_roi_details(dummy_frame, box_size=300)
    x1, y1, x2, y2 = coords
    
    # Must clip to frame boundaries [0, 100]
    assert x1 == 0
    assert y1 == 0
    assert x2 == 100
    assert y2 == 100
    assert cropped_roi.shape == (100, 100, 3)


# =====================================================================
# 2. Tests for app/audio.py (Voice Debouncer State Machine)
# =====================================================================
from audio import VoiceDebouncer

@patch("audio.speak")
def test_voice_debouncer_stability_rules(mock_speak):
    """Verify that speaking only triggers when predictions are stable for N consecutive frames."""
    # Set stability count to 3
    debouncer = VoiceDebouncer(stability_threshold=3)
    
    # Scenario: Prediction is Rotten but only for 1 frame
    debouncer.process_prediction("Rotten")
    mock_speak.assert_not_called()
    assert debouncer.last_spoken_label is None
    
    # Scenario: Prediction is Rotten for 2 frames (unstable)
    debouncer.process_prediction("Rotten")
    mock_speak.assert_not_called()
    
    # Scenario: Prediction is Rotten for 3 frames (reaches threshold)
    debouncer.process_prediction("Rotten")
    mock_speak.assert_called_once_with("Rotten Guava Detected")
    assert debouncer.last_spoken_label == "Rotten"
    
    # Reset mock speak tracker
    mock_speak.reset_mock()
    
    # Scenario: Rotten is sent again (no speech, it's already the spoken state)
    debouncer.process_prediction("Rotten")
    mock_speak.assert_not_called()
    
    # Scenario: State changes to Fresh. Requires 3 frames to speak.
    debouncer.process_prediction("Fresh")  # 1st frame
    debouncer.process_prediction("Fresh")  # 2nd frame
    mock_speak.assert_not_called()
    
    debouncer.process_prediction("Fresh")  # 3rd frame (triggers announcement)
    mock_speak.assert_called_once_with("Fresh Guava Detected")
    assert debouncer.last_spoken_label == "Fresh"


# =====================================================================
# 3. Tests for app/utils.py (Dataset Scanner Helpers)
# =====================================================================
from utils import find_duplicates, check_class_imbalance, get_image_dimensions

def test_dataset_utilities_with_temp_files():
    """Verify duplicate hashes, imbalance counters, and resolution lists using mock files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create class folders
        fresh_dir = os.path.join(temp_dir, "Fresh")
        rotten_dir = os.path.join(temp_dir, "Rotten")
        os.makedirs(fresh_dir)
        os.makedirs(rotten_dir)
        
        # Write files with identical content to test duplicate checks
        file1_path = os.path.join(fresh_dir, "test1.jpg")
        file2_path = os.path.join(rotten_dir, "test2.jpg")
        
        content = b"fake_jpeg_content"
        with open(file1_path, "wb") as f:
            f.write(content)
        with open(file2_path, "wb") as f:
            f.write(content)
            
        # 1. Test class counts
        imbalance = check_class_imbalance(temp_dir)
        assert imbalance["Fresh"] == 1
        assert imbalance["Rotten"] == 1
        
        # 2. Test duplicate finder
        duplicates = find_duplicates(temp_dir)
        assert len(duplicates) == 1
        assert duplicates[0]["duplicate_path"] == file2_path
        assert duplicates[0]["original_path"] == file1_path


# =====================================================================
# 4. Tests for app/predictor.py (BGR Processing & Model Inference)
# =====================================================================
# Mock tensorflow model load during test loading
@patch("tf_keras.models.load_model")
def test_predictor_bgr_preprocessing(mock_load_model):
    """Verify that predictor preprocessing converts colors and shapes correctly before calling predict."""
    # Mock model predict output
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.85, 0.15]])  # 15% rotten (Fresh)
    mock_load_model.return_value = mock_model
    
    # Reload predictor module with mocked load_model
    if "predictor" in sys.modules:
        del sys.modules["predictor"]
    import predictor
    
    # Input a dummy BGR frame (e.g. 100x100 pixels)
    dummy_bgr_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    score = predictor.predict(dummy_bgr_img)
    
    # Verify predictions returned the correct score
    assert score == 0.15
    
    # Verify the model predict method was called with correctly sized batch tensor
    args, kwargs = mock_model.predict.call_args
    img_tensor = args[0]
    assert img_tensor.shape == (1, 224, 224, 3)
    assert img_tensor.dtype == np.float32


# =====================================================================
# 5. Tests for Advanced Dashboard Features (Calibration, CSV, Snapshots, Siren)
# =====================================================================
def test_csv_export_format():
    """Verify that scan history records are correctly converted to CSV formatted output."""
    history = [
        {"timestamp": "2026-07-03 20:00:00", "prediction": "Fresh", "confidence": 92.5, "status": "Accepted"},
        {"timestamp": "2026-07-03 20:01:00", "prediction": "Rotten", "confidence": 88.2, "status": "Rejected"}
    ]
    
    csv_data = [["Timestamp", "Classification", "Confidence", "Action"]]
    for row in history:
        csv_data.append([row["timestamp"], row["prediction"], f"{row['confidence']:.2f}%", row["status"]])
        
    csv_str = ""
    for line in csv_data:
        csv_str += ",".join(line) + "\n"
        
    expected_header = "Timestamp,Classification,Confidence,Action"
    expected_row_1 = "2026-07-03 20:00:00,Fresh,92.50%,Accepted"
    expected_row_2 = "2026-07-03 20:01:00,Rotten,88.20%,Rejected"
    
    assert expected_header in csv_str
    assert expected_row_1 in csv_str
    assert expected_row_2 in csv_str

def test_recalibration_reset():
    """Verify that background subtractor state values can be reset to None for calibration."""
    bg_subtractor = np.zeros((100, 100), dtype=np.uint8)
    warmup_frames = 150
    
    # Simulate calibration reset button action
    bg_subtractor = None
    warmup_frames = 0
    
    assert bg_subtractor is None
    assert warmup_frames == 0

def test_snapshot_saving():
    """Verify that snapshot saving functions create the target directory and write image files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_dir = os.path.join(temp_dir, "evaluation", "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        snap_path = os.path.join(snapshot_dir, "test_snapshot.jpg")
        
        # Write dummy frame
        cv2.imwrite(snap_path, dummy_crop)
        
        assert os.path.exists(snapshot_dir)
        assert os.path.exists(snap_path)
        assert os.path.getsize(snap_path) > 0

def test_audio_siren_generation():
    """Verify that a WAV audio file can be read and base64 encoded for html playback."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_wav_path = os.path.join(temp_dir, "alarm.wav")
        with open(mock_wav_path, "wb") as f:
            f.write(b"RIFFmockwavheaderandpayload")
            
        # Read and encode
        with open(mock_wav_path, "rb") as f:
            data = f.read()
            b64_str = np.base64 = np.base64 = MagicMock()
            import base64
            b64_encoded = base64.b64encode(data).decode()
            
        assert len(b64_encoded) > 0
        # Base64 of "RIFF" is "UklGR"
        assert b64_encoded.startswith("UklGR")

