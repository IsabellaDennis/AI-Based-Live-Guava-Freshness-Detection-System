# Guava QC Conveyor System - Real-World Validation Checklist

This checklist provides a systematic procedure to manually verify the complete inspection pipeline of the Guava Quality Control application on site or in staging.

## 📋 Pre-Test Preparation
1. Ensure the camera lens is clean and the conveyor area is illuminated with uniform lighting.
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the browser interface (defaults to `http://localhost:8501`).
4. Set the side menu parameters:
   - **Accept Threshold (Fresh Limit)**: `0.25`
   - **Reject Threshold (Rotten Limit)**: `0.75`
   - **Sensors Trigger Limit (Presence)**: `10.0`
   - **Voice Enabled**: Toggle `ON`
   - **Siren Enabled**: Toggle `ON`
   - **Enable Rejection Filters**: Toggle `ON`
5. Ensure the guide box (ROI area) is empty and click **RECALIBRATE BACKGROUND** to establish a baseline.

---

## 🔍 Validation Tests

### Test 1: Empty ROI & Idle System
- [ ] **Step**: Ensure no object is in the conveyor guide box.
- [ ] **Expectation**: 
  - The FSM state is **OFFLINE** when stopped, or **WAITING_FOR_OBJECT** when started.
  - The overlay reads `"SENSOR READY - PLACE GUAVA"`.
  - The dashboard is stable with zero flickering.

### Test 2: Fresh Guava Acceptance
- [ ] **Step**: Place a clean, green/yellow fresh guava inside the guide box.
- [ ] **Expectation**:
  - The conveyor detects the presence of the object within 3 frames (hud changes to `"OBJECT IN CONVEYOR AREA"`).
  - The overlay displays `"SCANNING LAYERS..."` as scores buffer in the background (6 frames).
  - The final state resolves to **ACCEPTED**.
  - The voice announces: `"Fresh Guava Detected"` **exactly once**.
  - The **ACCEPTED (FRESH)** count increases by 1.
  - The logs update correctly with a unique Inspection ID.

### Test 3: Rotten Guava Rejection
- [ ] **Step**: Place a guava with visible mold or dark spots inside the guide box.
- [ ] **Expectation**:
  - The system detects the object and inspects it.
  - The final state resolves to **REJECTED**.
  - The siren alarm plays (if toggled ON).
  - The voice announces: `"Rotten Guava Detected"` **exactly once**.
  - The **REJECTED (ROTTEN)** count increases by 1.
  - The log updates correctly.

### Test 4: Rejection Filter (Mobile Phone)
- [ ] **Step**: Place a mobile phone inside the guide box.
- [ ] **Expectation**:
  - The system detects motion/presence.
  - ImageNet verification fails (identifies it as a non-guava device, like `'cellular_telephone'`).
  - Within 5 frames, the system transitions to **NOT_A_GUAVA**.
  - The display reads `"WRONG OBJECT - NOT A GUAVA"`.
  - The voice announces: `"Unknown fruit detected"` **exactly once**.
  - The **UNKNOWN / CALIBRATING** count increases by 1.

### Test 5: Rejection Filter (Plastic Bag)
- [ ] **Step**: Place a clear or colored plastic bag inside the guide box.
- [ ] **Expectation**:
  - The system detects motion/presence.
  - Guava color filters fail (hue saturation fails to match green/yellow fruit tones).
  - Within 5 frames, the system transitions to **NOT_A_GUAVA**.
  - The display reads `"WRONG OBJECT - NOT A GUAVA"`.
  - The voice announces: `"Unknown fruit detected"` **exactly once**.

### Test 6: Empty ROI Reset (Object Removal)
- [ ] **Step**: Remove any object from the guide box.
- [ ] **Expectation**:
  - Within 15 empty frames (~0.5 seconds), the system transitions back to **WAITING_FOR_OBJECT**.
  - No residual alerts or duplicate voice triggers are played during reset.
  - The system is ready to scan the next item.

### Test 7: Consecutive Inspections separation
- [ ] **Step**: Place a fresh guava, wait for acceptance, remove it, wait for reset, then place a second fresh guava.
- [ ] **Expectation**:
  - The system creates two distinct logs in the history table.
  - The **ACCEPTED** counter increments by 2.
  - The voice plays exactly once for the first item, and exactly once for the second item, with no overlap or duplicates.

### Test 8: Dashboards Stability & Verification
- [ ] **Step**: Keep the live camera feed running for 5 minutes.
- [ ] **Expectation**:
  - The UI does not flicker or freeze.
  - The frame rate maintains a stable value based on your camera hardware (e.g. 10 - 30 FPS).
  - Clicking **EXPORT INSPECTION REPORT (.CSV)** downloads a valid CSV file with correct fields.
  - No delayed or queued voice announcements play after an object is cleared.
