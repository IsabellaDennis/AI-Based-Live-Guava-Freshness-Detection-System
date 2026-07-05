import os
import cv2
import numpy as np
import random
import time

class Camera:
    def __init__(self, device_id=0, force_simulation=False):
        """Initialize the video capture from the webcam or fall back to simulation mode."""
        self.device_id = device_id
        self.force_simulation = force_simulation
        
        # Try to open the webcam
        self.cap = cv2.VideoCapture(device_id)
        
        # Determine if we should run in simulation mode
        self.is_simulation = force_simulation or not self.cap.isOpened()
        
        if self.is_simulation:
            print("[CAMERA] Hardware camera offline or simulation forced. Initializing CONVEYOR SIMULATOR Mode.")
            self.setup_simulation()
        else:
            print(f"[CAMERA] Hardware camera online at device_id={device_id}. Running in LIVE HARDWARE Mode.")

    def setup_simulation(self):
        """Set up directories and parameters for simulated conveyor belt."""
        self.frame_index = 0
        self.width = 640
        self.height = 480
        
        self.fresh_images = []
        self.rotten_images = []
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(current_dir, "..", "dataset")
        
        # Scan dataset categories
        for category, target_list in [("Fresh", self.fresh_images), ("Rotten", self.rotten_images)]:
            cat_path = os.path.join(dataset_dir, category)
            if os.path.exists(cat_path):
                try:
                    files = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    target_list.extend(files)
                except Exception:
                    pass
                    
        # Object sequence type for each 300-frame cycle: 
        # 0: Fresh Guava, 1: Rotten Guava, 2: Phone (Non-Guava), 3: Plastic Bag (Non-Guava)
        self.cycle_object_type = 0
        self.cycle_image_path = None
        self.choose_next_cycle_object()

    def choose_next_cycle_object(self):
        """Pick a random object type and image for the next simulation cycle."""
        # Randomly choose object category
        self.cycle_object_type = random.choice([0, 1, 2, 3])
        self.cycle_image_path = None
        
        if self.cycle_object_type == 0:  # Fresh Guava
            if self.fresh_images:
                self.cycle_image_path = random.choice(self.fresh_images)
        elif self.cycle_object_type == 1:  # Rotten Guava
            if self.rotten_images:
                self.cycle_image_path = random.choice(self.rotten_images)

    def read(self):
        """Read a frame from the webcam (if online) or generate a simulated conveyor frame."""
        if not self.is_simulation:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally to act as a mirror
                frame = cv2.flip(frame, 1)
                return frame
            else:
                # Hardware read failed, transition to simulation dynamically
                print("[CAMERA] Hardware camera disconnected mid-run. Falling back to simulation.")
                self.is_simulation = True
                self.setup_simulation()
                
        # Governor to throttle simulation output rate to 20 FPS (50ms interval)
        now = time.monotonic()
        last_time = getattr(self, "_last_read_time", 0.0)
        elapsed = now - last_time
        target_interval = 0.05  # 20 FPS
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        self._last_read_time = time.monotonic()

        # Generate simulated conveyor frame
        return self.generate_simulated_frame()

    def generate_simulated_frame(self):
        """Synthesize a frame containing a scrolling conveyor belt background and moving objects."""
        # 1. Create conveyor belt background (dark gray)
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 32  # Dark gray base
        
        # Draw moving conveyor horizontal lines
        line_spacing = 160
        offset = (self.frame_index * 4) % line_spacing
        for y in range(-line_spacing, self.height + line_spacing, line_spacing):
            draw_y = y + offset
            if 0 <= draw_y < self.height:
                cv2.line(frame, (0, draw_y), (self.width, draw_y), (48, 52, 60), 4)
                
        # Draw conveyor belt borders (metal guides)
        cv2.rectangle(frame, (0, 0), (25, self.height), (70, 75, 85), -1)
        cv2.rectangle(frame, (self.width - 25, 0), (self.width, self.height), (70, 75, 85), -1)
        
        # Draw mechanical guide lines on the borders
        for y in range(0, self.height, 40):
            cv2.line(frame, (5, y), (20, y), (100, 105, 115), 2)
            cv2.line(frame, (self.width - 20, y), (self.width - 5, y), (100, 105, 115), 2)

        # 2. Compute object coordinates inside the conveyor cycle
        # A cycle is 300 frames. At frame 0 of the cycle, reset selection
        cycle_frame = self.frame_index % 300
        if cycle_frame == 0:
            self.choose_next_cycle_object()
            
        # Determine x-coordinate based on time step
        obj_y = self.height // 2
        
        if cycle_frame < 50:
            # Stage 1: Empty conveyor belt
            obj_x = -150
        elif cycle_frame < 90:
            # Stage 2: Object entering from left
            t = (cycle_frame - 50) / 40.0  # Interpolate 0.0 -> 1.0
            obj_x = int(-150 + t * (self.width // 2 + 150))
        elif cycle_frame < 210:
            # Stage 3: Object static in center for scanning (120 frames = 4 seconds at 30 fps)
            obj_x = self.width // 2
        elif cycle_frame < 260:
            # Stage 4: Object exiting to right
            t = (cycle_frame - 210) / 50.0  # Interpolate 0.0 -> 1.0
            obj_x = int(self.width // 2 + t * (self.width // 2 + 150))
        else:
            # Stage 5: Empty conveyor belt
            obj_x = self.width + 150

        # 3. Overlay the selected object at (obj_x, obj_y)
        if -100 < obj_x < self.width + 100:
            self.draw_simulated_object(frame, obj_x, obj_y)
            
        self.frame_index += 1
        return frame

    def draw_simulated_object(self, frame, x, y):
        """Draw either a loaded dataset image (guava) or a synthetic object (phone, plastic bag) onto the frame."""
        # 1. Draw loaded guava images
        if self.cycle_object_type in [0, 1] and self.cycle_image_path and os.path.exists(self.cycle_image_path):
            try:
                obj_img = cv2.imread(self.cycle_image_path)
                if obj_img is not None:
                    # Resize object image to fit nicely in ROI (e.g. 180x180 px)
                    obj_sz = 180
                    obj_img = cv2.resize(obj_img, (obj_sz, obj_sz))
                    
                    # Overlay using transparency chroma-keying for white/bright backgrounds
                    h, w, c = obj_img.shape
                    start_x = x - w // 2
                    start_y = y - h // 2
                    
                    for row in range(h):
                        for col in range(w):
                            pixel = obj_img[row, col]
                            # If not bright white (background), copy to conveyor frame
                            if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < 710:
                                target_y = start_y + row
                                target_x = start_x + col
                                if 0 <= target_y < self.height and 0 <= target_x < self.width:
                                    frame[target_y, target_x] = pixel
                    return
            except Exception:
                pass
                
        # 2. Fallback drawing if images are missing or if it's a non-guava object
        if self.cycle_object_type == 0:  # Fresh Guava backup (Green Sphere)
            cv2.circle(frame, (x, y), 70, (46, 204, 113), -1)  # Emerald green circle
            cv2.circle(frame, (x - 20, y - 20), 10, (120, 230, 150), -1)  # Specular highlight
            # Draw leaf
            cv2.ellipse(frame, (x + 30, y - 50), (25, 12), -45, 0, 360, (39, 174, 96), -1)
            
        elif self.cycle_object_type == 1:  # Rotten Guava backup (Brown/Yellow sphere with blemishes)
            cv2.circle(frame, (x, y), 70, (120, 170, 220), -1)  # Yellow-brown circle
            cv2.circle(frame, (x - 15, y + 10), 12, (30, 40, 50), -1)  # Black spot blemish 1
            cv2.circle(frame, (x + 20, y - 15), 8, (25, 35, 45), -1)  # Black spot blemish 2
            cv2.circle(frame, (x - 25, y - 20), 10, (140, 180, 230), -1)  # Specular highlight
            
        elif self.cycle_object_type == 2:  # Mobile Phone (Non-Guava Object)
            # Draw phone body (black rounded rectangle)
            phone_w, phone_h = 70, 130
            top_left = (x - phone_w // 2, y - phone_h // 2)
            bottom_right = (x + phone_w // 2, y + phone_h // 2)
            cv2.rectangle(frame, top_left, bottom_right, (33, 33, 33), -1)  # Dark chassis
            cv2.rectangle(frame, top_left, bottom_right, (66, 66, 66), 2)   # Bezel border
            # Screen area
            cv2.rectangle(frame, (x - 30, y - 55), (x + 30, y + 55), (200, 120, 0), -1)  # Blue shining screen
            # Camera notch
            cv2.circle(frame, (x, y - 60), 4, (10, 10, 10), -1)
            
        elif self.cycle_object_type == 3:  # Plastic Bag (Non-Guava Object)
            # Draw semi-transparent crumpled irregular polygon
            pts = np.array([
                [x - 60, y - 30], [x - 20, y - 70], [x + 40, y - 50],
                [x + 65, y + 10], [x + 20, y + 60], [x - 45, y + 40]
            ], dtype=np.int32)
            # Draw translucent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (230, 240, 245))
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            # Draw fold lines
            cv2.polylines(frame, [pts], True, (250, 252, 255), 2)
            cv2.line(frame, (x - 20, y - 70), (x + 20, y + 60), (240, 245, 248), 1)
            cv2.line(frame, (x - 45, y + 40), (x + 40, y - 50), (240, 245, 248), 1)

    def get_roi_details(self, frame, box_size=250):
        """Calculate ROI coordinates, crop the ROI region, and return boundaries."""
        h, w, c = frame.shape
        center_x, center_y = w // 2, h // 2
        
        half_size = box_size // 2
        x1 = center_x - half_size
        y1 = center_y - half_size
        x2 = center_x + half_size
        y2 = center_y + half_size
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop the ROI
        cropped_roi = frame[y1:y2, x1:x2].copy()
        display_frame = frame.copy()
        
        return cropped_roi, display_frame, (x1, y1, x2, y2)

    def release(self):
        """Release the camera capture resources."""
        if self.cap.isOpened():
            self.cap.release()