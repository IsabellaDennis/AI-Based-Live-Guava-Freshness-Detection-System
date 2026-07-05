import sys
import threading
import queue
import pyttsx3

# Process-wide queue and thread caching to prevent pyttsx3 COM crash on reload
if not hasattr(sys, "_guava_speech_queue"):
    sys._guava_speech_queue = queue.Queue()
if not hasattr(sys, "_guava_browser_speech_queue"):
    sys._guava_browser_speech_queue = queue.Queue()
    
    def speech_worker():
        try:
            # Initialize COM in this worker thread for Windows compatibility
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
            
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
        except Exception as e:
            print(f"TTS Initialization error in thread: {e}")
            return

        while True:
            text = sys._guava_speech_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS execution error in thread: {e}")
            finally:
                sys._guava_speech_queue.task_done()

    sys._guava_tts_thread = threading.Thread(target=speech_worker, daemon=True)
    sys._guava_tts_thread.start()

speech_queue = sys._guava_speech_queue
browser_speech_queue = sys._guava_browser_speech_queue

def speak(text):
    """Enqueue a text string to be spoken in the background thread and browser queue."""
    speech_queue.put(text)
    browser_speech_queue.put(text)

class VoiceDebouncer:
    def __init__(self, stability_threshold=3):
        """
        Manages state announcements with stability checks to prevent repetitive speech.
        """
        self.stability_threshold = stability_threshold
        self.last_spoken_label = None
        self.current_candidate_label = None
        self.candidate_count = 0

    def process_prediction(self, label):
        """
        Announce label only if it shifts and stays stable for N consecutive frames.
        """
        # If label is already the current spoken state, reset candidate tracker
        if label == self.last_spoken_label:
            self.current_candidate_label = None
            self.candidate_count = 0
            return None

        # Track consecutive candidate frames
        if label == self.current_candidate_label:
            self.candidate_count += 1
        else:
            self.current_candidate_label = label
            self.candidate_count = 1

        # Check if stability threshold is reached
        if self.candidate_count >= self.stability_threshold:
            # Trigger speech based on class
            if label == "Fresh":
                speak("Fresh Guava Detected")
            elif label == "Rotten":
                speak("Rotten Guava Detected")
            elif label == "Unknown":
                speak("Unknown fruit detected")
            
            self.last_spoken_label = label
            self.current_candidate_label = None
            self.candidate_count = 0
            
    def reset(self):
        """Reset the speech debouncer state."""
        self.last_spoken_label = None
        self.current_candidate_label = None
        self.candidate_count = 0