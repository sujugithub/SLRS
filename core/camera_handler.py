"""
Camera handler for the Sign Language Recognition App.

Responsible for:
- Opening and releasing the webcam via OpenCV
- Capturing frames from the video feed
- Converting frames between color spaces (BGR to RGB)
- Providing a clean interface for other modules to consume camera frames
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class CameraHandler:
    """Manages webcam capture and frame conversion for tkinter display."""

    def __init__(self, camera_index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        """
        Args:
            camera_index: OpenCV camera device index.
            width: Desired frame width.
            height: Desired frame height.
        """
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._cap = None
        self._running = False

    def start(self):
        """Open the camera and begin capture.

        Tries the configured camera index first, then falls back to indices
        0 and 1 so the app works on machines where the built-in webcam is
        not at index 2.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._running and self._cap is not None:
            return True

        # Try configured index first, then fall back to 0 and 1.
        indices = list(dict.fromkeys([self._camera_index, 0, 1, 2]))

        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                if idx != self._camera_index:
                    print(f"[CameraHandler] Index {self._camera_index} "
                          f"unavailable — using camera {idx}")
                self._camera_index = idx
                self._cap = cap
                break
            cap.release()
        else:
            self._cap = None
            self._running = False
            print("[CameraHandler] ERROR: no camera found on any index")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._running = True
        print(f"[CameraHandler] Opened camera {self._camera_index} "
              f"({self._width}x{self._height})")
        return True

    def stop(self):
        """Release the camera."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self):
        """Read a single frame from the camera.

        Returns:
            A BGR numpy array, or None if the camera is not running or
            the read failed.
        """
        if not self._running or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        # Mirror for natural selfie-view
        return cv2.flip(frame, 1)

    def get_frame_for_display(self):
        """Read a frame and convert it to a PIL ImageTk.PhotoImage.

        Returns:
            A tuple (photo_image, bgr_frame) where photo_image is suitable
            for a tkinter Canvas/Label and bgr_frame is the raw BGR numpy
            array. Returns (None, None) on failure.
        """
        frame = self.get_frame()
        if frame is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        return photo, frame

    def capture_image(self):
        """Capture and return the current frame.

        Returns:
            A BGR numpy array snapshot, or None if unavailable.
        """
        return self.get_frame()

    def is_running(self):
        """Return True if the camera is actively capturing."""
        return self._running and self._cap is not None

    def __del__(self):
        """Ensure camera is released on garbage collection."""
        self.stop()


# ----------------------------------------------------------------- test
if __name__ == "__main__":
    import tkinter as tk

    print("=== CameraHandler Test ===")
    print(f"Camera index: {CAMERA_INDEX}")
    print(f"Resolution  : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    handler = CameraHandler()
    if not handler.start():
        print("ERROR: Could not open camera. Exiting.")
        sys.exit(1)

    print(f"Camera opened: is_running={handler.is_running()}")

    # Quick headless check — grab a frame
    frame = handler.get_frame()
    if frame is not None:
        print(f"Frame shape : {frame.shape}")
        print(f"Frame dtype : {frame.dtype}")
    else:
        print("WARNING: get_frame() returned None")

    captured = handler.capture_image()
    if captured is not None:
        print(f"Capture OK  : {captured.shape}")
    else:
        print("WARNING: capture_image() returned None")

    # --- tkinter live preview ---
    print("\nOpening tkinter preview window …")
    print("Press 'q' key or close the window to exit.\n")

    root = tk.Tk()
    root.title("Camera Handler Test")
    root.configure(bg="#1e1e2e")

    canvas = tk.Canvas(root, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg="#3a3a4e")
    canvas.pack(padx=10, pady=10)

    status_label = tk.Label(
        root, text="Starting …", fg="#e2e8f0", bg="#1e1e2e",
        font=("Helvetica", 12),
    )
    status_label.pack(pady=(0, 10))

    photo_ref = [None]  # prevent GC

    def update_frame():
        photo, bgr = handler.get_frame_for_display()
        if photo is not None:
            photo_ref[0] = photo  # prevent GC
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            h, w = bgr.shape[:2]
            status_label.config(text=f"Live  |  {w}x{h}  |  Camera {CAMERA_INDEX}")
        else:
            status_label.config(text="No frame")
        root.after(30, update_frame)  # ~33 fps

    def on_close():
        handler.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<q>", lambda e: on_close())

    update_frame()
    root.mainloop()

    handler.stop()
    print("Camera released. Test complete.")
