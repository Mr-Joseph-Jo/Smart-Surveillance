import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time

class MultiCameraTracker:
    def __init__(self, 
                 model_path: str = "yolo11n.pt",
                 reid_similarity_threshold: float = 0.7,
                 device: str = 'cpu'):
        
        # Initialize components
        self.yolo_model = YOLO(model_path)
        self.reid_model = PersonReID(device=device)
        self.global_tracker = GlobalTracker(self.reid_model, reid_similarity_threshold)
        self.camera_calibrator = CameraCalibrator()
        
        # Supervision trackers for each camera
        self.trackers = {
            1: sv.ByteTracker(),
            2: sv.ByteTracker()
        }
        
        # Visualization
        self.box_annotators = {
            1: sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1),
            2: sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        }
        
        self.label_annotators = {
            1: sv.LabelAnnotator(text_thickness=2, text_scale=1),
            2: sv.LabelAnnotator(text_thickness=2, text_scale=1)
        }
        
        # Camera feeds and processing
        self.camera_feeds = {1: None, 2: None}
        self.frame_queues = {1: queue.Queue(maxsize=10), 2: queue.Queue(maxsize=10)}
        self.result_queues = {1: queue.Queue(maxsize=10), 2: queue.Queue(maxsize=10)}
        
        # Control flags
        self.running = False
        self.threads = []
    
    def setup_cameras(self, camera1_source, camera2_source):
        """
        Setup camera feeds
        camera sources can be: 0, 1 (webcam indices) or 'rtsp://...' (IP cameras) or video file paths
        """
        self.camera_feeds[1] = cv2.VideoCapture(camera1_source)
        self.camera_feeds[2] = cv2.VideoCapture(camera2_source)
        
        # Set camera properties if needed
        for cam_id in [1, 2]:
            if self.camera_feeds[cam_id].isOpened():
                self.camera_feeds[cam_id].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera_feeds[cam_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera_feeds[cam_id].set(cv2.CAP_PROP_FPS, 30)
    
    def calibrate_cameras(self, method='auto'):
        """
        Calibrate cameras to find homography
        method: 'auto' for automatic SIFT-based, 'manual' for manual point selection
        """
        # Capture frames for calibration
        ret1, frame1 = self.camera_feeds[1].read()
        ret2, frame2 = self.camera_feeds[2].read()
        
        if not ret1 or not ret2:
            raise ValueError("Could not read from cameras")
        
        print("Starting camera calibration...")
        
        if method == 'auto':
            try:
                homography = self.camera_calibrator.find_homography_interactive(frame1, frame2)
                print("Automatic calibration successful!")
            except ValueError as e:
                print(f"Automatic calibration failed: {e}")
                print("Falling back to manual calibration...")
                homography = self.camera_calibrator.manual_point_selection(frame1.copy(), frame2.copy())
        else:
            homography = self.camera_calibrator.manual_point_selection(frame1.copy(), frame2.copy())
        
        # Define overlap regions
        self.camera_calibrator.define_overlap_regions(frame1.shape, frame2.shape)
        print("Camera calibration completed!")
        
        return homography
    
    def camera_thread(self, camera_id: int):
        """
        Thread function for camera capture and processing
        """
        while self.running:
            ret, frame = self.camera_feeds[camera_id].read()
            if not ret:
                continue
            
            try:
                # Add to frame queue (non-blocking)
                self.frame_queues[camera_id].put_nowait(frame)
            except queue.Full:
                # Skip frame if queue is full
                pass
    
    def detection_thread(self, camera_id: int):
        """
        Thread function for YOLO detection and tracking
        """
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queues[camera_id].get(timeout=1)
            except queue.Empty:
                continue
            
            # YOLO detection
            results = self.yolo_model(frame, classes=[0])  # class 0 is 'person'
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Apply confidence threshold
            detections = detections[detections.confidence > 0.5]
            
            # Local tracking with ByteTracker
            detections = self.trackers[camera_id].update_with_detections(detections)
            
            # Prepare detection data for global tracker
            detection_data = []
            if detections.tracker_id is not None:
                for i, (bbox, track_id, conf) in enumerate(zip(
                    detections.xyxy, detections.tracker_id, detections.confidence)):
                    detection_data.append({
                        'local_id': int(track_id),
                        'bbox': bbox,
                        'confidence': float(conf)
                    })
            
            # Update global tracker
            local_to_global = self.global_tracker.update_tracks(camera_id, detection_data, frame)
            
            # Create result
            result = {
                'frame': frame,
                'detections': detections,
                'local_to_global': local_to_global,
                'timestamp': time.time()
            }
            
            try:
                # Add to result queue
                self.result_queues[camera_id].put_nowait(result)
            except queue.Full:
                # Skip if queue is full
                pass
    
    def visualize_frame(self, camera_id: int, frame: np.ndarray, 
                       detections: sv.Detections, local_to_global: Dict[int, int]) -> np.ndarray:
        """
        Visualize detections with global IDs
        """
        annotated_frame = frame.copy()
        
        if detections.tracker_id is not None and len(detections) > 0:
            # Create labels with global IDs
            labels = []
            for local_id in detections.tracker_id:
                global_id = local_to_global.get(int(local_id), local_id)
                labels.append(f"Person G{global_id} L{local_id}")
            
            # Annotate
            annotated_frame = self.box_annotators[camera_id].annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotators[camera_id].annotate(
                scene=annotated_frame, detections=detections, labels=labels)
        
        # Add camera info
        cv2.putText(annotated_frame, f"Camera {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add statistics
        stats = self.global_tracker.get_statistics()
        stats_text = f"Global IDs: {stats['total_global_ids']}, Active: C1={stats['active_tracks_cam1']}, C2={stats['active_tracks_cam2']}"
        cv2.putText(annotated_frame, stats_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated_frame
    
    def start_tracking(self):
        """
        Start the multi-camera tracking system
        """
        self.running = True
        
        # Start camera threads
        for camera_id in [1, 2]:
            camera_thread = threading.Thread(target=self.camera_thread, args=(camera_id,))
            camera_thread.daemon = True
            camera_thread.start()
            self.threads.append(camera_thread)
        
        # Start detection threads
        for camera_id in [1, 2]:
            detection_thread = threading.Thread(target=self.detection_thread, args=(camera_id,))
            detection_thread.daemon = True
            detection_thread.start()
            self.threads.append(detection_thread)
        
        print("Multi-camera tracking started!")
        print("Press 'q' to quit, 's' to save current frames, 'r' to recalibrate")
        
        # Main visualization loop
        while self.running:
            frames_to_show = {}
            
            # Get latest results from both cameras
            for camera_id in [1, 2]:
                try:
                    result = self.result_queues[camera_id].get_nowait()
                    annotated_frame = self.visualize_frame(
                        camera_id, result['frame'], 
                        result['detections'], result['local_to_global']
                    )
                    frames_to_show[camera_id] = annotated_frame
                except queue.Empty:
                    continue
            
            # Display frames
            for camera_id, frame in frames_to_show.items():
                cv2.imshow(f'Camera {camera_id} - Multi-Camera Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frames
                timestamp = int(time.time())
                for camera_id, frame in frames_to_show.items():
                    cv2.imwrite(f'camera_{camera_id}_{timestamp}.jpg', frame)
                print(f"Frames saved with timestamp {timestamp}")
            elif key == ord('r'):
                # Recalibrate cameras
                print("Recalibrating cameras...")
                try:
                    self.calibrate_cameras()
                    print("Recalibration completed!")
                except Exception as e:
                    print(f"Recalibration failed: {e}")
        
        self.stop_tracking()
    
    def stop_tracking(self):
        """
        Stop the tracking system
        """
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Release cameras
        for camera_feed in self.camera_feeds.values():
            if camera_feed:
                camera_feed.release()
        
        cv2.destroyAllWindows()
        print("Multi-camera tracking stopped!")

# Usage example
def main():
    # Initialize the multi-camera tracker
    tracker = MultiCameraTracker(
        model_path="yolo11n.pt",  # Make sure you have this model
        reid_similarity_threshold=0.6,  # Adjust based on your needs
        device='cpu'  # or 'cuda' if you have GPU
    )
    
    # Setup cameras (adjust camera sources as needed)
    # For webcams: use 0, 1, 2, etc.
    # For IP cameras: use 'rtsp://username:password@ip:port/stream'
    # For video files: use file paths
    tracker.setup_cameras(0, 1)  # Using first two webcams
    
    try:
        # Calibrate cameras
        tracker.calibrate_cameras(method='auto')  # or 'manual'
        
        # Start tracking
        tracker.start_tracking()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracker.stop_tracking()

if __name__ == "__main__":
    main()