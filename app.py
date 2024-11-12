import cv2
import numpy as np
from ultralytics import YOLO
import time
from runway_detection import RunwayDetection
from estimation import StateEstimator

class AirplaneTracker:
    def __init__(self):
        """
        AirplaneTracker class handles the tracking of the airplane in the image
        """
        self.last_position = None
        self.frames_since_detection = 0
        self.positions = []

    def update(self, new_position):
        """
        Update the position of the airplane based on the new detection
        Args:
            new_position: tuple of (x1, y1, x2, y2) coordinates of the detected airplane
        """
        if self.last_position is None:
            self.last_position = new_position
            self.positions = [new_position]
            self.frames_since_detection = 0
        else:
            distance = np.linalg.norm(np.array(new_position) - np.array(self.last_position))
            if distance < 300: 
                self.last_position = new_position
                self.positions.append(new_position)
                if len(self.positions) > 10:
                    self.positions.pop(0)
                self.frames_since_detection = 0
            else:
                self.frames_since_detection += 1

    def get_position(self):
        """
        Return the current position of the airplane
        Returns:
            last_position: tuple of (x1, y1, x2, y2) coordinates of the detected airplane
        """
        if self.frames_since_detection > 10:  
            return None
        return self.last_position

def draw_overlays(image, aircraft_state, state, frame_count, fps, elapsed_time):
    overlay_top_left = image.copy()
    cv2.rectangle(overlay_top_left, (10, 10), (200, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay_top_left, 0.6, image, 0.4, 0, image)
    cv2.putText(image, f"Frame: {frame_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"FPS: {fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 1)
    cv2.putText(image, f"Elapsed: {elapsed_time:.2f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (221, 230, 50), 1)

    overlay_top_right = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(overlay_top_right, (w - 310, 10), (w - 10, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay_top_right, 0.6, image, 0.4, 0, image)
    
    cv2.putText(image, f"Aircraft State Estimation", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if aircraft_state:
        # cv2.putText(image, f"Distance: {aircraft_state.distance:.2f}m", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (221, 230, 50), 2)
        cv2.putText(image, f"Altitude: {aircraft_state.altitude:.2f}m", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (221, 230, 50), 1)
        cv2.putText(image, f"Approach Angle: {aircraft_state.approach_angle:.1f}Deg", (w - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 1)
        cv2.putText(image, f"State: {state}", (w - 300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 1)
    else:
        cv2.putText(image, "...Waiting for aircraft detection", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (221, 230, 50), 1)

def main():
    # Initialize models and trackers
    runway_detector = RunwayDetection('model.keras')
    airplane_model = YOLO('yolo11s.pt') 
    tracker = AirplaneTracker()
    state_estimator = StateEstimator()

    # Open video capture
    cap = cv2.VideoCapture('aircraft.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output2B.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    frame_count = 0
    current_state = None
    aircraft_state = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame_with_runway, runway_mask = runway_detector.pipeline(frame)

        # YOLO11 detection
        results = airplane_model(frame_with_runway)
        
        # Process YOLO11 results - looking for airplane class (typically class 4 in COCO)
        if len(results) > 0:
            for r in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = r.cpu().numpy()
                print(f'int(cls): {int(cls)}, Confidence: {conf:.2f}')
                if int(cls) == 4 and conf > 0.5:  # Airplane class with confidence > 0.5
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    tracker.update((x1, y1, x2, y2))
                    tracked_position = tracker.get_position()

                    if tracked_position and runway_mask is not None:
                        x1, y1, x2, y2 = tracked_position
                        cv2.rectangle(frame_with_runway, (x1, y1 + 60), 
                                    (x2 - 60, y2), (153, 0, 153), 2)
                        
                        cv2.putText(frame_with_runway, f'airplane {conf:.2f}', 
                                  (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (153, 0, 153), 2)

                        # Update state estimation
                        current_time = time.time() - start_time
                        current_state, aircraft_state = state_estimator.update(
                            (x1, y1, x2, y2),
                            runway_mask,
                            current_time,
                            frame_with_runway.shape[0]
                        )
                    break  # Take only the first detected airplane

        elapsed_time = time.time() - start_time
        draw_overlays(frame_with_runway, aircraft_state, current_state, frame_count, fps, elapsed_time)

        out.write(frame_with_runway)
        cv2.namedWindow('Airplane Monitoring', cv2.WINDOW_NORMAL)
        cv2.imshow('Airplane Monitoring', frame_with_runway)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()