import numpy as np
import cv2
from skimage.transform import resize
import tensorflow as tf
import torch
import time

# Load the runway detection model
runway_model = tf.keras.models.load_model('model.keras')

# Initialize YOLOv5 model for airplane detection
airplane_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class Runways():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

runways = Runways()

def runway_lines(image):
    original_shape = image.shape
    new_shape = (160, 320)
    small_img = resize(image, new_shape)
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = runway_model.predict(small_img)[0] * 255
    runways.recent_fit.append(prediction)

    if len(runways.recent_fit) > 5:
        runways.recent_fit = runways.recent_fit[1:]

    runways.avg_fit = np.mean(runways.recent_fit, axis=0)

    if len(runways.avg_fit.shape) == 3:
        runways.avg_fit = np.mean(runways.avg_fit, axis=2)

    runway_drawn = np.zeros((runways.avg_fit.shape[0], runways.avg_fit.shape[1], 3), dtype=np.uint8)
    runway_drawn[:, :, 1] = runways.avg_fit.astype(np.uint8)

    runway_image = cv2.resize(runway_drawn, (original_shape[1], original_shape[0]))

    result = cv2.addWeighted(image, 1, runway_image, 1, 0)

    return result, runway_image

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height, width = image.shape
    polygon = np.array([[(width//4, height), (3*width//4, height), (3*width//4, height//3), (width//4, height//3)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def hough_lines(image):
    edges = detect_edges(image)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, np.array([]), minLineLength=100, maxLineGap=50)
    line_image = draw_lines(image, lines)
    return line_image

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def pipeline(image):
    try:
        runway_image, runway_mask = runway_lines(image)
        hough_image = hough_lines(image)
        
        if len(hough_image.shape) == 2:
            hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
        
        result = weighted_img(hough_image, runway_image)
        return result, runway_mask
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return image, None

def estimate_distance(airplane_bottom, runway_mask, frame_count, distance_count):
    runway_bottom = np.argmax(runway_mask[:, :, 1], axis=0).max()
    runway_top = np.argmax(runway_mask[:, :, 1], axis=0).min()
    runway_center = (runway_bottom + runway_top) / 2
    
    pixels_per_meter = runway_bottom / 300  # Assume runway is 200 meters long
    distance_to_runway_center = abs(airplane_bottom - runway_center) / pixels_per_meter
    total_dist = distance_count - distance_to_runway_center
    if frame_count >= 420:
        total_dist = 0
    return max(0, total_dist)

class AirplaneTracker:
    def __init__(self):
        self.last_position = None
        self.frames_since_detection = 0
        self.positions = []

    def update(self, new_position):
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
        if self.frames_since_detection > 10:  
            return None
        return self.last_position
    
    def is_landing(self):
        if len(self.positions) < 2:
            return None
        return self.positions[-1][1] < self.positions[0][1]

tracker = AirplaneTracker()

def determine_state(y2, runway_mask, tracker, frame_count):
    if tracker.is_landing() is None:
        return "Landing"
    elif tracker.is_landing():
        return "Landing"
    else:
        return "On runway" if frame_count >= 420 else "Approaching"

def draw_overlays(image, distance, state, frame_count, fps, elapsed_time):
    overlay_top_left = image.copy()
    cv2.rectangle(overlay_top_left, (10, 10), (200, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay_top_left, 0.6, image, 0.4, 0, image)
    cv2.putText(image, f"Frame: {frame_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f"FPS: {fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 2)
    cv2.putText(image, f"Elapsed: {elapsed_time:.2f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 153), 2)

    overlay_top_right = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(overlay_top_right, (w - 210, 10), (w - 10, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay_top_right, 0.6, image, 0.4, 0, image)
    cv2.putText(image, f"Estimation", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if distance == "...":
        cv2.putText(image, f"Distance: {distance}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 153), 2)
    else:
        cv2.putText(image, f"Distance: {distance:.2f}m", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 153), 2)

    cv2.putText(image, f"State: {state}", (w - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 204), 2)

def non_max_suppression(boxes, scores, threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

# Open video capture
cap = cv2.VideoCapture('video.mp4')  

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output2.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()
frame_count = 0
distance_count = 500

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    distance_count -= 1

    frame_with_runway, runway_mask = pipeline(frame)

    results = airplane_model(frame_with_runway)

    distance = "..."
    state = "..."

    boxes = []
    scores = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 4 and conf > 0.5:  # Airplane class with confidence > 0.5
            boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            scores.append(conf.item())

    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, 0.5)

        if len(indices) > 0:
            best_box = boxes[indices[0]]
            best_score = scores[indices[0]]

            if best_box.ndim == 2 and best_box.shape[0] == 1:
                best_box = best_box[0]
            
            if len(best_box) == 4:
                x1, y1, x2, y2 = map(int, best_box)

                tracker.update((x1, y1, x2, y2))
                tracked_position = tracker.get_position()

                if tracked_position:
                    x1, y1, x2, y2 = tracked_position
                    cv2.rectangle(frame_with_runway, (x1, y1 + 60), (x2 - 60, y2), (204, 102, 0), 2)
                    
                    conf = best_score.item() if isinstance(best_score, np.ndarray) else best_score
                    cv2.putText(frame_with_runway, f'airplane {conf:.2f}', (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 102, 0), 2)

                    if runway_mask is not None:
                        distance = estimate_distance(y2, runway_mask, frame_count, distance_count)
                        state = determine_state(y2, runway_mask, tracker, frame_count)

            else:
                print("Unexpected shape of best_box")

    elapsed_time = time.time() - start_time

    draw_overlays(frame_with_runway, distance, state, frame_count, fps, elapsed_time)
    # if distance is not None and state is not None:
    #     draw_overlays(frame_with_runway, distance, state, frame_count, fps, elapsed_time)

    out.write(frame_with_runway)

    cv2.namedWindow('Airplane Monitoring', cv2.WINDOW_NORMAL)
    cv2.imshow('Airplane Monitoring', frame_with_runway)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
