import numpy as np
import cv2
from skimage.transform import resize
import tensorflow as tf

class RunwayDetection:
    def __init__(self, model_path='model.keras'):
        """
        RunwayDetection class handles the detection of the runway in a frame
        Args:
            model_path: Path to the keras model
        """
        self.runway_model = tf.keras.models.load_model(model_path)
        self.recent_fit = []
        self.avg_fit = []

    def runway_lines(self, image):
        """
        Detect the runway in the frame using the trained model"""
        original_shape = image.shape
        new_shape = (160, 320)
        small_img = resize(image, new_shape)
        small_img = np.array(small_img)
        small_img = small_img[None, :, :, :]

        prediction = self.runway_model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(self.recent_fit, axis=0)

        if len(self.avg_fit.shape) == 3:
            self.avg_fit = np.mean(self.avg_fit, axis=2)

        runway_drawn = np.zeros((self.avg_fit.shape[0], self.avg_fit.shape[1], 3), dtype=np.uint8)
        runway_drawn[:, :, 1] = self.avg_fit.astype(np.uint8)

        runway_image = cv2.resize(runway_drawn, (original_shape[1], original_shape[0]))
        result = cv2.addWeighted(image, 1, runway_image, 1, 0)

        return result, runway_image

    def detect_edges(self, image):
        """
        Detect edges in the frame using Canny edge detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, image):
        """
        Define the region of interest for runway detection
        """
        height, width = image.shape
        polygon = np.array([[(width//4, height), (3*width//4, height), (3*width//4, height//3), (width//4, height//3)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def draw_lines(self, image, lines):
        """
        Draw detected lines on the frame
        """
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def hough_lines(self, image):
        """
        Detect lines using Hough transform
        """
        edges = self.detect_edges(image)
        roi = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, np.array([]), minLineLength=100, maxLineGap=50)
        line_image = self.draw_lines(image, lines)
        return line_image

    def weighted_img(self, img, initial_img, α=0.8, β=1., γ=0.):
        """
        Combine frames using weighted addition
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def pipeline(self, image):
        """
        Run the complete pipeline to detect the runway in the frame
        """
        try:
            runway_image, runway_mask = self.runway_lines(image)
            hough_image = self.hough_lines(image)
            
            if len(hough_image.shape) == 2:
                hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
            
            result = self.weighted_img(hough_image, runway_image)
            return result, runway_mask
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return image, None



