import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2

@dataclass
class AircraftState:
    distance: float
    vertical_speed: float
    horizontal_speed: float
    altitude: float
    approach_angle: float
    landing_confidence: float

class StateEstimator:
    def __init__(self, runway_length_meters=3000, runway_width_meters=45):
        """
        StateEstimator class handles the estimation of the aircraft state based on the detected position.
        Args:
            runway_length_meters: Length of the runway in meters
            runway_width_meters: Width of the runway in meters
        """
        self.runway_length = runway_length_meters
        self.runway_width = runway_width_meters
        self.history: List[Tuple[float, float, float, float]] = []  # x, y, timestamp, altitude
        self.states: List[str] = []
        self.kalman = cv2.KalmanFilter(6, 3)  # 6 states (x, y, z, dx, dy, dz), 3 measurements (x, y, z)
        self._setup_kalman()
        
    def _setup_kalman(self):
        # State transition matrix
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        # Process noise
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1

    def estimate_altitude(self, aircraft_box: tuple, runway_mask) -> float:
        """Estimate aircraft altitude based on its position relative to runway."""
        _, y1, _, y2 = aircraft_box
        aircraft_bottom = y2
        
        # Find runway vanishing point
        runway_points = np.column_stack(np.where(runway_mask[:, :, 1] > 0))
        if len(runway_points) < 2:
            return 0.0
            
        # Fit line to runway points to find vanishing point
        vx, vy, x0, y0 = cv2.fitLine(runway_points, cv2.DIST_L2, 0, 0.01, 0.01)
        vanishing_point_y = y0[0]
        
        # Calculate relative altitude based on vertical position
        max_altitude = 1000  # meters
        relative_height = (vanishing_point_y - aircraft_bottom) / runway_mask.shape[0]
        estimated_altitude = max_altitude * relative_height
        
        return max(0, estimated_altitude)

    def estimate_distance(self, aircraft_box: tuple, runway_mask) -> float:
        """Estimate distance to touchdown point using perspective and runway markers."""
        x1, y1, x2, y2 = aircraft_box
        aircraft_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Find runway centerline
        runway_mask_gray = runway_mask[:, :, 1]
        runway_bottom = np.max(np.where(runway_mask_gray > 0)[0])
        runway_top = np.min(np.where(runway_mask_gray > 0)[0])
        
        # Calculate perspective scaling factor
        perspective_scale = (runway_bottom - runway_top) / self.runway_length
        
        # Find touchdown point (assumed to be 300m from runway threshold)
        touchdown_y = runway_bottom - (300 * perspective_scale)
        
        # Calculate distance based on vertical position difference
        distance = abs(aircraft_center[1] - touchdown_y) / perspective_scale
        
        return max(0, distance)

    def calculate_speeds(self, current_pos: tuple, current_time: float) -> Tuple[float, float]:
        """Calculate vertical and horizontal speeds using recent position history."""
        if len(self.history) < 2:
            return 0.0, 0.0
            
        prev_x, prev_y, prev_time, _ = self.history[-1]
        current_x, current_y = current_pos
        
        time_diff = current_time - prev_time
        if time_diff == 0:
            return 0.0, 0.0
            
        vertical_speed = (prev_y - current_y) / time_diff  # pixels per second
        horizontal_speed = abs(current_x - prev_x) / time_diff
        
        return vertical_speed, horizontal_speed

    def calculate_approach_angle(self, altitude: float, distance: float) -> float:
        """Calculate approach angle in degrees."""
        if distance == 0:
            return 0.0
        return np.degrees(np.arctan2(altitude, distance))

    def determine_state(self, aircraft_state: AircraftState) -> str:
        """Determine aircraft state based on multiple parameters."""
        # Initialize state confidence scores
        state_scores = {
            'Final Approach': 0,
            'Landing': 0,
            'Touchdown': 0,
            'Ground Roll': 0
        }
        
        # Evaluate approach angle
        if 2.5 <= aircraft_state.approach_angle <= 3.5:  # Ideal 3-degree approach
            state_scores['Final Approach'] += 1.0
        elif aircraft_state.approach_angle < 2.5:
            state_scores['Landing'] += 0.8
            
        # Evaluate altitude
        if aircraft_state.altitude > 100:
            state_scores['Final Approach'] += 0.8
        elif 0 < aircraft_state.altitude <= 100:
            state_scores['Landing'] += 1.0
        else:
            state_scores['Ground Roll'] += 1.0
            
        # Evaluate vertical speed
        if aircraft_state.vertical_speed < -2:  # Descending
            state_scores['Final Approach'] += 0.6
        elif -2 <= aircraft_state.vertical_speed <= -0.5:  # Flare
            state_scores['Landing'] += 1.0
        elif aircraft_state.vertical_speed > -0.5:  # Level or climbing
            state_scores['Ground Roll'] += 0.8
            
        # Get the state with highest confidence
        current_state = max(state_scores.items(), key=lambda x: x[1])[0]
        
        # Add hysteresis to prevent state oscillation
        if len(self.states) > 0:
            prev_state = self.states[-1]
            if prev_state == current_state:
                return current_state
            elif state_scores[current_state] - state_scores[prev_state] < 0.3:
                return prev_state
                
        self.states.append(current_state)
        if len(self.states) > 10:
            self.states.pop(0)
            
        return current_state

    def update(self, aircraft_box: tuple, runway_mask, timestamp: float, image_height: int) -> Tuple[str, AircraftState]:
        """
        Update state estimation with new aircraft position
        Args:
            aircraft_box: Bounding box coordinates of the aircraft
            runway_mask: Mask of the runway area
            timestamp: Timestamp of the frame
            image_height: Height of the image frame
        Returns:
            current_state: Current state of the aircraft
            aircraft_state: Aircraft state estimation
        """
        x1, y1, x2, y2 = aircraft_box
        aircraft_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Estimate basic parameters
        distance = self.estimate_distance(aircraft_box, runway_mask)
        altitude = self.estimate_altitude(aircraft_box, runway_mask)
        vertical_speed, horizontal_speed = self.calculate_speeds(aircraft_center, timestamp)
        approach_angle = self.calculate_approach_angle(altitude, distance)
        
        # Update position history
        self.history.append((aircraft_center[0], aircraft_center[1], timestamp, altitude))
        if len(self.history) > 30:  # Keep last 30 positions
            self.history.pop(0)
            
        # Create aircraft state
        aircraft_state = AircraftState(
            distance=distance,
            vertical_speed=vertical_speed,
            horizontal_speed=horizontal_speed,
            altitude=altitude,
            approach_angle=approach_angle,
            landing_confidence=0.0  # TODO: Implement confidence calculation
        )
        
        # Determine current state
        current_state = self.determine_state(aircraft_state)
        
        return current_state, aircraft_state

    def reset(self):
        """Reset the estimator state."""
        self.history.clear()
        self.states.clear()
        self.kalman = cv2.KalmanFilter(6, 3)
        self._setup_kalman()