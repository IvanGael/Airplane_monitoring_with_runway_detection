## Airplane Runway Detection and Tracking
This project detects and tracks airplane approaching and landing on a runway using a combination of deep learning model and image processing techniques. It processes video frames to identify runways, detect airplane, estimate distances, and determine airplane states. 

![Demo](https://github.com/user-attachments/assets/25c9f387-a1da-4b7a-8887-435a03b435d0)

### Features
- Runway line detection using a pre-trained TensorFlow model.
- Airplane detection using YOLO11.
- State(Landing, Taking Off, On Runway, Approaching) and distance estimation based on airplane position and runway proximity.

 
### Requirements
````
pip install -r requirements.txt
````