# Thermal System Dashboard Object Detection with YOLOv5

## Project Overview
This project leverages YOLOv5 to perform object detection on thermal system dashboard images. The goal is to detect and extract numerical readings from the dashboard to monitor and analyze the company's heating performance. Additionally, an interface was developed to integrate the detection results with the company's data management system, enabling automated reporting and real-time monitoring.

## Key Features
- **Object Detection with YOLOv5**:
  - Detects numerical readings on thermal system dashboard images.
  - Extracts and processes detected numbers for analysis.
- **Data Integration**:
  - Developed an interface to integrate the detection results into the company's data system.
  - Supports real-time data collection and automated updates.
- **Performance Optimization**:
  - Optimized YOLOv5 model for high accuracy and fast inference on dashboard images.
- **Customizable Parameters**:
  - Supports tuning confidence and IoU thresholds for optimal detection performance.
- **Scalability**:
  - Designed for easy deployment in production, allowing the system to handle large batches of dashboard images.

## Dataset
- **Source**: Dashboard images provided by the thermal energy company.
- **Labels**: Numerical readings on the dashboard (e.g., temperature, pressure, flow rate).
- **Size**: Approximately 3000 images for training and 500 images for testing.
- **Annotation Tool**: Labeled images using [LabelImg](https://github.com/heartexlabs/labelImg).

## Methods
### 1. Object Detection with YOLOv5
- **Preprocessing**:
  - Converts images to RGB format and resizes them to match YOLOv5 input dimensions (`640x640` pixels).
  - Normalized pixel values and applied data augmentation techniques (e.g., random rotation, flipping) to improve robustness.
- **Training**:
  - Used YOLOv5 pre-trained weights as the base model.
  - Fine-tuned the model on the thermal dashboard dataset using PyTorch.
  - Conducted hyperparameter tuning (e.g., learning rate, batch size) for optimal performance.
- **Postprocessing**:
  - Applied confidence thresholding to filter low-confidence predictions.
  - Used Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
- **Evaluation**:
  - Evaluated the model using Precision, Recall, and Inference Time.

### 2. Data Integration Interface
- **Implementation**:
  - Developed a Python-based API using Flask to process detection results and integrate them into the company's data system.
  - The API supports:
    - **Input**: Dashboard images (uploaded via HTTP requests or batch processing).
    - **Output**: Detected numerical readings in JSON format.
  - Automated periodic data uploads for real-time updates and analytics.
- **Technologies Used**:
  - **Flask**: For API development.
  - **Pandas and NumPy**: For data processing and formatting.
  - **Integration**: Connected to the company's existing database for centralized data storage.

## Performance Metrics
| Metric          | Value          |
|-----------------|----------------|
| Precision       | 98%            |
| Recall          | 96%            |
| Optimal Confidence Threshold | 0.35 |
| Optimal IoU Threshold        | 0.5  |

## Setup Instructions
### Prerequisites
- Python 3.11
- GPU support (optional but recommended for training and inference):
  - **CUDA Toolkit**: Version compatible with your GPU and PyTorch version.
  - **NVIDIA Drivers**: Ensure your GPU drivers are up to date.
