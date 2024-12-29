# OpenLane-V2 - Research Project

This project utilizes the OpenLane V2 dataset for advanced lane detection in autonomous driving scenarios. OpenLane V2 is a comprehensive dataset that provides rich, multi-modal data for various autonomous driving tasks, with a focus on complex urban environments.

This repository contains an enhanced implementation of three key tasks using the OpenLane-V2 dataset. These tasks are essential components for autonomous driving systems, focusing on lane topology detection, object detection, and centerline prediction. The repository also integrates the original OpenLaneV2 repository for a seamless workflow.

## Task Description

The main task of this project is to detect and predict 3D centerlines of lanes in road scenes. This is crucial for autonomous driving systems, as accurate lane detection is essential for proper vehicle positioning and navigation. The project involves:

1. Processing and preparing the OpenLane-V2 dataset
2. Implementing a Vision Transformer (ViT) model for 3D centerline prediction
3. Evaluating the model using Fréchet distance and Chamfer distance metrics
4. Visualizing the projected centerlines on images

## Table of Contents
- [Overview](#overview)
- [Tasks Implemented](#tasks-implemented)
  - [Task 1: Lane-Lane Topology Detection](#task-1-lane-lane-topology-detection)
  - [Task 2: Traffic Signs Detection](#task-2-traffic-signs-detection)
  - [Task 3: Centerline Detection](#task-3-centerline-detection)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository builds upon the OpenLane-V2 dataset to implement three advanced tasks in autonomous driving:
1. **Lane-Lane Topology Detection**: Understanding the relationships between adjacent lanes.
2. **Traffic Signs Detection**: Identifying and localizing traffic signs.
3. **Centerline Detection**: Predicting the 3D centerline of lanes for navigation.

By combining cutting-edge computer vision techniques with deep learning, this project provides robust solutions for these tasks. The models are trained and evaluated on real-world driving scenarios, ensuring practical applicability.

---

## Tasks Implemented

### Task 1: Lane-Lane Topology Detection
- **Objective**: Detect and predict lane connectivity and relationships.
- **Approach**: 
  - Utilized Multi Layer Perceptron (ANN), Recurrent Neural Network (RNN, LSTM), Convolutional Neural Networks (CNNs) to process lane topology features.
  - Designed a graph-based representation to model lane adjacency.
- **Notebook**: [Task-1-Lane-Lane_Topology_Detection.ipynb](Task-1-Lane-Lane_Topology_Detection.ipynb)

### Task 2: Traffic Signs Detection
- **Objective**: Detect and classify traffic signs in images.
- **Approach**:
  - Implemented object detection models based on YOLOv11.
  - Preprocessed images using augmentation techniques to improve detection accuracy.
- **Notebook**: [Task-2-Traffic_Signs_Detection.ipynb](Task-2-Traffic_Signs_Detection.ipynb)

### Task 3: Centerline Detection
- **Objective**: Predict the 3D centerline of lanes from images.
- **Approach**:
  - Built a Encoder-Decoder CNN based (ResNet50) Architecture for feature extraction to determine centerline from front-camera frame 
  - Also implemented Vision Transformer (ViT) model from scratch for feature extraction.
  - Combined the extracted features with dense layers for centerline prediction.
- **Notebook**: [Task-3-CenterLine_Detection.ipynb](Task-3-CenterLine_Detection.ipynb)

---

## To setup Project

1. Clone the Repository
```bash
git clone https://github.com/DevanshuGohil/OpenLaneV2-Research.git
cd OpenLaneV2-Research
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the OpenLane V2 dataset and place it in the appropriate directory.

## Libraries Used

The project relies on the following main libraries:

- TensorFlow 2.x
- YOLO
- Anything-Depth
- OpenCV
- NumPy
- Matplotlib
- Pandas

For a complete list of dependencies, refer to the `requirements.txt` file.

## Conclusion

This project demonstrates the effectiveness of Vision Transformers in the task of 3D centerline detection for autonomous driving. The implemented model shows significant improvements over traditional CNN-based approaches, particularly in handling complex road scenarios. The use of Fréchet and Chamfer distances for evaluation provides a robust measure of the model's performance in predicting accurate lane geometries.

Future work could focus on integrating this lane detection system with other components of autonomous driving pipelines and further optimizing the model for real-time performance.