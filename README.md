# OpenLane-V2 - Research Project

## Introduction

This project utilizes the OpenLane V2 dataset for advanced lane detection in autonomous driving scenarios. OpenLane V2 is a comprehensive dataset that provides rich, multi-modal data for various autonomous driving tasks, with a focus on complex urban environments.

## Task Description

The main task of this project is to detect and predict 3D centerlines of lanes in road scenes. This is crucial for autonomous driving systems, as accurate lane detection is essential for proper vehicle positioning and navigation. The project involves:

1. Processing and preparing the OpenLane-V2 dataset
2. Implementing a Vision Transformer (ViT) model for 3D centerline prediction
3. Evaluating the model using Fréchet distance and Chamfer distance metrics
4. Visualizing the projected centerlines on images

## Setup

To set up the project:

1. Clone the repository:
   ```
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the OpenLane V2 dataset and place it in the appropriate directory.

## Libraries Used

The project relies on the following main libraries:

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Pandas
- tqdm

For a complete list of dependencies, refer to the `requirements.txt` file.

## Conclusion

This project demonstrates the effectiveness of Vision Transformers in the task of 3D centerline detection for autonomous driving. The implemented model shows significant improvements over traditional CNN-based approaches, particularly in handling complex road scenarios. The use of Fréchet and Chamfer distances for evaluation provides a robust measure of the model's performance in predicting accurate lane geometries.

Future work could focus on integrating this lane detection system with other components of autonomous driving pipelines and further optimizing the model for real-time performance.