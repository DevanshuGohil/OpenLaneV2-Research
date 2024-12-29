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

4. Run the Jupyter notebooks in the following order:
   - `Data_Preparation.ipynb`
   - `Model_Training.ipynb`
   - `Evaluation_and_Visualization.ipynb`

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

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31101975/e21e304f-6334-4d7e-a560-2bdee4fea180/Task-3-CenterLine_Detection.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31101975/1bb96b20-fe1d-487c-b2ae-a920983d90ef/Assignment-CenterLine-Det-Transformer-3.ipynb
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31101975/a35e29c7-6c27-4ce9-a04c-dfbc1b2b0848/Assignment-2-3.ipynb
[4] https://proceedings.neurips.cc/paper_files/paper/2023/file/3c0a4c8c236144f1b99b7e1531debe9c-Paper-Datasets_and_Benchmarks.pdf
[5] https://paperswithcode.com/dataset/openlane-v2
[6] https://proceedings.neurips.cc/paper_files/paper/2023/file/3c0a4c8c236144f1b99b7e1531debe9c-Supplemental-Datasets_and_Benchmarks.pdf
[7] https://github.com/OpenDriveLab/OpenLane-V2
[8] https://paperswithcode.com/dataset/openlane-v2-test
[9] https://openreview.net/forum?id=OMOOO3ls6g&noteId=uv6QBzxYeI