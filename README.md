# Gait-Based Emotion Recognition

## Overview
This project is focused on recognizing emotions through gait analysis using advanced deep learning techniques. The goal is to explore how human emotions can be inferred from gait patterns, leveraging video-based data and MDT-GCN to classify various emotional states.

The implemented approach includes preprocessing of gait data, feature extraction, and emotion classification using a neural network model. This project can be beneficial for applications in security, healthcare, and user experience enhancement.

## Features
- Emotion recognition based on gait patterns.
- Multi-anchor adaptive fusion and bi-focus attention mechanism.
- Supports training, validation, and evaluation of deep learning models.
- GUI for visualization of results.

## Project Structure
```
Gait-Emotion-Recognition/
├── .idea/                      # IDE configuration files
├── .ipynb_checkpoints/         # Jupyter notebook checkpoints
├── config/                     # Configuration files for training and evaluation
├── feeders/                    # Data loader and preprocessing scripts
├── graph/                      # Graph-based model components
├── model/                      # Model architecture scripts
├── interface_8.py              # GUI for utilizing trained model weights (version 8)
├── interface_9.py              # GUI for utilizing trained model weights (version 9)
├── main.py                     # Main script for running the project
├── requirements.txt            # Dependencies and required packages
├── test_affective.npy          # Test data related to affective states
├── test_joint.npy              # Test data for joint positions
├── test_label.pkl              # Labels for the test dataset
├── test_movement.npy           # Test data for movement features
├── train_affective.npy         # Training data related to affective states
├── train_joint.npy             # Training data for joint positions
├── train_label.pkl             # Labels for the training dataset
```

## Getting Started

### Prerequisites
To run this project, you'll need to install the following dependencies:

- Python 3.10+
- OpenCV
- TensorFlow or PyTorch
- NumPy
- Matplotlib
- PyQt5 (for GUI)

You can install the required Python packages by running:

```sh
pip install -r requirements.txt
```


### Training the Model
You can train the model using the following instructions

```sh
python main.py
```
### Running the GUI
You can run the graphical interface for emotion recognition:

```sh
python interface_8.py
```

This will launch a simple interface allowing you to upload gait video clips and view predicted emotional states.
![Fig9](https://github.com/user-attachments/assets/d3a1a380-3b33-466f-a43e-8eb0b0df9d2c)


## Usage
- **Training**: Train the model on your dataset.
- **Inference**: Use the pre-trained model to infer emotions from new gait videos.
- **Visualization**: Use the GUI for a more interactive experience.

## Results
The current implementation achieves an accuracy of approximately 90% on the sample gait dataset, identifying emotions such as happiness, sadness, and anger.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests if you would like to add new features or fix bugs.


## Acknowledgments
- This project was inspired by research in human behavioral analysis and computer vision.
- Special thanks to contributors and the open-source community for providing useful tools and datasets.
