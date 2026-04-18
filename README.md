# Mask_RCNN_Update

Computer vision project for safety hat detection using Mask R-CNN. The repository contains training scripts, notebooks, datasets, visual results, and experiment artifacts for instance segmentation on workplace safety imagery.

## Project Goal

The goal of this project is to detect and segment safety hats in images with a custom Mask R-CNN pipeline and evaluate how well the model generalizes on validation and test data.

## Repository Highlights

- training code under `samples/hats`
- dataset and annotations for train, validation, and test workflows
- notebooks for experimentation and result review
- charts and screenshots in `docs/images`
- requirements files for Python environment setup

## Stack

- Python
- TensorFlow
- Keras
- OpenCV
- scikit-image
- Mask R-CNN

## Reported Results

The repository already includes training logs, validation plots, confusion matrix visuals, and example predictions. These assets make it a useful end-to-end experiment repo instead of only a training script dump.

## Setup

```bash
pip install -r requirements.txt
```

Then run the training or evaluation workflow from the `samples/hats` directory according to the scripts and notebooks included in the repository.

## Notes

This project is one of the strongest portfolio pieces on the profile because it combines modeling, evaluation, visualization, and documentation in one place.
