# Bearing Fault Diagnosis (CNN Project)

## Overview
- Dataset is read from `data/` with STFT features computed on-the-fly.
- Models are defined in `models.py` with a selector: BaselineCNN / ResNet18 / MobileNetV2.
- Metrics recorded in `train.py`: parameter count, average inference time, accuracy.

## Setup
Install dependencies (Windows, CPU by default):

```
pip install -r requirements.txt
```

## Select Model
Edit `config.py` and set:

- `CFG.model.name`: `"baseline_cnn"` or `"resnet18"` or `"mobilenet_v2"`
- `CFG.train.device`: `"cpu"` or your CUDA device (e.g. `"cuda:0"`)

## Run

```
python train.py
```

This prints total samples, a sample shape, split sizes, and metrics for the selected model.

## Notes
- Current script evaluates untrained models for pipeline validation. Add optimizer/loss to `train.py` for training.
