# Repository Architecture Overview

> **For setup instructions, see [SETUP_AND_RUN_GUIDE.md](SETUP_AND_RUN_GUIDE.md)**
> **For interview preparation, see [PROJECT_INTERVIEW_GUIDE.md](PROJECT_INTERVIEW_GUIDE.md)**

---

## **Problem Statement**

This repository tackles a **sleep state detection problem** for the [Child Mind Institute Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states). The goal is to detect two critical sleep events from wearable accelerometer data:

- **Sleep onset**: when a person falls asleep
- **Sleep wakeup**: when a person wakes up

The input data consists of time-series accelerometer readings (angle-z and movement measurements) from wearable devices, and the challenge is to precisely identify these transition points in potentially multi-day recordings.

---

## **Repository Architecture**

The repository follows a **modular, configuration-driven machine learning pipeline** built on PyTorch Lightning. It's structured around three main phases:

### **1. Data Preparation** ([run/prepare_data.py](run/prepare_data.py))
- Loads raw parquet files containing accelerometer data
- Engineers temporal features (hour, month, minute as cyclic coordinates)
- Normalizes sensor readings (anglez, enmo)
- Saves preprocessed features as individual numpy files per series

### **2. Training** ([run/train.py](run/train.py))
- Configures experiments using Hydra (YAML-based configuration)
- Trains models with PyTorch Lightning
- Validates using event detection metrics
- Saves best model weights

### **3. Inference** ([run/inference.py](run/inference.py))
- Loads trained weights
- Generates predictions on test data
- Post-processes predictions to detect events
- Creates submission files

---

## **Component Interaction**

The architecture uses a **composable model design** with three key layers:

### **Feature Extractor** → **Decoder** → **Model**

```
Raw Time Series → Feature Extractor → Decoder → Predictions
                  (CNNSpectrogram,     (UNet1D,   (Spec1D,
                   LSTM, PANNs,        LSTM,      Spec2DCNN,
                   Spectrogram)        MLP,       CenterNet,
                                       Transformer) DETR2DCNN)
```

**Data Flow:**
1. **Dataset Layer** ([src/dataset/](src/dataset/)): Loads features, creates labels with Gaussian smoothing, handles chunking for long sequences
2. **DataModule** ([src/datamodule.py](src/datamodule.py)): Manages train/validation splits, creates PyTorch DataLoaders
3. **Model Module** ([src/modelmodule.py](src/modelmodule.py)): PyTorch Lightning wrapper handling training loop, validation, metrics
4. **Model Architecture** ([src/models/](src/models/)): Combines feature extractors and decoders for different approaches

---

## **Global Challenges Addressed**

### **1. Multi-Scale Temporal Modeling**
- Sequences can be very long (multi-day recordings)
- Solution: Configurable downsampling/upsampling rates and chunking strategies
- Uses `duration` parameter to process fixed-length windows

### **2. Class Imbalance**
- Sleep onset/wakeup events are rare compared to background sleep states
- Solutions:
  - **Negative sampling**: Randomly sample background regions (`bg_sampling_rate`)
  - **Gaussian label smoothing**: Converts hard labels to soft targets around events
  - **Data augmentation**: Mixup and Cutmix applied in feature space

### **3. Flexible Model Architecture**
- Different problems require different approaches (segmentation vs. object detection)
- Solution: **Multiple model strategies**:
  - **Spec1D/Spec2DCNN**: Treat as segmentation (classify each timestep)
  - **CenterNet**: Treat as detection (find event centers)
  - **DETR2DCNN**: Detection transformer approach
  - All configurable via Hydra configs

### **4. Event Detection Metric**
- Evaluation uses Average Precision for event detection (not simple classification)
- Post-processing ([src/utils/post_process.py](src/utils/post_process.py)) converts probability maps to discrete events
- Parameters: `score_th` (threshold) and `distance` (minimum separation between events)

### **5. Configuration Management**
- Experiments require varying many hyperparameters
- Solution: **Hydra framework** with composable YAML configs
- Can easily run multi-parameter sweeps: `python run/train.py -m downsample_rate=2,4,6,8`

---

## **Design Patterns**

1. **Separation of Concerns**: Data processing, model architecture, training logic are independent
2. **Composition over Inheritance**: Mix-and-match feature extractors and decoders
3. **Configuration as Code**: All experiments defined in YAML ([run/conf/](run/conf/))
4. **Type Safety**: Uses dataclasses for configuration ([src/conf.py](src/conf.py))
5. **Reproducibility**: Seed management, logging with Weights & Biases

---

## **Key Technical Decisions**

- **PyTorch Lightning**: Simplifies distributed training, checkpointing, logging
- **Polars**: Fast dataframe operations for preprocessing
- **Hydra**: Enables systematic hyperparameter experimentation
- **Modular models**: Easy to test different architectural components independently
- **Feature engineering**: Heavy use of cyclic temporal features and normalization

This design allows rapid experimentation while maintaining code quality and reproducibility—essential for competitive machine learning.
