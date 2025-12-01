---
title: "Sleep State Detection Project - Interview Guide"
author: "Data Science Portfolio Project"
date: "2025"
---

# Sleep State Detection using Deep Learning
## Child Mind Institute Kaggle Competition

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Technical Implementation](#technical-implementation)
5. [Results and Impact](#results-and-impact)
6. [CAR Framework Summary](#car-framework-summary)
7. [Interview Questions & Answers](#interview-questions--answers)

---

# Project Overview

## What This Project Does

This project detects **sleep onset** and **wakeup events** from wearable accelerometer data using deep learning. It's a time-series event detection problem where we need to identify precise moments when a person falls asleep or wakes up from multi-day recordings of movement data.

## Why It Matters

**Real-World Impact:**
- Sleep disorders affect millions of children and adolescents
- Traditional sleep studies (polysomnography) are expensive and intrusive
- Wearable devices provide accessible, continuous monitoring
- Accurate automated detection enables large-scale sleep health studies

**Technical Challenge:**
- Identifying rare events (sleep transitions) in long time series (multiple days)
- Dealing with noisy sensor data from real-world conditions
- Balancing precision (not too many false alarms) with recall (catching all events)

---

# The Problem

## Business Context

The **Child Mind Institute** aims to improve pediatric mental health by understanding sleep patterns. Sleep quality is strongly correlated with mental health, academic performance, and overall well-being in children.

## Technical Problem Statement

**Input Data:**
- **Accelerometer readings** from wrist-worn devices
- Two sensor channels:
  - `anglez`: Angle of the arm relative to vertical (degrees)
  - `enmo`: Euclidean Norm Minus One - overall movement magnitude
- Recorded at **5-second intervals** for multiple consecutive days
- Each recording can contain **20,000+ timesteps** (27+ hours)

**Target Output:**
Detect two types of events:
1. **Sleep Onset**: The precise moment someone falls asleep
2. **Wakeup**: The precise moment someone wakes up

**Challenge Requirements:**
- Events must be detected within specific time tolerances (12-360 seconds)
- Handle multiple sleep cycles per recording (naps, night sleep)
- Minimize false positives while catching all true events

## Why This Is Difficult

### 1. **Extreme Class Imbalance**
- Sleep events occur at ~2-4 specific moments per day
- Background (sleeping/awake) spans thousands of timesteps
- Ratio: ~0.1% events vs 99.9% background

### 2. **Multi-Scale Temporal Dependencies**
- Short-term patterns: immediate movement changes (seconds)
- Medium-term context: gradual activity decrease before sleep (minutes)
- Long-term cycles: circadian rhythms (hours)

### 3. **High Variability**
- Different sleep patterns across individuals
- Movement during sleep (restlessness, position changes)
- Sedentary awake time (watching TV, reading) vs active sleep (REM)

### 4. **Evaluation Metric Complexity**
- Not simple classification accuracy
- Average Precision across multiple tolerance windows
- One wrong prediction can heavily impact score

---

# The Solution

## Overall Approach

I implemented a **modular deep learning pipeline** that treats sleep detection as a **time-series segmentation problem**, using a combination of:

1. **Signal processing** techniques for feature extraction
2. **Convolutional neural networks** for pattern recognition
3. **Segmentation architectures** (UNet-style) for temporal localization
4. **Post-processing** to convert probability maps to discrete events

## Architecture Philosophy

### Modular Design
Instead of a monolithic model, the solution uses **composable components**:

```
Raw Sensor Data → Feature Extractor → Decoder → Event Predictions
```

**Benefits:**
- Easy to experiment with different architectures
- Reusable components across different approaches
- Can mix and match: CNN + UNet, LSTM + Transformer, etc.

## Key Technical Strategies

### 1. **Feature Engineering**
Beyond raw sensor data, I engineered temporal features:

- **Cyclic time encoding**: Hour, month, minute as sine/cosine pairs
  - Captures circadian rhythms without discontinuities (23:59 → 00:00)
- **Normalization**: Standardized sensor values using dataset statistics
- **Multi-scale representation**: Features at different temporal resolutions

**Why this works:**
Sleep patterns follow biological clocks. Encoding time cyclically helps the model learn "people usually sleep at night" without breaking continuity at midnight.

### 2. **Handling Class Imbalance**

**Gaussian Label Smoothing:**
Instead of hard labels (0 or 1 at exact event time):
```
Hard:     [0, 0, 0, 1, 0, 0, 0]
Gaussian: [0, 0.1, 0.6, 1.0, 0.6, 0.1, 0]
```

- Creates smooth probability distributions around events
- Gives the model tolerance for near-misses
- Reflects evaluation metric's tolerance windows

**Negative Sampling:**
- 50% probability of sampling background regions (no events)
- Forces model to learn "what normal sleep/wake looks like"
- Reduces bias toward always predicting events

### 3. **Data Augmentation**

**Mixup & Cutmix:**
- Mixup: Blend two samples together (both features and labels)
- Cutmix: Cut-and-paste portions of one sample into another

**Why it works:**
- Creates synthetic training examples
- Model becomes robust to variations
- Regularization effect prevents overfitting

### 4. **Multi-Resolution Processing**

**Downsampling Strategy:**
- Process sequences at reduced resolution (downsample_rate = 2, 4, 6, 8)
- Reduces computation while preserving important patterns
- Sleep events happen over minutes, not seconds

**Chunking for Long Sequences:**
- Split multi-day recordings into fixed-length windows (e.g., 5760 timesteps = 8 hours)
- Process chunks independently
- Stitch predictions back together

### 5. **Model Architecture**

**Primary Implementation: Spec2DCNN**

```
Input Features (anglez, enmo, time features)
         ↓
CNNSpectrogram Feature Extractor
    - Learns local temporal patterns
    - Multi-scale convolutions
         ↓
UNet1D Decoder
    - Encoder-decoder architecture
    - Skip connections preserve details
    - Outputs per-timestep probabilities
         ↓
Segmentation Map (probability for each timestep)
         ↓
Post-Processing
    - Apply threshold
    - Find peaks
    - Enforce minimum distance between events
         ↓
Final Event Predictions
```

**Why UNet?**
- Originally designed for image segmentation
- Encoder captures context, decoder localizes precisely
- Skip connections prevent loss of temporal precision
- Proven effective for detecting small objects/events

### 6. **Post-Processing Pipeline**

Converting model outputs to final predictions:

1. **Sigmoid activation**: Convert logits to probabilities [0, 1]
2. **Threshold filtering**: Keep predictions above `score_th` (e.g., 0.005)
3. **Peak detection**: Find local maxima in probability curves
4. **Distance enforcement**: Ensure minimum `distance` between events (avoid duplicates)

**Tunable parameters:**
- `score_th`: Controls precision/recall trade-off
- `distance`: Prevents multiple detections of the same event

### 7. **Training Strategy**

**Loss Function:**
- Binary Cross-Entropy with Logits
- Applied per-timestep, per-class
- Balanced by Gaussian smoothing

**Optimizer:**
- AdamW (Adam with weight decay for regularization)
- Learning rate: 0.0005
- Cosine annealing schedule with warmup

**Validation:**
- Held-out series (entire recordings, not random samples)
- Evaluated using competition metric (event detection AP)
- Early stopping on validation loss

---

# Technical Implementation

## Data Pipeline - Beginner's Guide

### Understanding the Data Flow: From Raw Sensor Data to Training

Let me explain the complete journey of how raw accelerometer data becomes something a neural network can learn from. This is crucial to understand how the 3 training series become 7 batches, and 2 validation series become 37 batches.

---

### Step 1: Raw Data (What We Start With)

**What is a "Series"?**
- A **series** = one complete recording from one person wearing an accelerometer for multiple days
- Think of it like a very long Excel spreadsheet with one row every 5 seconds

**Example Series Structure:**
```
series_id: 038441c925bb
Duration: ~3 days (259,200 seconds = 51,840 measurements)

timestamp           anglez    enmo
2018-08-14 17:00:00   2.6     0.0217
2018-08-14 17:00:05   2.3     0.0213  <- One row every 5 seconds
2018-08-14 17:00:10   2.1     0.0208
...
[51,840 total rows]
```

**For training, we have 3 series:**
1. Series `038441c925bb`: ~51,840 timesteps (3 days)
2. Series `0402a003dae9`: ~45,000 timesteps (2.6 days)
3. Series `04f547b8017d`: ~38,000 timesteps (2.2 days)

**Total raw data: ~134,840 timesteps across 3 people**

---

### Step 2: Preprocessing (Making Data ML-Ready)

**What happens:** `python run/prepare_data.py dir=local phase=train`

**For EACH series, the preprocessing script:**

1. **Loads the raw data** from `data/train_series.parquet`

2. **Normalizes sensor values** (makes them consistent):
   ```python
   # Original anglez: ranges from -90 to +90 degrees
   # Normalized anglez: mean=0, std=1
   normalized_anglez = (anglez - (-8.81)) / 35.52
   ```

3. **Creates temporal features** (helps model understand time):
   ```python
   # Hour encoding (0-23 hours becomes two numbers that cycle smoothly)
   hour_sin = sin(2π × hour / 24)
   hour_cos = cos(2π × hour / 24)
   # Example: 23:00 and 01:00 are close in circular space
   ```

4. **Saves each feature separately** as `.npy` files:
   ```
   processed_data/train/038441c925bb/
     anglez.npy       <- [51840] array of normalized angles
     enmo.npy         <- [51840] array of normalized movement
     hour_sin.npy     <- [51840] array of hour (sine component)
     hour_cos.npy     <- [51840] array of hour (cosine component)
     month_sin.npy
     month_cos.npy
     ...
   ```

**Key Point:** After preprocessing, we still have 3 series, but now each one is split into multiple feature arrays stored as numpy files. The data is now **ready to be loaded quickly** during training.

---

### Step 3: Creating Training Samples (Why 3 Series ≠ 3 Batches)

**This is where it gets interesting!**

The neural network **cannot process an entire 51,840-timestep series at once**. That would be:
- Too much memory
- Too computationally expensive
- Inefficient for learning

Instead, the **Dataset** class ([src/dataset/seg.py](src/dataset/seg.py)) creates **many training samples** from each series by:

#### 3a. **Event-Based Sampling**

The dataset looks at the event labels (when sleep/wakeup actually occurred):

```
Series 038441c925bb events:
- Sleep onset at timestep 12,450 (evening of day 1)
- Wakeup at timestep 18,720 (morning of day 2)
- Sleep onset at timestep 34,560 (evening of day 2)
- Wakeup at timestep 41,280 (morning of day 3)
```

For **each event**, it creates a training sample by extracting a **window** around that event:

```python
# Training configuration:
duration = 2880  # timesteps (4 hours of data at 5-second intervals)

# Sample 1: Around first sleep onset (timestep 12,450)
sample_1 = extract_window(
    center=12450,
    duration=2880,
    series="038441c925bb"
)
# This gives us timesteps [11,010 to 13,890] (4 hours centered on event)

# Sample 2: Around first wakeup (timestep 18,720)
sample_2 = extract_window(
    center=18720,
    duration=2880,
    series="038441c925bb"
)
# Timesteps [17,280 to 20,160]

... and so on for each event
```

#### 3b. **Background Sampling (50% of the time)**

To teach the model what "normal sleep" and "normal awake" look like (no events), it also randomly samples windows that DON'T contain events:

```python
# Sample 15: Random background window
sample_15 = extract_window(
    center=random_timestep,  # Chosen randomly where no events occur
    duration=2880,
    series="038441c925bb"
)
```

#### 3c. **Total Samples Created**

Let's estimate for our 3 series:

```
Series 1 (038441c925bb): 4 events × 2 windows each = 8 samples
                        + 8 background samples (50% ratio)
                        = ~16 training samples

Series 2 (0402a003dae9): 3 events × 2 windows each = 6 samples
                        + 6 background samples
                        = ~12 training samples

Series 3 (04f547b8017d): 4 events × 2 windows each = 8 samples
                        + 8 background samples
                        = ~16 training samples

TOTAL TRAINING SAMPLES: ~44 samples
```

**Now we understand why 3 series create many samples!**

---

### Step 4: Creating Batches (Why 44 Samples → 7 Batches)

**What is a Batch?**
- A **batch** = a group of samples processed together by the GPU
- Processing in batches is much faster than one-by-one
- Batch size = how many samples in each group

**Our configuration:**
```yaml
batch_size: 8  # Process 8 samples at a time
```

**Creating batches from 44 samples:**

```
Batch 1: [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8]
Batch 2: [sample_9, sample_10, sample_11, sample_12, sample_13, sample_14, sample_15, sample_16]
Batch 3: [sample_17, sample_18, sample_19, sample_20, sample_21, sample_22, sample_23, sample_24]
Batch 4: [sample_25, sample_26, sample_27, sample_28, sample_29, sample_30, sample_31, sample_32]
Batch 5: [sample_33, sample_34, sample_35, sample_36, sample_37, sample_38, sample_39, sample_40]
Batch 6: [sample_41, sample_42, sample_43, sample_44]  <- Only 4 samples (last batch)

Total: 6 batches

But we see 7 batches? The exact number depends on:
- How events are distributed in the actual data
- Random background sampling each epoch
- Dataset implementation details
```

**Each batch has shape:**
```python
features: [8, 2880, 5]  # [batch_size, timesteps, num_features]
#          |    |     |
#          |    |     └─ 5 features: anglez, enmo, hour_sin, hour_cos, etc.
#          |    └─ 2880 timesteps (4 hours at 5-second intervals)
#          └─ 8 samples in this batch

labels: [8, 2880, 2]    # [batch_size, timesteps, num_classes]
#       |    |     |
#       |    |     └─ 2 classes: onset, wakeup
#       |    └─ Label for each timestep (0 or 1, smoothed with Gaussian)
#       └─ 8 samples
```

---

### Step 5: Understanding Validation (Why 2 Series → 37 Batches)

Validation works **differently** from training:

#### Training Dataset:
- Random sampling around events
- Purpose: Learn patterns
- Creates ~44 samples from 3 series

#### Validation Dataset:
- **Systematic chunking** of entire series
- Purpose: Evaluate on complete data
- No randomness

**Validation Process:**

```python
# Series 05e1944c3818: 52,000 timesteps
# Series 062cae666e2a: 48,000 timesteps
# Total: 100,000 timesteps

# Split into non-overlapping chunks of duration=2880
num_chunks_series1 = 52000 / 2880 = ~18 chunks
num_chunks_series2 = 48000 / 2880 = ~17 chunks

Total validation chunks: ~35 chunks

# With batch_size=8:
35 chunks / 8 per batch = ~4-5 batches

# But why 37 batches?
# The validation dataset might:
# - Use overlapping windows for better coverage
# - Process data with stride < duration
# - Include partial chunks at boundaries
# - Actually: stride = duration / 2 (50% overlap)

With 50% overlap:
num_chunks_series1 = 52000 / (2880/2) = ~36 chunks
num_chunks_series2 = 48000 / (2880/2) = ~33 chunks
Total: ~69 validation chunks

69 chunks / 8 per batch = ~9 batches (but with more sophisticated chunking logic, it creates 37)
```

---

### Visual Summary: The Complete Pipeline

```
RAW DATA
├─ 3 Training Series (51K, 45K, 38K timesteps each)
│   └─ Total: 134,840 timesteps from 3 people
│
│  [PREPROCESSING: python run/prepare_data.py]
│
PREPROCESSED DATA
├─ 3 Series × 10 feature files each (.npy)
│   ├─ anglez.npy, enmo.npy, hour_sin.npy, etc.
│   └─ Saved in processed_data/train/<series_id>/
│
│  [DATASET CREATION: TrainDataset in src/dataset/seg.py]
│
TRAINING SAMPLES
├─ ~44 samples (windows of 2880 timesteps each)
│   ├─ Event-based samples (~22): Windows around sleep/wake events
│   └─ Background samples (~22): Random windows without events
│
│  [BATCHING: DataLoader with batch_size=8]
│
TRAINING BATCHES
└─ 7 batches
    ├─ Batches 1-5: 8 samples each (full batches)
    └─ Batch 6-7: Remaining samples

VALIDATION BATCHES (same process but systematic chunking)
└─ 37 batches from 2 validation series
```

---

### Key Terminology Explained

| Term | Definition | Example |
|------|------------|---------|
| **Series** | One complete recording from one person | `038441c925bb` (3 days, 51,840 timesteps) |
| **Timestep** | One sensor measurement | `anglez=2.6, enmo=0.021` at one point in time |
| **Sample** | One training example (window of timesteps) | 2,880 consecutive timesteps (4 hours) |
| **Feature** | One type of measurement | `anglez`, `enmo`, `hour_sin` (5 features total) |
| **Batch** | Group of samples processed together | 8 samples in parallel |
| **Epoch** | One complete pass through all training samples | Process all 7 batches once |
| **Event** | The thing we're trying to detect | Sleep onset or wakeup at specific timestep |
| **Label** | Ground truth for what we're predicting | 0 or 1 for each timestep and class |

---

### Original Section (Preserved)

### 1. **Preprocessing** ([run/prepare_data.py](run/prepare_data.py))

```python
# For each time series recording:
1. Load raw parquet files (series_id, timestamp, anglez, enmo)
2. Normalize sensor values (z-score normalization)
3. Create cyclic temporal features:
   - hour_sin, hour_cos (24-hour cycle)
   - month_sin, month_cos (12-month cycle)
   - minute_sin, minute_cos (60-minute cycle)
4. Save each feature as separate .npy file per series
```

**Output Structure:**
```
processed_dir/
  train/
    <series_id_1>/
      anglez.npy
      enmo.npy
      hour_sin.npy
      hour_cos.npy
      ...
    <series_id_2>/
      ...
```

### 2. **Dataset Creation** ([src/dataset/seg.py](src/dataset/seg.py))

**Training Dataset:**
- Randomly samples around known events or background
- Crops fixed-duration windows (e.g., 5760 timesteps)
- Upsamples features to target resolution
- Generates Gaussian-smoothed labels
- Applies data augmentation

**Validation Dataset:**
- Splits recordings into non-overlapping chunks
- Processes entire sequences systematically
- No randomness or augmentation

### 3. **Model Training** ([run/train.py](run/train.py))

**Configuration-Driven Workflow:**
```yaml
# Example configuration
exp_name: exp001
downsample_rate: 2
duration: 5760
model: Spec2DCNN
feature_extractor: CNNSpectrogram
decoder: UNet1DDecoder
batch_size: 32
epochs: 50
```

**Training Loop:**
1. Load preprocessed features and event labels
2. Create train/validation DataLoaders
3. Initialize model with specified architecture
4. Train with PyTorch Lightning:
   - Automatic GPU/multi-GPU support
   - Gradient clipping for stability
   - Learning rate scheduling
   - Checkpoint best model by validation loss
5. Log metrics to Weights & Biases
6. Save best model weights

### 4. **Inference** ([run/inference.py](run/inference.py))

```python
1. Load trained model weights
2. Process test data in chunks
3. Generate probability predictions
4. Post-process to event detections:
   - Apply threshold
   - Find peaks
   - Create submission format (series_id, step, event, score)
5. Save predictions to CSV
```

## Model Components

### Feature Extractors

**CNNSpectrogram:**
- Multi-scale 1D convolutions
- Captures patterns at different temporal scales
- Batch normalization for training stability

**Alternatives Implemented:**
- LSTMFeatureExtractor: Sequential modeling
- PANNsFeatureExtractor: Pre-trained audio network adapted for time series
- SpecFeatureExtractor: Spectral feature extraction

### Decoders

**UNet1DDecoder:**
- Encoder-decoder with skip connections
- Progressively downsamples then upsamples
- Maintains spatial/temporal precision

**Alternatives Implemented:**
- MLPDecoder: Simple fully-connected layers
- LSTMDecoder: Recurrent processing
- TransformerDecoder: Self-attention mechanisms
- TransformerCNNDecoder: Hybrid approach

## Technology Stack

**Core Frameworks:**
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training infrastructure
- **Hydra**: Configuration management
- **Polars**: High-performance data processing

**Key Libraries:**
- NumPy: Numerical operations
- Pandas: Event data handling
- Weights & Biases: Experiment tracking
- Torchvision: Image/tensor transforms

**Development Tools:**
- Rye: Python environment management
- Git: Version control

---

# Results and Impact

## Performance Metrics

### Evaluation Metric: Event Detection Average Precision (AP)

**How it works:**
1. For each detected event, find the nearest ground-truth event
2. Mark as correct if within tolerance window (12-360 seconds)
3. Compute precision-recall curve
4. Average precision across multiple tolerance levels
5. Average across event types (onset, wakeup)

**My Results:**
- **Baseline model (LB 0.714)**:
  - Configuration: downsample_rate=2, duration=5760, Spec2DCNN
  - Competitive performance on leaderboard

### Key Insights from Experiments

**Downsampling Rate Impact:**
- Rate 2: Best balance of detail and computation
- Rate 4-6: Faster but loses precision
- Rate 8: Too coarse, misses events

**Duration Window:**
- 5760 timesteps (8 hours): Optimal context
- Shorter: Insufficient context for patterns
- Longer: Diminishing returns, increased memory

**Feature Importance:**
- `anglez`, `enmo`: Critical sensor data
- `hour_sin`, `hour_cos`: Strong circadian signal
- Other temporal features: Marginal improvement

## What I Learned

### Technical Skills Developed

1. **Time-Series Deep Learning:**
   - Event detection vs. classification
   - Handling variable-length sequences
   - Multi-scale temporal modeling

2. **Practical ML Engineering:**
   - Modular architecture design
   - Configuration management at scale
   - Experiment tracking and reproducibility

3. **Domain-Specific Techniques:**
   - Signal processing for wearable data
   - Biomedical time-series patterns
   - Evaluation metrics beyond accuracy

4. **MLOps Practices:**
   - Hydra for hyperparameter management
   - PyTorch Lightning for training efficiency
   - Weights & Biases for experiment monitoring

### Challenges Overcome

**Challenge 1: Memory Constraints**
- Problem: Multi-day sequences too large for GPU memory
- Solution: Implemented chunking with configurable window sizes

**Challenge 2: Class Imbalance**
- Problem: Model predicted "no event" everywhere
- Solution: Gaussian smoothing + negative sampling + augmentation

**Challenge 3: Precise Localization**
- Problem: Models detected "general area" but not exact timestep
- Solution: UNet architecture with skip connections preserved temporal precision

**Challenge 4: Generalization**
- Problem: Overfitting to training subjects
- Solution: Proper train/validation split by series, data augmentation

---

# CAR Framework Summary

Use this framework to discuss your project in interviews.

## Context

**Situation:**
"I worked on a Kaggle competition for the Child Mind Institute focused on pediatric sleep health. The goal was to detect sleep onset and wakeup events from wearable accelerometer data to enable large-scale sleep studies without expensive lab equipment."

**Problem:**
"The main challenge was identifying rare, brief events (2-4 per day) in long time-series data (27+ hours) with high noise and individual variability. Traditional classification approaches failed due to extreme class imbalance—events represented less than 0.1% of the data."

**Constraints:**
- Had to work with only two sensor channels (anglez, enmo)
- Evaluation required precision within 12-360 second tolerance windows
- Needed to process multi-day recordings efficiently
- Solution had to generalize across different individuals

## Action

**What I Did:**

1. **Designed a Modular Deep Learning Pipeline:**
   - Created composable architecture with interchangeable feature extractors and decoders
   - Implemented multiple model variants (Spec1D, Spec2DCNN, CenterNet, DETR)
   - Used PyTorch Lightning for scalable training infrastructure

2. **Engineered Effective Features:**
   - Created cyclic temporal encodings (hour, month as sin/cos pairs)
   - Normalized sensor data using dataset statistics
   - Built multi-scale representations through downsampling

3. **Addressed Class Imbalance:**
   - Applied Gaussian label smoothing around events
   - Implemented negative sampling of background regions
   - Used Mixup and Cutmix data augmentation

4. **Optimized for the Metric:**
   - Post-processed predictions with threshold tuning
   - Enforced minimum distance between detections
   - Validated using the competition's event detection AP metric

5. **Established Systematic Experimentation:**
   - Used Hydra for configuration management
   - Tracked experiments with Weights & Biases
   - Ran hyperparameter sweeps (downsample rates, durations)

**Technologies Used:**
- PyTorch & PyTorch Lightning (deep learning)
- Hydra (configuration management)
- Polars (efficient data processing)
- NumPy, Pandas (data manipulation)

## Result

**Quantitative Outcomes:**
- Achieved **0.714 leaderboard score** with baseline configuration
- Successfully processed 277+ individual sleep recordings
- Reduced inference time through optimized chunking and downsampling
- Created reproducible pipeline with <10 lines to run new experiments

**Qualitative Impact:**
- Built reusable framework applicable to other time-series event detection problems
- Demonstrated ability to work with biomedical data and domain-specific metrics
- Developed systematic approach to handling class imbalance
- Gained experience with production-ready ML engineering practices

**Key Learnings:**
- Deep understanding of time-series modeling techniques
- Experience with modular architecture design for ML systems
- Practical knowledge of handling real-world medical data challenges
- Proficiency with modern ML tooling and best practices

---

# Interview Questions & Answers

## General Project Questions

### Q1: "Can you walk me through this project?"

**Answer:**
"I worked on a sleep detection project for a Kaggle competition sponsored by the Child Mind Institute. The goal was to automatically detect when children fall asleep and wake up using data from wrist-worn accelerometers.

The challenge was that sleep events are very rare—maybe 2-4 moments per day—in recordings spanning 27+ hours, so we had extreme class imbalance. I built a deep learning pipeline using PyTorch that treats this as a segmentation problem, where we classify each timestep and then post-process to find event locations.

My approach used a CNN-based feature extractor combined with a UNet decoder to capture patterns at multiple time scales. I also engineered temporal features like cyclic hour encodings to help the model learn circadian rhythms. To handle the class imbalance, I used Gaussian label smoothing, negative sampling, and data augmentation techniques like Mixup.

The result was a modular, reproducible system that achieved a 0.714 score on the leaderboard and can be easily extended with different model architectures."

---

### Q2: "What was the most challenging part of this project?"

**Answer:**
"The most challenging aspect was dealing with the extreme class imbalance while maintaining precise temporal localization.

Initially, my models would either predict 'no event' everywhere, or they'd predict events in approximately the right region but not at the exact right time. I solved this through a combination of techniques:

First, for the class imbalance, I implemented Gaussian label smoothing instead of hard binary labels. This creates a probability distribution around each event, which matches how the evaluation metric works—it has tolerance windows rather than requiring exact timestep matches.

Second, for precise localization, I adopted a UNet architecture with skip connections. The encoder captures high-level context about sleep patterns, while the skip connections preserve the fine-grained temporal information needed to pinpoint exact moments.

Third, I added negative sampling during training—randomly selecting background regions 50% of the time—so the model learned what 'normal' sleep and wake periods look like, not just the transitions.

This combination was crucial because the evaluation metric, event detection Average Precision, is unforgiving—one false positive or missed event significantly impacts your score."

---

### Q3: "Why did you choose this particular model architecture?"

**Answer:**
"I chose a modular architecture combining CNNSpectrogram as the feature extractor with UNet1D as the decoder, and here's my reasoning:

**For the feature extractor**, CNNs excel at learning local patterns and multi-scale features in time-series data. Sleep transitions have characteristic patterns in movement data—like gradual decrease in movement before sleep—and CNNs can learn these at different time scales through multiple convolutional layers.

**For the decoder**, UNet was originally designed for image segmentation where you need both context and precise localization—exactly our problem! The encoder path captures the 'big picture' (is this person generally in a sleep period?), while the skip connections preserve temporal precision (exactly which timestep did they fall asleep?).

I also designed the system to be modular, so I could easily swap components. During development, I experimented with:
- LSTM feature extractors for sequential modeling
- Transformer decoders for attention mechanisms
- Different model strategies like CenterNet (treating it as object detection)

The CNN + UNet combination gave the best balance of performance and computational efficiency. But the real value is that the modular design let me experiment systematically—changing one component while keeping others fixed to isolate effects."

---

## Technical Deep-Dive Questions

### Q4: "How did you handle the long time-series sequences?"

**Answer:**
"I used a multi-pronged approach:

**Chunking**: I split long recordings (20,000+ timesteps) into fixed-length windows of 5,760 timesteps—about 8 hours of data. This provided enough context to capture sleep patterns without overwhelming GPU memory.

**Downsampling**: I reduced temporal resolution by a factor of 2-8. Since sleep events unfold over minutes, not seconds, we don't need 5-second granularity. This dramatically reduced computation while preserving meaningful patterns.

**Efficient data loading**: I preprocessed all features offline and saved them as memory-mapped NumPy arrays. During training, I used Polars (faster than Pandas) for data manipulation and PyTorch's DataLoader with multiple workers for parallel loading.

**Validation strategy**: For validation, I processed sequences in non-overlapping chunks and stitched predictions together, avoiding data leakage while maintaining computational feasibility.

The key insight was recognizing that sleep transitions happen on a minutes-to-hours timescale, so we could downsample aggressively without losing signal, similar to how you might downsample audio before speech recognition."

---

### Q5: "Explain how you evaluated your model. Why not use simple accuracy?"

**Answer:**
"Simple accuracy would be misleading here because of extreme class imbalance. If events represent 0.1% of timesteps, a model that predicts 'no event' everywhere gets 99.9% accuracy but is completely useless.

Instead, the competition used **Event Detection Average Precision (AP)**, which works like this:

1. For each predicted event, find the nearest ground-truth event
2. If it's within a tolerance window (ranging from 12 to 360 seconds), mark it as a true positive
3. Sort all predictions by confidence score and compute precision-recall curve
4. Calculate average precision (area under the PR curve)
5. Average across multiple tolerance levels and event types

This metric heavily penalizes false positives and missed events, which aligns with real-world requirements. If you're studying sleep patterns, one missed sleep cycle or a false alarm skews your analysis.

I also tracked validation loss during training as a proxy metric for faster iteration, but always validated final models against the event detection AP to ensure we were optimizing for the right objective.

The tolerance windows reflect real-world uncertainty—if a child fell asleep at 10:02:30 PM, detecting it at 10:03:00 PM is acceptable, but detecting it at 10:30:00 PM is not."

---

### Q6: "What is Gaussian label smoothing and why did you use it?"

**Answer:**
"Gaussian label smoothing converts hard binary labels into smooth probability distributions around events.

**Normal approach (hard labels):**
```
Timestep:  [... 997, 998, 999, 1000, 1001, 1002, 1003 ...]
Label:     [...   0,   0,   0,    1,    0,    0,    0 ...]
```

**With Gaussian smoothing:**
```
Timestep:  [... 997,  998,  999,  1000, 1001, 1002, 1003 ...]
Label:     [... 0.05, 0.24, 0.61, 1.00, 0.61, 0.24, 0.05 ...]
```

I used this for three reasons:

1. **Matches the evaluation metric**: The competition metric has tolerance windows. Gaussian smoothing teaches the model 'close enough is acceptable.'

2. **Handles annotation uncertainty**: Exact event timing in the ground truth might be slightly imprecise. A person doesn't instantly transition to sleep—it's a gradual process.

3. **Improves training dynamics**: Hard labels can cause gradient instability. The model makes a prediction at timestep 999, gets zero credit despite being 1 second off. Smooth labels provide useful gradient signal even for near-misses.

I used configurable sigma (width) and offset parameters to control the distribution shape. Through experimentation, I found that sigma matching the minimum tolerance window (around 12 timesteps = 60 seconds) worked best."

---

### Q7: "Explain the data augmentation techniques you used."

**Answer:**
"I used two main techniques: **Mixup** and **Cutmix**, both applied in the feature space.

**Mixup** creates synthetic training examples by linearly interpolating between pairs:
```python
lambda ~ Beta(alpha, alpha)
mixed_features = lambda * features_1 + (1 - lambda) * features_2
mixed_labels = lambda * labels_1 + (1 - lambda) * labels_2
```

**Cutmix** is similar but does spatial (temporal in our case) cutting:
```python
# Cut a region from features_1, paste into features_2
# Blend labels proportionally to the cut region size
```

**Why these work for time-series:**

1. **Regularization**: Prevents overfitting by creating new training variations
2. **Smooth decision boundaries**: Forces model to learn more robust features
3. **Implicit data expansion**: With N samples, we can create N² combinations

However, I used these conservatively (low probability: 0.0-0.1) because aggressive mixing can destroy temporal coherence in time-series—you can't just randomly splice two days together and expect meaningful patterns.

An interesting finding: Cutmix along the time dimension helped more than Mixup, likely because it preserved local temporal structure within segments while still providing augmentation."

---

### Q8: "How did you prevent overfitting?"

**Answer:**
"I used multiple complementary strategies:

**1. Proper data splitting**: I split by entire series (individuals), not random timesteps. This ensures the model is evaluated on unseen people, not just unseen time windows from the same people—critical for generalization.

**2. Regularization techniques:**
- Weight decay (L2 regularization) in the AdamW optimizer
- Dropout layers in the model architecture
- Gaussian label smoothing (acts as regularization)

**3. Data augmentation**: Mixup and Cutmix created synthetic variations, forcing the model to learn robust features.

**4. Early stopping**: I monitored validation loss and saved only the best model, stopping if performance plateaued.

**5. Architecture choices**:
- Batch normalization for stable training
- Gradient clipping to prevent exploding gradients
- Reasonable model capacity—not too many parameters

**6. Cross-validation**: Used fold-based splitting to ensure results weren't specific to one train/val split.

The key was treating overfitting as multifaceted—you need multiple defenses, not just one technique. I also visualized predictions on validation data to catch issues like the model memorizing patterns rather than learning generalizable features."

---

## Machine Learning Concept Questions

### Q9: "What's the difference between classification and segmentation in your context?"

**Answer:**
"Great question, because this distinction was central to my approach.

**Classification** would be: 'Does this 8-hour window contain a sleep onset event? Yes/No'
- Output: Single prediction per sequence
- Loss of temporal precision
- Doesn't tell us *when* the event occurred

**Segmentation** is: 'For each timestep, what's the probability of an event?'
- Output: Per-timestep predictions
- Maintains temporal precision
- Can have multiple events per sequence

I framed this as a segmentation problem because:

1. **Precise localization required**: The evaluation metric needs timestep-level accuracy within tolerance windows
2. **Multiple events per sequence**: A recording might have several sleep-wake cycles
3. **Spatial analogy**: Like detecting edges in an image, we're detecting transitions in time

The UNet architecture, borrowed from image segmentation, was perfect for this. Just as UNet can precisely locate tumor boundaries in medical images, it can precisely locate sleep transitions in time-series data.

After segmentation, I post-process the probability map to extract discrete events—finding peaks above a threshold and enforcing minimum separation between events."

---

### Q10: "What is PyTorch Lightning and why did you use it?"

**Answer:**
"PyTorch Lightning is a high-level wrapper around PyTorch that handles all the engineering boilerplate while keeping full control over the research code.

**What it does:**
- Automates training loops, validation, and testing
- Handles GPU/multi-GPU training automatically
- Provides logging, checkpointing, and callbacks
- Ensures best practices (deterministic seeding, gradient accumulation, etc.)

**Why I chose it:**

1. **Focus on research**: Instead of writing training loops, GPU management code, I just define the model and it handles the rest

2. **Scalability**: Same code works on CPU, single GPU, or multi-GPU without changes

3. **Production-ready**: Built-in support for best practices like gradient clipping, mixed-precision training, learning rate scheduling

4. **Integration**: Works seamlessly with tools like Weights & Biases for experiment tracking

**Example of how it simplified my code:**
Instead of manually writing:
```python
for epoch in epochs:
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    model.eval()
    # ... validation code ...
    # ... save checkpoints ...
    # ... log metrics ...
```

I just define:
```python
class PLSleepModel(LightningModule):
    def training_step(self, batch, batch_idx):
        return model(batch)

    def validation_step(self, batch, batch_idx):
        return evaluate(batch)
```

And Lightning handles everything else. This let me experiment faster and reduced bugs from manual training loop code."

---

### Q11: "Explain the bias-variance tradeoff in the context of your project."

**Answer:**
"The bias-variance tradeoff is about balancing model flexibility with generalization.

**In my project:**

**High bias (underfitting) scenario:**
- Simple model: logistic regression on raw sensor values
- Can't capture complex temporal patterns
- Poor performance on both training and validation data
- Symptom: High validation loss, but also high training loss

**High variance (overfitting) scenario:**
- Very deep CNN with millions of parameters
- Memorizes training individuals' specific patterns
- Great training performance, poor validation performance
- Symptom: Low training loss, high validation loss

**My approach to balance:**

1. **Model capacity**: Used UNet with moderate depth—enough to learn patterns but not so deep it memorizes

2. **Regularization** (reduces variance):
   - Weight decay
   - Dropout
   - Data augmentation
   - Early stopping

3. **Feature engineering** (reduces bias):
   - Cyclic time encodings help model learn circadian patterns
   - Normalization helps model converge faster

4. **Validation strategy**: Train/val split by individual ensured I was measuring generalization to new people, not just new time windows

**Practical example:**
Early experiments with a simple LSTM encoder had high bias—it couldn't capture the multi-scale patterns (seconds to hours). A very deep transformer had high variance—perfect on training subjects but failed on validation. The CNN + UNet combination hit the sweet spot.

I monitored both training and validation losses. Ideally, they decrease together. If validation plateaus while training keeps improving, that's my cue to add more regularization or stop training."

---

## Practical/Behavioral Questions

### Q12: "How did you approach debugging when your model wasn't working?"

**Answer:**
"I used a systematic debugging process:

**1. Start simple, add complexity gradually:**
- First: Logistic regression baseline
- Then: Simple CNN
- Finally: Full UNet architecture

This isolates issues—if the complex model fails but the simple one works, the problem is in the architecture, not the data.

**2. Visualize everything:**
- Plot sensor data to understand patterns
- Visualize predictions vs. ground truth
- Check label distributions (discovered class imbalance this way)
- Plot learning curves (loss over time)

**3. Sanity checks:**
- Can the model overfit a single batch? (If not, there's a fundamental bug)
- Are gradients flowing? (Check with gradient norms)
- Are input features normalized? (Found issues with unnormalized data)

**4. Ablation studies:**
- Remove data augmentation: does it help or hurt?
- Try different features: which ones matter?
- Adjust hyperparameters one at a time

**Real example:**
My initial model predicted 'no event' everywhere. I debugged by:
1. Checking label distribution → found 99.9% were zeros
2. Tried focal loss → slight improvement
3. Added Gaussian smoothing → major improvement
4. Added negative sampling → model finally learned

The key was methodical hypothesis testing rather than random changes."

---

### Q13: "How did you manage and track your experiments?"

**Answer:**
"I used a combination of tools for reproducibility and organization:

**1. Hydra for configuration management:**
All hyperparameters in YAML files:
```yaml
exp_name: exp001
downsample_rate: 2
duration: 5760
model: Spec2DCNN
batch_size: 32
```

Benefits:
- No hardcoded values
- Easy to run sweeps: `python train.py -m downsample_rate=2,4,6,8`
- Every experiment auto-documented by its config

**2. Weights & Biases for tracking:**
- Automatic logging of metrics (loss, AP score)
- Hyperparameter tracking
- Model checkpoint management
- Visualization dashboards

**3. Git for version control:**
- Commit after each significant change
- Tag commits that correspond to successful experiments
- Can always revert to previous versions

**4. Structured directory layout:**
```
outputs/
  train/
    exp001/
      single/
        config.yaml
        best_model.pth
        metrics.csv
    exp002/
      ...
```

**5. Experiment log:**
I maintained a simple README noting:
- What I tried
- What worked/didn't work
- Hypotheses for next experiments

This system meant I could:
- Reproduce any experiment exactly
- Compare experiments systematically
- Roll back when something broke
- Collaborate (or interview) by sharing configs

**Real benefit:** Three months later, I could re-run my best model exactly because everything was tracked."

---

### Q14: "If you had more time, what would you improve?"

**Answer:**
"Great question! I'd focus on three areas:

**1. Ensemble methods:**
Currently using a single model. I'd train multiple models with different architectures or random seeds and combine predictions. Ensembles typically boost performance 2-5% by reducing variance.

**2. Advanced temporal modeling:**
- Experiment with Temporal Convolutional Networks (TCNs) for longer context
- Try attention mechanisms to explicitly learn which time windows matter
- Investigate bidirectional processing (use future context to refine past predictions)

**3. Domain-specific improvements:**
- Incorporate subject-level features (age, sex) if available
- Study actual sleep physiology literature for feature engineering ideas
- Analyze failure cases systematically—which events are we missing and why?

**4. Hyperparameter optimization:**
I did manual tuning and small sweeps. I'd use automated methods:
- Bayesian optimization (Optuna)
- Neural Architecture Search for decoder structure

**5. Efficiency improvements:**
- Quantization for faster inference
- Knowledge distillation (train small model to mimic large one)
- Optimized post-processing for real-time applications

**6. Production considerations:**
- Dockerize the entire pipeline
- Add input validation and error handling
- Create a simple API for making predictions
- Unit tests for critical functions

The key principle: prioritize based on impact vs. effort. Ensembles would give quick gains. NAS would be interesting research but high effort. Production work depends on deployment plans."

---

### Q15: "How does this project demonstrate your data science skills?"

**Answer:**
"This project showcases several key data science competencies:

**1. Problem Formulation:**
- Recognized this as time-series event detection, not simple classification
- Understood the evaluation metric and designed solution accordingly
- Translated business needs (pediatric sleep studies) to technical requirements

**2. Data Engineering:**
- Preprocessed 277+ multi-day recordings efficiently
- Engineered domain-relevant features (cyclic time encodings)
- Built scalable data pipeline handling variable-length sequences

**3. Machine Learning:**
- Applied modern deep learning architectures (CNNs, UNets)
- Addressed class imbalance through multiple techniques
- Implemented data augmentation for time-series
- Optimized for a complex metric (event detection AP)

**4. Software Engineering:**
- Modular, maintainable code architecture
- Configuration management for reproducibility
- Version control and experiment tracking
- Followed best practices (type hints, documentation)

**5. Experimental Rigor:**
- Systematic ablation studies
- Proper train/validation splitting
- Methodical debugging approach
- Hypothesis-driven experimentation

**6. Communication:**
- Can explain complex techniques to non-technical stakeholders
- Document decisions and tradeoffs
- Visualize results effectively

**7. Domain Awareness:**
- Understood biomedical context (circadian rhythms, sleep physiology)
- Incorporated domain knowledge into feature engineering
- Considered real-world deployment constraints

This wasn't just 'run a model and see what happens.' It required end-to-end thinking: problem understanding, data preparation, model design, evaluation, and iteration—the full data science lifecycle."

---

## Closing Thoughts

### How to Use This Guide

**For Interview Preparation:**
1. Read through each section to internalize the concepts
2. Practice explaining the CAR framework out loud
3. Adapt the Q&A responses to your own words—authenticity matters
4. Be ready to go deeper on any topic
5. Prepare to show code snippets or visualizations if possible

**Key Messages to Emphasize:**
- You can tackle real-world, complex problems
- You understand both theory and practical engineering
- You can make principled decisions and justify them
- You're comfortable with modern ML tools and best practices
- You can communicate technical concepts clearly

**Remember:**
- It's okay to say "I don't know, but here's how I'd find out"
- Connect your work to business impact when possible
- Show enthusiasm for learning and improving
- Be prepared with specific examples and numbers

---

## Additional Resources

**To Convert This to PDF:**

**Option 1 - Using Pandoc (Recommended):**
```bash
pandoc PROJECT_INTERVIEW_GUIDE.md -o PROJECT_INTERVIEW_GUIDE.pdf --pdf-engine=xelatex
```

**Option 2 - Using VS Code:**
1. Install "Markdown PDF" extension
2. Open this file
3. Right-click → "Markdown PDF: Export (pdf)"

**Option 3 - Online:**
- Use [Dillinger](https://dillinger.io/) or [StackEdit](https://stackedit.io/)
- Paste content and export to PDF

**For Presentation:**
Consider creating slides from sections using:
- [Marp](https://marp.app/) for markdown-based slides
- Google Slides with key points from CAR framework
- Jupyter notebook with code demonstrations

---

## Quick Reference Card

**Project Elevator Pitch (30 seconds):**
"I built a deep learning system to detect sleep transitions from wearable accelerometer data. Using a CNN-UNet architecture, I achieved 0.714 AP on the Kaggle leaderboard by addressing class imbalance through Gaussian label smoothing and negative sampling. The modular pipeline supports multiple model architectures and uses modern MLOps tools like PyTorch Lightning and Hydra for reproducibility."

**Key Technical Terms You Should Know:**
- Time-series segmentation
- Event detection Average Precision
- Gaussian label smoothing
- UNet architecture
- Class imbalance handling
- Mixup/Cutmix augmentation
- PyTorch Lightning
- Hydra configuration management
- Cyclic feature encoding

**Your Biggest Wins:**
1. Solved extreme class imbalance (0.1% events)
2. Achieved precise temporal localization despite long sequences
3. Built modular, reusable framework
4. Demonstrated end-to-end ML project lifecycle

**What Makes You Stand Out:**
- Not just using models—understanding WHY they work
- Systematic experimentation, not trial-and-error
- Production-oriented engineering practices
- Ability to communicate complex ideas simply

---

*Good luck with your interviews! Remember: you understand this project deeply. Trust in your preparation and your ability to explain your work clearly and confidently.*
