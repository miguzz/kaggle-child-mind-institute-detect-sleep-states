# Complete Setup and Run Guide

> **For architecture details, see [REPOSITORY_OVERVIEW.md](REPOSITORY_OVERVIEW.md)**
> **For interview preparation, see [PROJECT_INTERVIEW_GUIDE.md](PROJECT_INTERVIEW_GUIDE.md)**

This is your single source for setting up and running the sleep detection project locally.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (30 Minutes)](#quick-setup-30-minutes)
3. [Running in Development Mode](#running-in-development-mode)
4. [Understanding the Output](#understanding-the-output)
5. [Experimenting with Parameters](#experimenting-with-parameters)
6. [Running Full Training](#running-full-training)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)

---

## What You'll Accomplish

- ✅ Set up Python environment with all dependencies (using existing venv_dev)
- ✅ Download and preprocess competition data
- ✅ Run training on a small development dataset (5 series, 2 epochs, ~5-10 minutes)
- ✅ Understand how to experiment with different configurations
- ✅ Scale up to full training

---

## Prerequisites

- **Python 3.10+** installed
- **~10GB free disk space**
- **Kaggle account** (free)
- **Internet connection**
- **CPU or GPU** (CPU works, GPU is faster)

---

## Quick Setup (30 Minutes)

### Step 1: Install Rye (Python Package Manager)

Rye is a modern Python package manager that handles virtual environments and dependencies.

**Windows:**
1. Visit https://rye-up.com/guide/installation/
2. Download and run the Windows installer
3. Restart your terminal

**Verify:**
```bash
rye --version
# Should show: rye x.x.x
```

**Estimated time:** 2-3 minutes

---

### Step 2: Install Dependencies

Navigate to project and install all packages:

```bash
# Navigate to project root
cd "c:\Users\Miickael Alvarez\Documents\DataScience\Kaggle\Child Mind Institute - Detect Sleep States\kaggle-child-mind-institute-detect-sleep-states"

# Install all dependencies (creates .venv automatically)
rye sync
```

This creates a `.venv` folder and installs:
- PyTorch 2.1.0
- PyTorch Lightning 2.0.9
- Hydra 1.3.2
- torchaudio 2.1.0
- transformers 4.33.3
- And 30+ other packages

**Estimated time:** 5-10 minutes

---

### Step 3: Activate Virtual Environment

Every time you work on this project, activate the environment:

```bash
# Windows
.venv\Scripts\activate

# You should see (.venv) in your prompt
```

**Verify:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Should show: PyTorch 2.1.0 (or similar)
```

---

### Step 4: Set Up Kaggle API

Download competition data using Kaggle API.

#### 4.1 Get Your API Token

1. Go to https://www.kaggle.com/ and log in
2. Click your profile picture → **Account**
3. Scroll to **API** section → **"Create New Token"**
4. This downloads `kaggle.json`

#### 4.2 Install the Token

**Windows:**
```bash
# Create Kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move the downloaded token
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# Verify it's there
dir %USERPROFILE%\.kaggle\
```

**Estimated time:** 2 minutes

---

### Step 6: Download Competition Data

Now download the dataset from Kaggle:

```bash
# Create data directory
mkdir data
cd data

# Download competition data (~1GB compressed)
kaggle competitions download -c child-mind-institute-detect-sleep-states

# Unzip the data
unzip child-mind-institute-detect-sleep-states.zip

# Go back to project root
cd ..
```

**Expected output:**
```
Downloading child-mind-institute-detect-sleep-states.zip to data
100%|████████████████████| 1.02G/1.02G [XX:XX<00:00, XXMiB/s]
Archive:  child-mind-institute-detect-sleep-states.zip
  extracting: train_series.parquet
  extracting: train_events.csv
  extracting: test_series.parquet
  extracting: sample_submission.csv
```

**You should now have:**
```
data/
├── train_series.parquet    (~900MB - sensor readings)
├── train_events.csv        (~50KB - ground truth labels)
├── test_series.parquet     (~400MB - test data)
└── sample_submission.csv
```

**Estimated time:** 3-5 minutes (depending on internet speed)

---

### Step 7: Verify Complete Setup

Run the verification script again to confirm everything is ready:

```bash
python check_setup.py
```

**Expected output:**
```
============================================================
SUMMARY
============================================================

✓ All checks passed! You're ready to run the project.

Next step:
  python run/prepare_dev.py dir=local
```

**Estimated time:** 1 minute

---

## Running in Development Mode

Now you're ready to run the project! We'll start with a small development dataset to verify everything works before running full training.

### Step 8: Create Development Dataset

Instead of using all 277 recordings, we'll create a small subset of 200 random samples:

```bash
python run/prepare_dev.py dir=local
```

**What this does:**
- Randomly samples 200 series from the full training dataset
- Creates `processed_data/dev_series.parquet`
- Creates `processed_data/dev_events.csv`

**Expected output:**
```
Done
```

**Estimated time:** 1-2 minutes

---

### Step 9: Preprocess Development Data

Now preprocess the development data (feature engineering):

```bash
python run/prepare_data.py dir=local phase=dev
```

**What this does:**
- Loads the 200 dev series
- Normalizes sensor values (anglez, enmo)
- Creates cyclic temporal features:
  - `hour_sin`, `hour_cos` (24-hour cycle)
  - `month_sin`, `month_cos` (12-month cycle)
  - `minute_sin`, `minute_cos` (60-minute cycle)
  - `anglez_sin`, `anglez_cos` (angular encoding)
- Saves each feature as a separate `.npy` file per series

**Expected output:**
```
Load series: 100%|████████████████| 200/200
Save features: 100%|████████████████| 200/200
```

**Directory structure created:**
```
processed_data/dev/
├── <series_id_1>/
│   ├── anglez.npy
│   ├── enmo.npy
│   ├── hour_sin.npy
│   ├── hour_cos.npy
│   ├── step.npy
│   └── ...
├── <series_id_2>/
│   └── ...
└── ... (200 total)
```

**Estimated time:** 5-10 minutes

---

### Step 10: Run Your First Training! 🚀

Now for the exciting part - train your first model on a **tiny subset** (5 series, 2 epochs):

```bash
python run/train.py --config-name=train_dev dir=local
```

**What this configuration does:**
- Uses only **5 series** (3 training, 2 validation) defined in `dev_tiny.yaml`
- Trains for **2 epochs** (instead of 50)
- Uses **reduced duration** (2880 = 4 hours vs default 5760 = 8 hours)
- **Aggressive downsampling** (rate=4 vs default=2) for speed
- **Smaller batch size** (8 vs 32) to reduce memory usage

**Expected output:**

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Layer (type)               Output Shape     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ CNNSpectrogram                  [8, 64, 128, 720] │
│ UNet1DDecoder                   [8, 720, 3]       │
└────────────────────────────────────────────────────┘
Total params: 2.1M
Trainable params: 2.1M

Epoch 1/2
Training:   0%|                                    | 0/12 [00:00<?, ?it/s]
Training:  50%|████████████                | 6/12 [01:15<01:15, 12.5s/it]
Training: 100%|████████████████████████████| 12/12 [02:34<00:00, 12.8s/it]
train_loss: 0.543

Validation:   0%|                                  | 0/3 [00:00<?, ?it/s]
Validation: 100%|████████████████████████████| 3/3 [00:45<00:00, 15.2s/it]
val_loss: 0.421 | val_score: 0.156

Epoch 2/2
Training: 100%|████████████████████████████| 12/12 [02:31<00:00, 12.6s/it]
train_loss: 0.398

Validation: 100%|████████████████████████████| 3/3 [00:43<00:00, 14.8s/it]
val_loss: 0.376 | val_score: 0.234

Saved best model: output/train/dev_test/single/best_model.pth
Extracting and saving best weights: model_weights.pth
```

**What you're seeing:**
- **Model architecture**: CNNSpectrogram + UNet1D with 2.1M parameters
- **Training progress**: 12 batches per epoch
- **Validation progress**: 3 batches
- **Metrics**:
  - `train_loss`: Binary cross-entropy loss (lower is better)
  - `val_loss`: Validation loss (lower is better)
  - `val_score`: Event detection Average Precision (higher is better, range 0-1)

**Estimated time:** 5-10 minutes (GPU) or 20-30 minutes (CPU)

**🎉 Congratulations!** You've just trained a sleep detection model!

---

## Understanding the Output

### Output Files

After training completes, check the output directory:

```bash
dir output\train\dev_test\single
```

**Files created:**

```
output/train/dev_test/single/
├── .hydra/                    # Hydra configuration logs
│   └── config.yaml           # Exact config used for this run
├── wandb/                     # Weights & Biases logs (if enabled)
│   └── ...
├── config.yaml                # Training configuration
├── model_weights.pth          # Final model weights (PyTorch state dict)
├── best_model.pth            # Best model checkpoint (full Lightning module)
├── keys.npy                   # Validation series IDs
├── labels.npy                 # Ground truth labels [shape: (N, duration, 2)]
├── preds.npy                  # Model predictions [shape: (N, duration, 2)]
└── val_pred_df.csv           # Validation predictions in submission format
```

### Inspecting Results

**Check prediction shapes:**
```bash
python -c "import numpy as np; print('Predictions shape:', np.load('output/train/dev_test/single/preds.npy').shape)"
# Output: Predictions shape: (6, 2880, 2)
# 6 validation chunks, 2880 timesteps each, 2 classes (onset, wakeup)
```

**View validation predictions:**
```bash
python -c "import pandas as pd; df = pd.read_csv('output/train/dev_test/single/val_pred_df.csv'); print(df.head())"
```

**Check configuration used:**
```bash
type output\train\dev_test\single\.hydra\config.yaml
```

---

## Experimenting with Parameters

Now that you have a working setup, you can experiment with different configurations.

### Understanding the Configuration System

This project uses **Hydra** for configuration management. All settings are in YAML files under `run/conf/`:

```
run/conf/
├── train.yaml              # Main training config
├── train_dev.yaml          # Development config (what you just used)
├── dir/
│   ├── local.yaml         # Your local paths ✓
│   └── kaggle.yaml        # Kaggle notebook paths
├── split/
│   ├── fold_0.yaml        # Full train/val split (277 series)
│   └── dev_tiny.yaml      # Tiny split (5 series) ✓
├── model/
│   ├── Spec1D.yaml
│   ├── Spec2DCNN.yaml     # Default model
│   ├── CenterNet.yaml
│   └── ...
├── feature_extractor/
│   ├── CNNSpectrogram.yaml  # Default
│   ├── LSTMFeatureExtractor.yaml
│   └── ...
├── decoder/
│   ├── UNet1DDecoder.yaml   # Default
│   ├── LSTMDecoder.yaml
│   └── ...
└── dataset/
    ├── seg.yaml             # Segmentation dataset (default)
    └── ...
```

### Override Parameters from Command Line

You can override any parameter without modifying files:

**Try different downsampling rate:**
```bash
python run/train.py \
  --config-name=train_dev \
  downsample_rate=2 \
  exp_name=dev_test_ds2 \
  dir=local
```

**Try different model architecture:**
```bash
python run/train.py \
  --config-name=train_dev \
  model=Spec1D \
  decoder=LSTMDecoder \
  exp_name=dev_test_lstm \
  dir=local
```

**Increase epochs:**
```bash
python run/train.py \
  --config-name=train_dev \
  trainer.epochs=5 \
  exp_name=dev_test_5epochs \
  dir=local
```

**Change batch size (if memory issues):**
```bash
python run/train.py \
  --config-name=train_dev \
  batch_size=4 \
  exp_name=dev_test_bs4 \
  dir=local
```

### Run Multiple Experiments (Parameter Sweep)

Test multiple configurations automatically:

```bash
# Test 3 different downsampling rates
python run/train.py \
  --config-name=train_dev \
  dir=local \
  -m downsample_rate=2,4,8
```

This runs 3 separate experiments with different `downsample_rate` values.

### Available Model Configurations

**Models:**
- `Spec1D`: 1D segmentation without UNet
- `Spec2DCNN`: 2D CNN with UNet (default, best performance)
- `CenterNet`: Object detection approach
- `DETR2DCNN`: Detection transformer

**Feature Extractors:**
- `CNNSpectrogram`: Multi-scale CNN (default)
- `LSTMFeatureExtractor`: Sequential LSTM
- `PANNsFeatureExtractor`: Pre-trained audio network
- `SpecFeatureExtractor`: Spectrogram-based

**Decoders:**
- `UNet1DDecoder`: Encoder-decoder with skip connections (default)
- `LSTMDecoder`: Recurrent decoder
- `MLPDecoder`: Simple fully-connected
- `TransformerDecoder`: Attention-based
- `TransformerCNNDecoder`: Hybrid approach

**Example - Try CenterNet:**
```bash
python run/train.py \
  --config-name=train_dev \
  model=CenterNet \
  dataset=centernet \
  feature_extractor=CNNSpectrogram \
  decoder=UNet1DDecoder \
  exp_name=dev_centernet \
  dir=local
```

---

## Running Full Training

Once you're comfortable with the development setup, you can scale up.

### Option 1: Train on Full Dev Set (200 Series)

Use the larger dev set but still faster than full training:

```bash
python run/train.py \
  exp_name=dev_full_200 \
  split=fold_0 \
  trainer.epochs=10 \
  dir=local
```

**Note:** `split=fold_0` uses more series than `dev_tiny`. Make sure to check what series IDs are in `fold_0.yaml`.

**Estimated time:** 1-2 hours

### Option 2: Preprocess and Train on Full Dataset (277 Series)

**Step 1: Preprocess full training data:**
```bash
python run/prepare_data.py dir=local phase=train
```

This processes all 277 training series. **Estimated time:** 15-20 minutes

**Step 2: Run full training:**
```bash
python run/train.py \
  exp_name=full_training \
  downsample_rate=2 \
  duration=5760 \
  trainer.epochs=50 \
  dir=local
```

**Estimated time:** Several hours (depends on GPU)

### Reproduce Leaderboard Score (0.714)

To reproduce the baseline result mentioned in the README:

```bash
python run/train.py \
  downsample_rate=2 \
  duration=5760 \
  exp_name=exp001 \
  dataset.batch_size=32 \
  dir=local
```

Then for inference:
```bash
python run/inference.py \
  dir=local \
  exp_name=exp001 \
  weight.run_name=single \
  downsample_rate=2 \
  duration=5760 \
  model.params.encoder_weights=null \
  pp.score_th=0.005 \
  pp.distance=40 \
  phase=test
```

---

## Troubleshooting

### Common Issues and Solutions

#### ❌ "CUDA out of memory"

**Cause:** GPU doesn't have enough memory for the batch size.

**Solutions:**
```bash
# Option 1: Reduce batch size
python run/train.py --config-name=train_dev batch_size=4 dir=local

# Option 2: Use CPU (slower but works)
python run/train.py --config-name=train_dev trainer.accelerator=cpu dir=local

# Option 3: Reduce duration
python run/train.py --config-name=train_dev duration=1440 dir=local

# Option 4: Increase downsampling
python run/train.py --config-name=train_dev downsample_rate=8 dir=local
```

---

#### ❌ "ModuleNotFoundError: No module named 'src'"

**Cause:** Virtual environment not activated or wrong directory.

**Solutions:**
```bash
# Ensure you're in project root
cd "c:\Users\Miickael Alvarez\Documents\DataScience\Kaggle\Child Mind Institute - Detect Sleep States\kaggle-child-mind-institute-detect-sleep-states"

# Activate virtual environment
.venv\Scripts\activate

# Verify
python -c "import src; print('OK')"
```

---

#### ❌ "FileNotFoundError: [data_dir]/train_series.parquet"

**Cause:** Data not downloaded or paths incorrect.

**Solutions:**

1. **Check data exists:**
```bash
dir data
# Should show: train_series.parquet, train_events.csv, etc.
```

2. **Check paths in config:**
```bash
type run\conf\dir\local.yaml
# Verify data_dir points to correct location
```

3. **Download data if missing:**
```bash
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
cd ..
```

---

#### ⚠️ "Wandb login required"

**Cause:** Weights & Biases wants you to log in for experiment tracking.

**Solutions:**

**Option 1: Login (recommended for tracking experiments):**
```bash
wandb login
# Follow prompts to create free account
```

**Option 2: Run offline:**
```bash
set WANDB_MODE=offline
python run/train.py --config-name=train_dev dir=local
```

**Option 3: Disable wandb:**
Edit `run/train.py` and comment out the WandbLogger initialization.

---

#### ❌ Training is very slow on CPU

**Cause:** CPU training is 10-50x slower than GPU.

**Solutions:**

1. **Use even smaller config:**
```bash
python run/train.py \
  --config-name=train_dev \
  duration=1440 \
  downsample_rate=8 \
  batch_size=4 \
  dir=local
```

2. **Use Google Colab (free GPU):**
   - Upload your code to Colab
   - Change `dir=kaggle` in commands
   - Use their free GPU

3. **Be patient:** First run on CPU is fine for verification

---

#### ❌ "RuntimeError: Ninja is required to load C++ extensions"

**Cause:** Missing build tools for PyTorch extensions.

**Solutions:**

**Windows:**
- Install Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
- Or ignore (usually not critical for basic usage)

---

## Project Structure

Understanding the directory layout helps you navigate the code:

```
kaggle-child-mind-institute-detect-sleep-states/
│
├── 📄 README.md                           # Original project README
├── 📄 QUICKSTART.md                       # Your quick start guide
├── 📄 REPOSITORY_OVERVIEW.md              # Architecture explanation
├── 📄 PROJECT_INTERVIEW_GUIDE.md          # Interview preparation
├── 🔧 check_setup.py                      # Setup verification script
│
├── 📊 data/                               # Downloaded competition data
│   ├── train_series.parquet              # Raw accelerometer data
│   ├── train_events.csv                  # Ground truth sleep events
│   ├── test_series.parquet
│   └── sample_submission.csv
│
├── 💾 processed_data/                     # Preprocessed features
│   ├── dev/                              # Dev set (200 series)
│   │   ├── <series_id_1>/
│   │   │   ├── anglez.npy
│   │   │   ├── enmo.npy
│   │   │   ├── hour_sin.npy
│   │   │   └── ...
│   │   └── ...
│   └── train/                            # Full training set (277 series)
│       └── ...
│
├── 📈 output/                             # Training outputs
│   └── train/
│       ├── dev_test/                     # Your dev experiments
│       │   └── single/
│       │       ├── config.yaml
│       │       ├── model_weights.pth
│       │       └── ...
│       └── <other_experiments>/
│
├── 🏃 run/                                # Executable scripts
│   ├── train.py                          # Main training script
│   ├── inference.py                      # Inference/prediction script
│   ├── prepare_data.py                   # Data preprocessing
│   ├── prepare_dev.py                    # Create dev dataset
│   └── conf/                             # Configuration files (Hydra)
│       ├── train.yaml                    # Main config
│       ├── train_dev.yaml                # Dev config ✓
│       ├── inference.yaml
│       ├── prepare_data.yaml
│       ├── dir/
│       │   └── local.yaml                # Your paths ✓
│       ├── split/
│       │   ├── fold_0.yaml               # Full split
│       │   └── dev_tiny.yaml             # Tiny split ✓
│       ├── model/
│       │   ├── Spec1D.yaml
│       │   ├── Spec2DCNN.yaml            # Default
│       │   └── ...
│       ├── feature_extractor/
│       │   └── ...
│       ├── decoder/
│       │   └── ...
│       └── dataset/
│           └── ...
│
├── 🧠 src/                                # Source code
│   ├── models/                           # Model architectures
│   │   ├── base.py                       # Base model class
│   │   ├── spec1D.py                     # Spec1D model
│   │   ├── spec2Dcnn.py                  # Spec2DCNN model
│   │   ├── centernet.py                  # CenterNet model
│   │   ├── feature_extractor/            # Feature extractors
│   │   │   ├── cnn.py
│   │   │   ├── lstm.py
│   │   │   └── ...
│   │   └── decoder/                      # Decoders
│   │       ├── unet1ddecoder.py
│   │       ├── lstmdecoder.py
│   │       └── ...
│   ├── dataset/                          # Dataset classes
│   │   ├── seg.py                        # Segmentation dataset
│   │   ├── centernet.py                  # CenterNet dataset
│   │   └── common.py
│   ├── augmentation/                     # Data augmentation
│   │   ├── mixup.py
│   │   └── cutmix.py
│   ├── utils/                            # Utilities
│   │   ├── metrics.py                    # Event detection AP
│   │   ├── post_process.py               # Post-processing
│   │   └── common.py
│   ├── datamodule.py                     # PyTorch Lightning DataModule
│   ├── modelmodule.py                    # PyTorch Lightning ModelModule
│   └── conf.py                           # Configuration dataclasses
│
├── 📓 notebook/                           # Jupyter notebooks (if any)
│
├── 🛠️ tools/                              # Utility scripts
│   └── upload_dataset.py                 # Kaggle dataset upload
│
├── 🐍 .venv/                              # Virtual environment (created by rye)
│
├── 📦 pyproject.toml                      # Project dependencies
├── 🔒 requirements.lock                   # Locked dependencies
└── ⚙️ .python-version                     # Python version (3.10)
```

### Key Files to Understand

**Configuration:**
- `run/conf/train_dev.yaml` - Development training config you'll use most
- `run/conf/dir/local.yaml` - Your local paths
- `run/conf/split/dev_tiny.yaml` - Minimal dataset split (5 series)

**Source Code:**
- `src/modelmodule.py` - Training loop, validation, metrics
- `src/datamodule.py` - Data loading and batching
- `src/models/spec2Dcnn.py` - Default model architecture
- `src/dataset/seg.py` - Dataset class with label generation
- `src/utils/metrics.py` - Event detection Average Precision metric

**Scripts:**
- `run/train.py` - Main entry point for training
- `run/prepare_data.py` - Preprocessing pipeline
- `run/inference.py` - Generate predictions

---

## Summary: Complete Workflow

Here's the complete workflow from setup to training:

```bash
# ONE-TIME SETUP
# ==============

# 1. Install Rye
# (Download from https://rye-up.com/guide/installation/)

# 2. Install dependencies
rye sync
.venv\Scripts\activate

# 3. Verify setup
python check_setup.py

# 4. Set up Kaggle API
# (Download kaggle.json from Kaggle account settings)
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# 5. Download data
mkdir data
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
cd ..

# 6. Create and preprocess dev dataset
python run/prepare_dev.py dir=local
python run/prepare_data.py dir=local phase=dev

# EVERY TIME YOU WANT TO TRAIN
# =============================

# 1. Activate environment (if not already active)
.venv\Scripts\activate

# 2. Run training
python run/train.py --config-name=train_dev dir=local

# 3. Check results
dir output\train\dev_test\single

# 4. Experiment with parameters
python run/train.py --config-name=train_dev downsample_rate=2 exp_name=experiment1 dir=local
```

---

## What's Next?

### Learning Path

1. **You are here:** ✅ Project runs successfully in dev mode
2. **Next:** Read [REPOSITORY_OVERVIEW.md](REPOSITORY_OVERVIEW.md) to understand the architecture
3. **Then:** Experiment with different parameters and models
4. **After:** Study [PROJECT_INTERVIEW_GUIDE.md](PROJECT_INTERVIEW_GUIDE.md) for interview prep
5. **Finally:** Run full training and analyze results

### Experiment Ideas

- **Try different downsampling rates:** 2, 4, 6, 8
- **Test different models:** Spec1D, CenterNet, DETR2DCNN
- **Adjust duration:** 1440, 2880, 5760
- **Enable data augmentation:** Set `aug.mixup_prob=0.5`
- **Try different optimizers:** Modify `optimizer.lr`
- **Change post-processing:** Adjust `pp.score_th` and `pp.distance`

### Monitoring Training

**Terminal output** shows real-time progress.

**Weights & Biases** (optional but recommended):
```bash
wandb login
# Then run training
# View experiments at: https://wandb.ai/your-username/child-mind-institute-detect-sleep-states
```

---

## Final Checklist

Before you start experimenting, make sure:

- [ ] Rye installed and working
- [ ] Virtual environment activated (`(.venv)` in prompt)
- [ ] All packages installed (run `python check_setup.py`)
- [ ] Data downloaded to `data/`
- [ ] Dev dataset created (`processed_data/dev/` exists)
- [ ] First training completed successfully
- [ ] Output files generated in `output/train/dev_test/single/`

**If all checked:** 🎉 You're ready to experiment and learn!

**If any issues:** Check the [Troubleshooting](#troubleshooting) section above.

---

## Additional Resources

- **Hydra Documentation:** https://hydra.cc/docs/intro/
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/stable/
- **Competition Page:** https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states
- **Discussion Thread:** https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/452940

---

**You're all set!** 🚀 Happy training and learning!
