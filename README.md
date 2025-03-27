# ML-Digital-Twin

## Skoltech Machine Learning 2025 Final Project

### Project Description
This project is dedicated to creating a digital twin for relay protection using machine learning methods. Within the scope of the Machine Learning course, a model was developed to predict and monitor the state of relays, using various data processing techniques, such as binary classifiers, as well as more complex approaches based on recurrent neural networks (RNN/LSTM).

### Project Structure
The project consists of three main folders:

1. Baseline

The Baseline folder contains the basic model and the performance metrics used to evaluate and compare the developed models. 

2. Binary

The Binary folder includes all materials related to binary classifiers developed and applied within the project. It contains code, models, and other files related to binary classification methods for the relay protection task.

3. Multi

The Multi folder contains all the work related to the use of recurrent neural networks (RNN) and LSTM for more complex predictions and analyses. Here you will find all the materials and experiments related to multi-step forecasts, time series, and other approaches using RNN and LSTM.

Due to the large file size and GitHub limitations, some files were uploaded to drive:

### 1. Dataset

Dataset Link: https://drive.google.com/drive/folders/1KWtmV1qqF36b5aH3KAMECNyPtTqHNrBv?usp=sharing

### 2. Multi-Label Task Precomp

Link: https://drive.google.com/drive/folders/140vtzV2U9MAjyOtj6aDX7pxNHxysCR4Q?usp=sharing

### Overview
This repository provides utilities and models for deploying a machine learning model based on Wavelet Embedding. It includes dataset preprocessing, embedding computation, and model training.

### Utility Classes (utils.py)
The utils.py file contains essential utility classes:

WindowSamplerTorch() – A vectorized class for extracting windows with a given stride and start index.

DatasetWindowed() – A subclass of torch.utils.data.Dataset, which returns a sequence of windows for each unit in the dataset.

PrecomputedWaveletEmbeddingDataset() – A dataset class that returns precomputed wavelet embeddings instead of raw windows.

TempRWE_Encoder() – A class that computes Temporal Relative Wavelet Energy (RWE) Embeddings for each window in the sequence of a dataset unit.

Preparing Data for Wavelet Embedding Model
To deploy the model with wavelet embeddings, the data must be structured as:


[data]:  [num_units, seq_len]
[labels]: [num_units, seq_len]
Step 1: Initialize the Encoder
Import TempRWE_Encoder from utils.py and initialize it:
```
from utils import TempRWE_Encoder

encoder = TempRWE_Encoder(maxcurrent=500, wavename="rbio1.3", maxlevel=6, verbosity=0)
```
Step 2: Create a Dataset with Precomputed Embeddings
Import PrecomputedWaveletEmbeddingDataset and create a dataset object:

```
from utils import PrecomputedWaveletEmbeddingDataset

train_embs = PrecomputedWaveletEmbeddingDataset(
    data, labels, encoder, wsize=1024, stride=1, start_idx=80*20
)
Note: Precomputing embeddings takes some time.
```

Step 3: Create a DataLoader
Use torch.utils.data.DataLoader to create a data loader:

```
from torch.utils.data import DataLoader

loader_train = DataLoader(train_embs, batch_size=32, shuffle=False)
```
### Model Inference
The models are available in corresponding .ipynb files.

Load a batch from the DataLoader.

Pass it to the model and obtain logits.

Convert logits to probabilities using ```torch.sigmoid()```:
```
import torch

logits = model(batch)
probabilities = torch.sigmoid(logits)
```
Baseline Model: RMS Hysteresis Estimator
To work with the baseline model, import RMSHisteresisEstimator from utils.py:

from utils import RMSHisteresisEstimator
This model follows a model-based approach, relying on general protection logics to create a digital twin.

The demonstration .ipynb file provides:

A guide on using RMSHisteresisEstimator

Performance metrics on the test dataset

### Binary Classification with XGBoost
In the binary/ directory, you will find a .ipynb notebook where we trained an XGBoost Classifier using Temporal RWE Embeddings computed over the full sequence length.

### Available Resources:
Performance metrics for the best model (determined via GridSearch)

results.joblib: A joblib dump of the GridSearchCV object

### Conclusion
This repository provides the necessary tools to preprocess data, compute wavelet embeddings, and train models efficiently. Explore the notebooks and experiment with different configurations to optimize performance. 🚀








