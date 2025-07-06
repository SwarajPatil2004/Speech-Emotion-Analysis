# Speech Emotion Recognition (SER)

A comprehensive project for **Speech Emotion Recognition** using four benchmark datasets: **RAVDESS**, **TESS**, **CREMA**, and **SAVEE**. The goal is to classify emotions from speech audio using state-of-the-art machine learning and deep learning techniques.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features & Methods](#features--methods)
- [Libraries & Tools](#libraries--tools)
- [Results & Evaluation](#results--evaluation)
- [License](#license)
- [References](#references)

---

## Overview

**Speech Emotion Recognition (SER)** aims to automatically identify human emotions from speech signals. This technology has applications in human-computer interaction, call centers, healthcare, and more. By leveraging multiple public datasets and robust feature extraction, this project provides a reproducible pipeline for SER research and development[^2][^4][^8].

---

## Datasets

This project uses the following datasets:

| Dataset | Description | Emotions Covered | Size/Actors | License |
|---------|-------------|------------------|-------------|---------|
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised | 24 actors, 7356 files | CC BY-NC-SA 4.0[^1][^3][^7][^9] |
| **TESS** | Toronto Emotional Speech Set | 7 emotions (e.g., Happy, Sad, Angry) | 2 actors, 2800 files | Research use |
| **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset | Anger, Disgust, Fear, Happy, Neutral, Sad | 91 actors, 7442 clips | Research use |
| **SAVEE** | Surrey Audio-Visual Expressed Emotion | Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise | 4 male actors, 480 files | Research use |

> **Note:** Please download each dataset from its official source and place it in the respective `data/` subfolders as per the directory structure.

---

## Project Structure

```

speech-emotion-recognition/
├── .env
├── Dataset/
│   ├── CREMA/
│   ├── RAVDESS/
│   ├── SAVEE/
│   └── TESS/
├── .gitignore
├── app.py
├── data_path.csv
├── README.md
├── requirements.txt
├── SEA_model.h5
├── Speech Emotion Recognition Sound Classification.ipynb
├── speech_emotion_analysis.py
└── utils.py


```

---

## Installation

1. **Clone the repository:**
```

git clone https://github.com/SwarajPatil2004/Speech-Emotion-Analysis
cd Speech-Emotion-Analysis

```

2. **Install required packages:**
```

pip install -r requirements.txt

```

3. **Download and organize datasets** as per the `Dataset/` directory structure above.

    Download Links for Speech Emotion Datasets
      - [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
      - [TESS - Toronto Emotional Speech Set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
      - [CREMA-D - Crowd-sourced Emotional Multimodal Actors Dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
      - [SAVEE - Surrey Audio-Visual Expressed Emotion](https://www.kaggle.com/datasets/ejlok1/savee-database)


---

## Usage

1. **Prepare the datasets:** Ensure all datasets are placed in their respective folders under `data/`.

2. **Run the main script:**
```

python main.py

```

3. **Jupyter Notebook:** For interactive exploration, open `main.ipynb` (if provided).

---

## Features & Methods

- **Audio Feature Extraction:**  
Utilizes **MFCCs**, **chroma features**, **mel spectrograms**, and more for robust emotion representation[^8].
- **Data Preprocessing:**  
- Silence removal
- Normalization
- Resampling
- **Modeling Approaches:**  
- Traditional ML: SVM, Random Forest, MLP[^4]
- Deep Learning: CNNs, RNNs (LSTM/GRU)[^2][^8]
- **Evaluation Metrics:**  
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix Visualization

---

## Libraries & Tools

The following libraries are used throughout the project:

| Library             | Purpose                                 |
|---------------------|-----------------------------------------|
| **pandas**          | Data manipulation and analysis          |
| **numpy**           | Numerical operations                    |
| **os**              | File and directory operations           |
| **seaborn**         | Statistical data visualization          |
| **matplotlib.pyplot** | Plotting graphs and figures           |
| **librosa**         | Audio processing and feature extraction |
| **librosa.display** | Visualization of audio features         |
| **IPython.display** | Audio playback in notebooks             |
| **warnings**        | Suppressing unnecessary warnings        |

```

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

```

> *Additional libraries such as scikit-learn, keras, and soundfile are used for modeling and audio I/O as needed.*

---

## Results & Evaluation

- **Performance:**  
  Models are evaluated on a held-out test set from each dataset. Performance metrics and confusion matrices are plotted for analysis.
- **Visualization:**  
  Feature distributions and model predictions are visualized using seaborn and matplotlib.

---

## License

- **RAVDESS:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)[^1].
- **Other datasets:** For research use only. Please refer to individual dataset licenses.

---

## References

- [RAVDESS Dataset][^1][^3][^7][^9]
- [TESS Dataset]
- [CREMA-D Dataset]
- [SAVEE Dataset]
- [Speech Emotion Recognition GitHub Projects][^2][^4][^6][^8]

---

*For any questions or contributions, please open an issue or pull request.*


