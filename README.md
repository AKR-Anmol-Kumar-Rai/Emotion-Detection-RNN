# Emotion Detection using RNN

A deep learning-based **Emotion Detection System** built using **Recurrent Neural Networks (RNN)**, **TF-IDF vectorization**, and **PyTorch** for multi-class text emotion classification. This project takes user input text, preprocesses it, and predicts the emotional state in real time through a deployed **Streamlit** web application.

## Live Demo

🚀 **Try the App Here:**
https://emotion-detection-rnn.streamlit.app/

---

## Project Overview

Emotion detection is an important Natural Language Processing (NLP) task used to identify human emotions from textual data.

This project classifies text into six emotional categories:

* Sadness
* Joy
* Love
* Anger
* Fear
* Surprise

The system uses a full machine learning pipeline:

```text
Text Input
→ Preprocessing
→ TF-IDF Vectorization
→ RNN Model
→ Dropout Regularization
→ Fully Connected Layer
→ Emotion Prediction
```

---

## Features

* Real-time emotion detection
* Text preprocessing pipeline
* TF-IDF based vectorization
* RNN-based sequence classification
* Streamlit web deployment
* Interactive premium UI
* GitHub integrated deployment
* Supports six emotion classes

---

## Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* PyTorch
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* Joblib

---

## Model Architecture

The model follows a **Many-to-One RNN architecture**.

### Architecture Flow:

```text
Input Text
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
RNN Layer (2 Layers)
↓
Dropout Layer (0.3)
↓
Fully Connected Layer
↓
Output Layer (6 Classes)
```

### Hyperparameters:

| Parameter         | Value            |
| ----------------- | ---------------- |
| Input Size        | 15154            |
| Hidden Layer Size | 500              |
| Number of Layers  | 2                |
| Output Classes    | 6                |
| Dropout           | 0.3              |
| Optimizer         | Adam             |
| Loss Function     | CrossEntropyLoss |

---

## Dataset

The model was trained on an emotion classification dataset containing labeled text samples across six emotional categories.

Dataset split:

* Training Set
* Validation Set
* Test Set

---

## Preprocessing Pipeline

The following preprocessing steps were applied:

* URL removal
* HTML tag removal
* Punctuation removal
* Stopword removal
* Stemming

This helps clean textual data before vectorization.

---

## Model Performance

### Training Accuracy:

**~79%**

### Testing Accuracy:

**~75%**

The model performs well for a beginner-level NLP deep learning project and generalizes effectively on unseen data.

---

## Project Structure

```text
emotion-detection-rnn/
│── app.py
│── model.py
│── utils.py
│── emotion_model.pth
│── tfidf.pkl
│── requirements.txt
│── README.md
│── train.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd emotion-detection-rnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

---

## How It Works

Example:

Input:

```text
I feel so happy today
```

Preprocessing:

```text
feel happi today
```

TF-IDF converts text into numerical vectors.

RNN processes the sequence and predicts:

```text
JOY
```

---

## Limitations

Current model limitations:

* Uses TF-IDF instead of embeddings
* Simple RNN may struggle with long-term dependencies
* Sometimes misclassifies ambiguous or negated sentences

Example:

```text
I am not happy
```

This may be difficult due to TF-IDF limitations.

---

## Future Improvements

Planned upgrades:

* Replace TF-IDF with word embeddings
* Implement LSTM
* Implement GRU
* Add confidence score visualization
* Improve preprocessing for negation handling
* Deploy advanced transformer-based version

---

## Learning Outcomes

Through this project, I learned:

* Text preprocessing in NLP
* TF-IDF vectorization
* RNN architecture and hidden states
* Many-to-One sequence modeling
* Model training and evaluation
* Streamlit deployment
* Git LFS for large model handling
* End-to-end NLP project deployment

---

## Author

Developed by **Anmol** as part of deep learning and NLP practice.

This project represents an end-to-end implementation of text emotion classification from training to deployment.
