# Text Classification with TensorFlow

## Overview

This project implements a **binary sentiment classification model** using TensorFlow and TensorFlow Hub.

The model is trained on the **IMDB movie reviews dataset** and predicts whether a review is **positive or negative**.

Natural Language Processing (NLP) techniques are used to convert text into embeddings before feeding them into a neural network classifier.

---

## Dataset

The project uses the **IMDB Reviews dataset** provided by TensorFlow Datasets.

Dataset characteristics:

* 50,000 movie reviews
* Binary sentiment labels
* Balanced dataset

The dataset is automatically downloaded using `tensorflow_datasets`.

---

## Model Architecture

The model consists of:

1. **Input Layer**

   * Accepts raw text input

2. **Embedding Layer**

   * Pretrained text embedding from TensorFlow Hub
   * Converts text into numerical vectors

3. **Dense Hidden Layer**

   * 16 neurons with ReLU activation

4. **Output Layer**

   * Sigmoid activation for binary classification

---

## Technologies Used

* Python
* TensorFlow
* TensorFlow Hub
* TensorFlow Datasets
* NumPy

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install tensorflow tensorflow-hub tensorflow-datasets
```

---

## Running the Project

Run the training script:

```bash
python main.py
```

The script will:

1. Download the IMDB dataset
2. Preprocess the text data
3. Train a neural network classifier
4. Evaluate model performance

---

## Example Training Output

The model will output metrics such as:

```
Epoch 1/5
accuracy: 0.84
val_accuracy: 0.86
```

---

## Possible Improvements

Future improvements could include:

* Using advanced models such as **BERT**
* Adding dropout layers for regularization
* Hyperparameter tuning
* Saving and exporting the trained model
* Deploying the model as an API

---

## Author

Arul Gupta
BITS Goa – Electronics and Communication Engineering
