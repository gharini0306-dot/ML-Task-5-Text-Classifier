# Deep Learning Text Sentiment Classifier

## Project Overview

This project uses a PyTorch neural network to classify text reviews as positive or negative.

## Objective

To build a text classification pipeline using TF-IDF vectorization and a feed-forward neural network.

## Dataset

A small, manually created sample dataset of 30 positive and negative reviews was used for educational purposes.

## Technologies Used

* Python
* PyTorch
* Pandas
* Scikit-learn
* NumPy
* Matplotlib
* Joblib

## Methodology

1. Prepare text samples and sentiment labels.
2. Split the dataset into training and testing sets.
3. Convert text into numerical features using TF-IDF.
4. Train a neural network using PyTorch.
5. Evaluate predictions and plot training and validation curves.
6. Test the model on new text samples.

## Model Architecture

* Input layer
* Fully connected hidden layers
* ReLU activation
* Dropout
* Sigmoid applied to the output logits for binary classification

## Evaluation

The notebook reports test accuracy and a classification report. Training and validation loss and accuracy curves are also plotted.

## Saved Artifacts

* `models/sentiment_model.pth` — Trained model weights
* `models/tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer

## Limitations

The dataset is very small and intended only for demonstration. The reported metrics do not establish real-world performance. A larger, representative dataset and independent validation are needed before practical use.
