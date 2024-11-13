# Disaster Tweet Classification with NLP & Machine Learning

An advanced machine learning project that classifies tweets in real-time to determine if they indicate a disaster event. Built using Natural Language Processing (NLP), this project leverages various ML algorithms and a Flask-based web application for easy accessibility and deployment.

##  Features

* Comprehensive NLP Pipeline: Data cleaning, tokenization, vectorization with TF-IDF, and GloVe word embeddings.

* Diverse Model Ensemble: Evaluates multiple models, including Naive Bayes, Logistic Regression, Random Forest, and more. Gaussian Naive Bayes serves as the final model for classification.

* Sentiment Analysis Integration: Uses VADER and TextBlob to extract sentiment features from tweet text, enhancing prediction accuracy.

* User-Friendly Web Interface: Allows users to input tweet text and receive real-time predictions.

* Developed a web application: Developed a web application using Flask

## Project Overview

Disaster Tweet Classification is essential for monitoring social media platforms during emergencies. By filtering relevant information from noise, it supports emergency responders, organizations, and communities in managing resources effectively and responding faster.

### Project Structure

1. Data Preprocessing: Handles text cleaning, tokenization, stopword removal, stemming, and vectorization with TF-IDF.

2. Feature Engineering: Adds sentiment scores from VADER and TextBlob, as well as tweet structure features (hashtags, mentions, URLs).

3. Model Training: Trains, evaluates, and selects from multiple ML algorithms to ensure robust performance.

4. Web Deployment: A user-friendly Flask web app that predicts disaster tweets in real-time, deployed seamlessly on Heroku.

##  Tech Stack

* Languages & Libraries: Python, Flask, Scikit-Learn, NLTK, Gensim, TextBlob, Pandas, NumPy

* NLP & Machine Learning: TF-IDF Vectorization, GloVe Embeddings, Gaussian Naive Bayes

Web Deployment: Flask

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.x

### Installation

#### 1. Clone the repository:

git clone https://github.com/your-username/disaster-tweet-classification.git
cd disaster-tweet-classification

#### 2. Install dependencies:

pip install -r requirements.txt

#### 3. Download NLTK & GloVe resources:
Run the following Python script to download necessary NLTK and GloVe files:

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

#### 4. Set up GloVe embeddings: 
The app requires the GloVe 100-dimensional embeddings. Run this in your script or main file:

import gensim.downloader as gensim_api
glove = gensim_api.load("glove-twitter-100")

### Running the App Locally

#### 1. Start the Flask app:
python app.py
#### 2. Open your browser
and go to http://127.0.0.1:5000 to access the app.
