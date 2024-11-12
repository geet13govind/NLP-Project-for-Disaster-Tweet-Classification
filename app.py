# from flask import Flask, request, render_template, jsonify
# import pickle
# import pandas as pd
# import numpy as np
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
# import gensim.downloader as gensim_api

# # Load the saved model, vectorizer, and scaler
# with open("gaussian_nb_model.pkl", "rb") as file:
#     model = pickle.load(file)
# with open("tfidf_vectorizer.pkl", "rb") as file:
#     tfidf_vectorizer = pickle.load(file)
# with open("scaler.pkl", "rb") as file:
#     scaler = pickle.load(file)

# # Load pre-trained GloVe embeddings (100-dimensional)
# glove = gensim_api.load("glove-twitter-100")

# # Initialize the Flask app
# app = Flask(__name__)

# def vectorized_tweet(text):
#     # Text cleaning and tokenization
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'#', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()

#     stop_words = set(stopwords.words('english'))
#     tokens = nltk.word_tokenize(text)
#     tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
#     return ' '.join(tokens)  # Join tokens into a single string

# def get_embedding(text, embedding_size=100):
#     words = text.split()
#     word_embeddings = [glove[word] for word in words if word in glove]
#     if not word_embeddings:
#         return np.zeros(embedding_size)
#     return np.mean(word_embeddings, axis=0)

# def extract_features(text):
#     from nltk.sentiment.vader import SentimentIntensityAnalyzer
#     from textblob import TextBlob
#     import nltk
#     nltk.download('vader_lexicon')
#     nltk.download('stopwords')
#     nltk.download('punkt')
    
#     # Cleaned and tokenized text as single string
#     cleaned_text = vectorized_tweet(text)
    
#     # Basic text features
#     features = {
#         'text_length': len(text),
#         'num_words': len(text.split()),
#         'avg_word_length': np.mean([len(word) for word in text.split()]),
#         'hashtag_count': text.count('#'),
#         'mention_count': text.count('@'),
#         'url_count': text.count('http'),
#         'numeric_count': sum(1 for word in text.split() if word.isdigit()),
#         'exclamation_count': text.count('!'),
#         'question_count': text.count('?'),
#         'punctuation_count': sum(1 for char in text if char in '.,;!?')
#     }

#     # Sentiment features using VADER and TextBlob
#     sia = SentimentIntensityAnalyzer()
#     vader_scores = sia.polarity_scores(cleaned_text)
#     features.update({
#         'vader_negative': vader_scores['neg'],
#         'vader_neutral': vader_scores['neu'],
#         'vader_positive': vader_scores['pos'],
#         'vader_compound': vader_scores['compound'],
#         'textblob_polarity': TextBlob(cleaned_text).sentiment.polarity,
#         'textblob_subjectivity': TextBlob(cleaned_text).sentiment.subjectivity
#     })
    
#     # Convert to DataFrame
#     feature_df = pd.DataFrame([features])
    
#     # TF-IDF features
#     tfidf_features = tfidf_vectorizer.transform([cleaned_text])
#     tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

#     # Embedding features
#     embedding_features = pd.DataFrame([get_embedding(cleaned_text)], columns=[f'embed_{i}' for i in range(100)])
    
#     # Combine all features
#     combined_features = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True), embedding_features.reset_index(drop=True)], axis=1)
    
#     # Scale combined features
#     scaled_features = scaler.transform(combined_features)
#     return scaled_features

# # Home route to display the form
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Prediction route to handle user input
# @app.route('/predict', methods=['POST'])
# def predict():
#     tweet_text = request.form['tweet']

#     # Extract features
#     features = extract_features(tweet_text)
#     # Predict
#     prediction = model.predict(features)
#     result = "Disaster" if prediction[0] == 1 else "Non-Disaster"

#      # Render result in template
#     return render_template('index.html', tweet=tweet_text, prediction=result)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import gensim.downloader as gensim_api

# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Load the saved model, vectorizer, and scaler
with open("gaussian_nb_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load pre-trained GloVe embeddings (100-dimensional)
glove = gensim_api.load("glove-twitter-100")

# Initialize the Flask app
app = Flask(__name__)

def vectorized_tweet(text):
    # Text cleaning and tokenization
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)  # Join tokens into a single string

def get_embedding(text, embedding_size=100):
    words = text.split()
    word_embeddings = [glove[word] for word in words if word in glove]
    if not word_embeddings:
        return np.zeros(embedding_size)
    return np.mean(word_embeddings, axis=0)

def extract_features(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    
    # Ensure cleaned_text is a single string
    cleaned_text = vectorized_tweet(text)
    
    # Basic text features
    features = {
        'text_length': len(text),
        'num_words': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]),
        'hashtag_count': text.count('#'),
        'mention_count': text.count('@'),
        'url_count': text.count('http'),
        'numeric_count': sum(1 for word in text.split() if word.isdigit()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'punctuation_count': sum(1 for char in text if char in '.,;!?')
    }

    # Sentiment features using VADER and TextBlob
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(cleaned_text)  # cleaned_text is a single string here
    features.update({
        'vader_negative': vader_scores['neg'],
        'vader_neutral': vader_scores['neu'],
        'vader_positive': vader_scores['pos'],
        'vader_compound': vader_scores['compound'],
        'textblob_polarity': TextBlob(cleaned_text).sentiment.polarity,
        'textblob_subjectivity': TextBlob(cleaned_text).sentiment.subjectivity
    })
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Embedding features
    embedding_features = pd.DataFrame([get_embedding(cleaned_text)], columns=[f'embed_{i}' for i in range(100)])
    
    # Combine all features
    combined_features = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True), embedding_features.reset_index(drop=True)], axis=1)
    
    # Scale combined features
    scaled_features = scaler.transform(combined_features)
    return scaled_features

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle user input
@app.route('/predict', methods=['POST'])
def predict():
    tweet_text = request.form['tweet']

    # Extract features
    features = extract_features(tweet_text)
    # Predict
    prediction = model.predict(features)
    result = "Disaster" if prediction[0] == 1 else "Non-Disaster"

    # Render result in template
    return render_template('index.html', tweet=tweet_text, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
