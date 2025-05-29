# import libraties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# text & NLP  
import re  
import nltk  
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer
import json
  
# feature extraction & modeling  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline  
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils import resample 
  
# classifiers  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.naive_bayes import MultinomialNB 
from xgboost import XGBClassifier  
from scipy.sparse import hstack 
  
# imbalance  
from imblearn.over_sampling import SMOTE  
nltk.download('punkt')  
nltk.download('stopwords')  
nltk.download('wordnet') 
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify

# Load models and preprocessors
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('num_scaler.pkl')
full_cat_dummy_cols = joblib.load('full_cat_dummy_cols.pkl')

# Loading Data
product_reviews = pd.read_csv('product_reviews.csv')
cat_cols= ['brand', 'reviews_doRecommend', 'reviews_rating', 'reviews_month']
num_cols= ['review_length']
# Creating function for pre processing of text
stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer()  

def preprocess(text):  
    text = text.lower()  
    text = re.sub(r'<.*?>',' ', text)                # remove HTML  
    text = re.sub(r'[^a-z ]',' ', text)              # keep letters only  
    text = re.sub(r'[^a-z0-9\s]', '', text)          # remove punctuation and special chars  
    tokens = nltk.word_tokenize(text)  
    tokens = [t for t in tokens if t not in stop_words and len(t)>1]  
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  
    return ' '.join(tokens) 

# Function to calculate adjusted cosine similarity for user-item matrix
def adjusted_cosine_similarity(user_item):  
    user_means = user_item.mean(axis=1)  
    centered = user_item.sub(user_means, axis=0).fillna(0)  
    sim_matrix = cosine_similarity(centered)  
    return pd.DataFrame(sim_matrix, index=user_item.index, columns=user_item.index) 

# Function to predict ratings for a user based on adjusted cosine similarity
def predict_ratings_for_user(user, user_item, user_similarity, k=10):  
    user_means = user_item.mean(axis=1)  
    items = user_item.columns  
    user_ratings = user_item.loc[user]  
    unrated_items = user_ratings[user_ratings.isnull()].index  
    predictions = {}  
    for item in unrated_items:  
        users_with_rating = user_item[item][user_item[item].notnull()].index  
        if len(users_with_rating) == 0:  
            continue  
        sims = user_similarity.loc[user, users_with_rating]  
        top_k_users = sims.abs().sort_values(ascending=False).head(k).index  
        top_k_sims = sims[top_k_users]  
        top_k_ratings = user_item.loc[top_k_users, item] - user_means.loc[top_k_users]  
        if top_k_sims.abs().sum() == 0:  
            continue  
        pred = (top_k_sims * top_k_ratings).sum() / top_k_sims.abs().sum()  
        pred += user_means.loc[user]  
        predictions[item] = pred  
    return predictions 

# Function to predict sentiment
def get_sentiment_scores_with_logreg(product_reviews, best_model, vectorizer, scaler, cat_cols, num_cols, full_cat_dummy_cols):  
    sentiment_dict = {}  
  
    # Preprocess all data first  
    product_reviews = product_reviews.copy()  
    product_reviews['clean_review'] = product_reviews['reviews_text'].apply(preprocess)  
    text_X = vectorizer.transform(product_reviews['clean_review'])  
  
    cat_X = pd.get_dummies(product_reviews[cat_cols])  
    cat_X = cat_X.reindex(columns=full_cat_dummy_cols, fill_value=0)  
  
    num_X = product_reviews[num_cols].copy()  
    num_X_scaled = scaler.transform(num_X)  
  
    grouped = product_reviews.groupby('name')  
    for item, group in grouped:  
        # Key: get positional indices, not label indices!  
        pos_idx = product_reviews.index.get_indexer(group.index)  
  
        text_X_group = text_X[pos_idx]  
        cat_X_group = cat_X.values[pos_idx, :]  
        num_X_group = num_X_scaled[pos_idx, :]  
  
        features = np.hstack([text_X_group.toarray(), cat_X_group, num_X_group])  
  
        preds = best_model.predict(features)  
        sentiment_score = np.mean(preds)  
        sentiment_dict[item] = sentiment_score  
  
    return sentiment_dict
# Function to get final recomendations
def hybrid_recommendations(username, product_reviews, top_n=15, k=10, alpha=0.7):  
    user_item = product_reviews.pivot_table(index='reviews_username', columns='name', values='reviews_rating', aggfunc='mean')  
    if username not in user_item.index:  
        raise ValueError(f"Username '{username}' not found in data.")  
  
    user_sim = adjusted_cosine_similarity(user_item)  
    cf_predictions = predict_ratings_for_user(username, user_item, user_sim, k=k)  
    if not cf_predictions:  
        print("No predictions could be madef or this user (possibly all items already rated).")
        return pd.DataFrame()
    cf_pred_series = pd.Series(cf_predictions)  

    # Use the logistic regression-based sentiment scores   
    sentiment_scores = get_sentiment_scores_with_logreg(product_reviews, best_model, vectorizer, scaler, cat_cols, num_cols, full_cat_dummy_cols)  
    sentiment_series = pd.Series(sentiment_scores)  

    # Candidates: items the user hasn't rated  
    candidates = cf_pred_series.index  
    sentiment_for_candidates = sentiment_series.loc[candidates]  

    # Normalize both series (min-max scaling)  
    cf_norm = (cf_pred_series - cf_pred_series.min()) / (cf_pred_series.max() - cf_pred_series.min() + 1e-9)  
    sent_norm = (sentiment_for_candidates - sentiment_for_candidates.min()) / (sentiment_for_candidates.max() - sentiment_for_candidates.min() + 1e-9)  

    # Combine collaborative filtering and sentiment scores  
    hybrid_score = alpha * cf_norm + (1 - alpha) * sent_norm  

    # Top N recommendations  
    top_n_idx = hybrid_score.sort_values(ascending=False).head(top_n).index  

    # Prepare recommendation DataFrame  
    results = pd.DataFrame({  
        'PredictedRating': cf_pred_series.loc[top_n_idx],  
        'SentimentScore': sentiment_for_candidates.loc[top_n_idx],  
        'HybridScore': hybrid_score.loc[top_n_idx]  
    }).sort_values('HybridScore', ascending=False)  

    return results


app = Flask(__name__)

@app.route('/')
def home():
    return "Sentiment Based Product Recommendation System is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    username = data.get('username')
    if username not in product_reviews['reviews_username'].values:
        return jsonify({'error': f"User '{username}' not found."}), 404
    try:
        recommendations = hybrid_recommendations(username, product_reviews, top_n=15)
        recommendations= (pd.DataFrame(recommendations.index).to_json(orient='records'))
        recommendations=  recommendations.loads(recommendations)
        return (recommendations, 200)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)