from flask import Flask, render_template, request  
import pandas as pd  
import numpy as np  
import pickle  
import model # Import our model.py file
  
# Load model and preprocessors  
with open('model/best_model.pkl', 'rb') as f:  
    best_model = pickle.load(f)  
with open('model/tfidf_vectorizer.pkl', 'rb') as f:  
    vectorizer = pickle.load(f)  
with open('model/num_scaler.pkl', 'rb') as f:  
    scaler = pickle.load(f)  
with open('model/full_cat_dummy_cols.pkl', 'rb') as f:  
    full_cat_dummy_cols = pickle.load(f)  
  
# You may want to wrap your feature engineering in a helper file (utils.py)  
from utils import get_sentiment_scores_with_logreg  
  
# Load your product/review data  
product_reviews = pd.read_csv("sample30.csv")  # or your data file  
  
app = Flask(__name__)  
  
@app.route('/', methods=['GET'])  
def index():  
    return render_template('index.html')  
  
@app.route('/recommend', methods=['POST'])  
def recommend():  
    user_input = request.form['user_input']
      
  
    # Filter your product_reviews or process as per your logic (here it's a placeholder)  
    if not user_input:
        # Handle case where username is empty
        user_input = model.get_all_usernames()
        return render_template('index.html', usernames=user_input, recommendations=[], error="Please enter a valid username.")

    # Get final recommendations from our model
    # The get_final_recommendations function handles the entire logic
    # of initial recommendations and sentiment-based filtering.
    recommendations = model.hybrid_recommendations(user_input, product_reviews, top_n=20) 

    usernames = model.get_all_usernames() # Pass usernames back for dropdown
    
    if not recommendations:
        return render_template('index.html', usernames=usernames, recommendations=[], error=f"Could not find recommendations for '{username}'. Please try another username.")
  
    # (Optionally) Update clean_review, or update your get_sentiment_scores fn to preprocess  
    cat_cols = ['brand', 'reviews_doRecommend', 'reviews_rating', 'reviews_month'] 
    num_cols = ['review_length']    
    
    recommendations = pd.DataFrame(user_recom.index, columns=['Recommended Items']) 
  
    return render_template('index.html', usernames=usernames, recommendations=recommendations, user_input=user_input)  
  
if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=10000)  