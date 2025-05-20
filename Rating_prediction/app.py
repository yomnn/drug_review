
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load your dataset and preprocess it
data = pd.read_csv('C:/Users/LENOVO/Downloads/archive_5/drugsComTrain_raw.csv')  # Update with your dataset path


# Function to calculate sentiment scores using TextBlob
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['rating'], test_size=0.2, random_state=42)

# Adding sample review to the test data
sample_review = "This drug is so amazing!"
X_test = X_test.reset_index(drop=True)
X_test.loc[len(X_test)] = sample_review
y_test = y_test.reset_index(drop=True)
y_test.loc[len(y_test)] = None  # Placeholder value for the sample review

# Calculate sentiment scores for the reviews in the testing set
X_test_sentiment = pd.DataFrame(X_test[:-1].tolist(), columns=['review_text'])
X_test_sentiment['sentiment_polarity'], X_test_sentiment['sentiment_subjectivity'] = zip(*X_test_sentiment['review_text'].apply(calculate_sentiment))

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test_sentiment['review_text'])

# Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(model, 'decision_tree_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    
    # Calculate sentiment scores
    blob = TextBlob(review)
    sentiment_polarity = blob.sentiment.polarity
    
    # Vectorize the review
    review_tfidf = tfidf_vectorizer.transform([review])
    
    # Predict rating
    pred_rating = model.predict(review_tfidf)
    
    # Adjust prediction based on sentiment polarity
    adjusted_rating = pred_rating + (pred_rating * sentiment_polarity)
    adjusted_rating_clipped = np.clip(adjusted_rating, 0, 10)
    
    # Round the adjusted rating
    rounded_rating = np.round(adjusted_rating_clipped)
    
    return render_template('index.html', sentiment=f'Predicted Rating: {rounded_rating[0]}')

if __name__ == '__main__':
    app.run(debug=True)