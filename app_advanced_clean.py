from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import WordCloud
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
import json

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

app = Flask(__name__)

df = None
sentiment_models = {}
ml_models = {}
vectorizer = None
tfidf_vectorizer = None
lstm_model = None
tokenizer_lstm = None
sia = SentimentIntensityAnalyzer()
nlp = None

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")

def load_data():
    try:
        df = pd.read_csv("reviews.csv")
    except FileNotFoundError:
        df = pd.read_csv("reviews_new.csv")
    
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'], errors='coerce')
    df = df.dropna(subset=['ReviewDate', 'ReviewText'])
    
    df['SentimentScore_TextBlob'] = df['ReviewText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['SentimentScore_VADER'] = df['ReviewText'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(str(x))['compound'])
    df['ReviewLength'] = df['ReviewText'].apply(len)
    df['WordCount'] = df['ReviewText'].apply(lambda x: len(str(x).split()))
    df['SentenceCount'] = df['ReviewText'].apply(lambda x: len(sent_tokenize(str(x))))
    
    df['SentimentLabel'] = (df['SentimentScore_TextBlob'] >= 0).astype(int)
    df['Sentiment'] = df['SentimentScore_TextBlob'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    
    return df

def extract_noun_phrases(text):
    if nlp is None:
        return []
    try:
        doc = nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return noun_chunks[:5]
    except:
        return []

def emotion_detection(text):
    try:
        emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        emotions = emotion_pipeline(text[:512])
        return {e['label']: float(e['score']) for e in emotions}
    except:
        return {}

def summarize_review(text):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        if len(text.split()) > 50:
            summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
            return summary[0]['summary_text']
        else:
            return text
    except:
        return text[:100] + "..."

def train_ml_models(texts, labels):
    global ml_models, vectorizer, tfidf_vectorizer
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_count = vectorizer.fit_transform(texts)
    
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
    X_train_count, X_test_count, _, _ = train_test_split(X_count, labels, test_size=0.2, random_state=42)
    
    ml_models = {}
    
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_count, y_train)
    ml_models['Naive Bayes'] = {'model': nb, 'vectorizer': vectorizer, 'accuracy': accuracy_score(y_test, nb.predict(X_test_count))}
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    ml_models['Logistic Regression'] = {'model': lr, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, lr.predict(X_test_tfidf))}
    
    svm = LinearSVC(max_iter=2000, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    ml_models['SVM'] = {'model': svm, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, svm.predict(X_test_tfidf))}
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)
    ml_models['Random Forest'] = {'model': rf, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, rf.predict(X_test_tfidf))}
    
    return ml_models

def train_lstm_model(texts, labels, max_words=5000, max_len=100, epochs=10):
    global lstm_model, tokenizer_lstm
    
    tokenizer_lstm = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer_lstm.fit_on_texts(texts)
    
    X = tokenizer_lstm.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=max_len, padding='post')
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    lstm_model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                   validation_data=(X_test, y_test), verbose=0)
    
    test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
    return {'accuracy': float(test_accuracy), 'loss': float(test_loss)}

def analyze_trends(df, product_id):
    product_df = df[df['ProductID'] == product_id]
    
    analysis = {
        'total_reviews': len(product_df),
        'avg_review_length': float(product_df['ReviewLength'].mean()),
        'avg_word_count': float(product_df['WordCount'].mean()),
        'sentiment_variance': float(product_df['SentimentScore_TextBlob'].var()),
        'positive_ratio': float((product_df['SentimentLabel'] == 1).sum() / len(product_df)),
        'textblob_vs_vader_correlation': float(product_df['SentimentScore_TextBlob'].corr(product_df['SentimentScore_VADER'])),
        'review_frequency': int(len(product_df) / ((product_df['ReviewDate'].max() - product_df['ReviewDate'].min()).days + 1) * 30)
    }
    
    return analysis

def detect_fake_reviews(df, product_id, threshold=2.0):
    product_df = df[df['ProductID'] == product_id]
    
    product_df = product_df.copy()
    product_df['length_zscore'] = np.abs(stats.zscore(product_df['ReviewLength']))
    product_df['wordcount_zscore'] = np.abs(stats.zscore(product_df['WordCount']))
    
    fake_reviews = product_df[(product_df['length_zscore'] > threshold) | (product_df['wordcount_zscore'] > threshold)]
    
    return fake_reviews

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, ml_models, lstm_model
    
    df = load_data()
    
    product_reviews = df.iloc[0:0]
    sentiment_stats = {}
    graphs = {
        'univariate': None,
        'bivariate': None,
        'multivariate': None,
        'wordcloud': None,
        'ml_comparison': None,
        'topic_modeling': None,
        'anomaly': None
    }
    detailed_analysis = {}
    ml_results = {}
    dl_results = {}
    manual_predictions = {}
    new_review_added = False
    product_id = ""

    if request.method == 'POST':
        if 'product_id' in request.form:
            product_id = request.form['product_id'].strip()
            product_reviews = df[df['ProductID'] == product_id]

            if not product_reviews.empty:
                sentiment_stats = {
                    'total': len(product_reviews),
                    'positive': (product_reviews['SentimentLabel'] == 1).sum(),
                    'negative': (product_reviews['SentimentLabel'] == 0).sum(),
                    'avg_score': round(product_reviews['SentimentScore_TextBlob'].mean(), 3),
                    'from_date': product_reviews['ReviewDate'].min().strftime('%Y-%m-%d'),
                    'to_date': product_reviews['ReviewDate'].max().strftime('%Y-%m-%d')
                }
                
                detailed_analysis = analyze_trends(df, product_id)
                
                top_phrases = extract_noun_phrases(' '.join(product_reviews['ReviewText'].head(10)))
                detailed_analysis['top_phrases'] = top_phrases

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(product_reviews['SentimentScore_TextBlob'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='TextBlob')
                ax.hist(product_reviews['SentimentScore_VADER'], bins=20, color='coral', edgecolor='black', alpha=0.5, label='VADER')
                ax.set_title("Sentiment Score Distribution (Univariate)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Sentiment Score")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                graphs['univariate'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                fig, ax = plt.subplots(figsize=(12, 6))
                product_by_date = product_reviews.sort_values('ReviewDate')
                ax.scatter(product_by_date['ReviewDate'], product_by_date['SentimentScore_TextBlob'], 
                          alpha=0.6, c=product_by_date['SentimentLabel'], cmap='RdYlGn', s=50)
                ax.set_title("Sentiment Score Over Time (Bivariate)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Review Date")
                ax.set_ylabel("Sentiment Score")
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label('Sentiment Label')
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                graphs['bivariate'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                product_reviews['Month'] = product_reviews['ReviewDate'].dt.to_period('M')
                monthly_data = product_reviews.groupby('Month').agg({
                    'SentimentScore_TextBlob': 'mean',
                    'ReviewLength': 'mean',
                    'WordCount': 'count'
                }).reset_index()
                
                ax = axes[0, 0]
                ax.plot(range(len(monthly_data)), monthly_data['SentimentScore_TextBlob'], marker='o', label='Avg Sentiment', color='blue')
                ax.set_title("Monthly Sentiment Trend", fontweight='bold')
                ax.set_ylabel("Sentiment Score")
                ax.grid(alpha=0.3)
                
                ax = axes[0, 1]
                ax.hist(product_reviews['ReviewLength'], bins=30, color='green', alpha=0.7, edgecolor='black')
                ax.set_title("Review Length Distribution", fontweight='bold')
                ax.set_xlabel("Characters")
                ax.set_ylabel("Count")
                
                ax = axes[1, 0]
                scatter = ax.scatter(product_reviews['WordCount'], product_reviews['SentimentScore_TextBlob'], 
                                    c=product_reviews['SentimentLabel'], cmap='RdYlGn', alpha=0.6, s=50)
                ax.set_title("Word Count vs Sentiment", fontweight='bold')
                ax.set_xlabel("Word Count")
                ax.set_ylabel("Sentiment Score")
                plt.colorbar(scatter, ax=ax)
                
                ax = axes[1, 1]
                sentiment_counts = product_reviews['SentimentLabel'].value_counts()
                ax.pie([sentiment_counts.get(1, 0), sentiment_counts.get(0, 0)], 
                       labels=['Positive', 'Negative'], autopct='%1.1f%%',
                       colors=['#90EE90', '#FF6B6B'], startangle=90)
                ax.set_title("Sentiment Distribution", fontweight='bold')
                
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                graphs['multivariate'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                text = ' '.join(product_reviews['ReviewText'].dropna())
                if len(text.split()) > 10:
                    wc = WordCloud(width=1000, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    graphs['wordcloud'] = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close()

        if 'train_ml' in request.form and len(df) > 50:
            ml_models = train_ml_models(df['ReviewText'].values, df['SentimentLabel'].values)
            ml_results = {model: data['accuracy'] for model, data in ml_models.items()}
            
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(ml_results.keys())
            accuracies = list(ml_results.values())
            bars = ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_title("ML Models Accuracy Comparison", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy Score")
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            graphs['ml_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

        if 'train_dl' in request.form and len(df) > 50:
            dl_results = train_lstm_model(df['ReviewText'].values, df['SentimentLabel'].values, epochs=5)

        if 'manual_review' in request.form:
            review_text = request.form['manual_review'].strip()
            product_id_manual = request.form.get('product_id_manual', '').strip()
            
            textblob_score = TextBlob(review_text).sentiment.polarity
            
            vader_scores = SentimentIntensityAnalyzer().polarity_scores(review_text)
            
            ml_predictions = {}
            if ml_models and tfidf_vectorizer:
                for model_name, model_data in ml_models.items():
                    if model_name == 'Naive Bayes':
                        X = vectorizer.transform([review_text])
                    else:
                        X = tfidf_vectorizer.transform([review_text])
                    pred = model_data['model'].predict(X)[0]
                    ml_predictions[model_name] = 'Positive' if pred == 1 else 'Negative'
            
            lstm_prediction = None
            if lstm_model and tokenizer_lstm:
                X_lstm = tokenizer_lstm.texts_to_sequences([review_text])
                X_lstm = pad_sequences(X_lstm, maxlen=100, padding='post')
                lstm_pred = lstm_model.predict(X_lstm, verbose=0)[0][0]
                lstm_prediction = {'score': float(lstm_pred), 'sentiment': 'Positive' if lstm_pred > 0.5 else 'Negative'}
            
            emotions = emotion_detection(review_text)
            
            manual_predictions = {
                'review': review_text,
                'textblob_score': float(textblob_score),
                'vader_score': float(vader_scores['compound']),
                'vader_details': {k: float(v) for k, v in vader_scores.items()},
                'ml_predictions': ml_predictions,
                'lstm_prediction': lstm_prediction,
                'emotions': emotions,
                'summary': summarize_review(review_text)
            }
            
            if product_id_manual and review_text:
                review_date = datetime.now().strftime('%Y-%m-%d')
                try:
                    df_existing = pd.read_csv("reviews.csv" if pd.io.common.file_exists("reviews.csv") else "reviews_new.csv")
                    new_row = pd.DataFrame({
                        'ProductID': [product_id_manual],
                        'ReviewText': [review_text],
                        'ReviewDate': [review_date]
                    })
                    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                    try:
                        df_updated.to_csv("reviews.csv", index=False)
                    except PermissionError:
                        df_updated.to_csv("reviews_new.csv", index=False)
                    new_review_added = True
                    df = load_data()
                except Exception as e:
                    print(f"Error saving review: {e}")

    reviews_list = product_reviews.to_dict(orient='records')
    for review in reviews_list:
        review['ReviewDate'] = review['ReviewDate'].strftime('%Y-%m-%d')
    
    return render_template('index_advanced.html',
                           product_id=product_id,
                           reviews=reviews_list,
                           stats=sentiment_stats,
                           graphs=graphs,
                           detailed_analysis=detailed_analysis,
                           ml_results=ml_results,
                           dl_results=dl_results,
                           manual_predictions=manual_predictions,
                           new_review_added=new_review_added,
                           models_available=len(ml_models) > 0,
                           lstm_available=lstm_model is not None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
