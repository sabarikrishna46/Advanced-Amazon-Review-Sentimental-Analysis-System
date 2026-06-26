from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============ NLP IMPORTS ============
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import importlib.util
import spacy
import logging
import os
import re

transformers_available = False
transformers_pipeline = None
try:
    if importlib.util.find_spec('torch') is not None:
        from transformers import pipeline as transformers_pipeline
        transformers_available = True
    else:
        transformers_available = False
except Exception:
    transformers_available = False
    transformers_pipeline = None

# ============ VISUALIZATION ============
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import WordCloud
import matplotlib.dates as mdates

# ============ MACHINE LEARNING ============
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import joblib

# ============ DEEP LEARNING ============
tf_available = False
try:
    if importlib.util.find_spec('tensorflow') is not None:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam
        tf_available = True
    else:
        tf = None
        keras = None
        Tokenizer = None
        pad_sequences = None
        Sequential = None
        Embedding = None
        LSTM = None
        Dense = None
        Dropout = None
        Bidirectional = None
        Adam = None
except Exception:
    tf = None
    keras = None
    Tokenizer = None
    pad_sequences = None
    Sequential = None
    Embedding = None
    LSTM = None
    Dense = None
    Dropout = None
    Bidirectional = None
    Adam = None
    tf_available = False

# ============ DATA SCIENCE & STATS ============
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
import json

# Robust NLTK resource handling with local download and fallbacks
NLTK_OK = True
_local_nltk_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(_local_nltk_dir, exist_ok=True)
if _local_nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk_dir)

def _ensure_nltk_resource(res_id):
    try:
        nltk.data.find(res_id)
        return True
    except Exception:
        try:
            nltk.download(res_id.split('/')[-1], download_dir=_local_nltk_dir, quiet=True)
            return True
        except Exception:
            return False

try:
    ok_punkt = _ensure_nltk_resource('tokenizers/punkt')
    ok_stop = _ensure_nltk_resource('corpora/stopwords')
    ok_vader = _ensure_nltk_resource('sentiment/vader_lexicon')
    if not (ok_punkt and ok_stop and ok_vader):
        NLTK_OK = False
        print("Warning: NLTK resources missing and could not be downloaded. Using fallback tokenizers/stopwords.")
except Exception as _e:
    NLTK_OK = False
    print(f"Warning: NLTK resource check failed: {_e}. Using fallbacks.")

# Fallback safe tokenizers and VADER wrapper
def safe_sent_tokenize(text):
    if NLTK_OK:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text.strip())

def safe_word_tokenize(text):
    if NLTK_OK:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return text.split()

def get_vader_scores(text):
    if NLTK_OK:
        try:
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)
        except Exception:
            pass
    # fallback neutral scores
    return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

app = Flask(__name__)

# ============ GLOBAL VARIABLES ============
df = None
sentiment_models = {}
ml_models = {}
vectorizer = None
tfidf_vectorizer = None
lstm_model = None
tokenizer_lstm = None
sia = None
nlp = None

# Try to load spaCy model only when it is available
nlp = None
try:
    if importlib.util.find_spec('spacy') is not None:
        nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ============ DATA LOADING ============
def load_data():
    """Load and process data"""
    try:
        df = pd.read_csv("reviews.csv")
    except FileNotFoundError:
        df = pd.read_csv("reviews_new.csv")
    
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'], errors='coerce')
    df = df.dropna(subset=['ReviewDate', 'ReviewText'])
    
    # Add sentiment features
    df['SentimentScore_TextBlob'] = df['ReviewText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['SentimentScore_VADER'] = df['ReviewText'].apply(lambda x: get_vader_scores(str(x))['compound'])
    df['ReviewLength'] = df['ReviewText'].apply(len)
    df['WordCount'] = df['ReviewText'].apply(lambda x: len(str(x).split()))
    df['SentenceCount'] = df['ReviewText'].apply(lambda x: len(safe_sent_tokenize(str(x))))
    
    # Binary label for ML/DL
    df['SentimentLabel'] = (df['SentimentScore_TextBlob'] >= 0).astype(int)
    df['Sentiment'] = df['SentimentScore_TextBlob'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    
    return df

# ============ PHASE 1: NLP FUNCTIONS ============

def extract_noun_phrases(text):
    """Extract noun phrases using NER"""
    if nlp is None:
        return []
    try:
        doc = nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return noun_chunks[:5]  # Top 5
    except:
        return []

def emotion_detection(text):
    """Detect emotions in text"""
    if not transformers_available or transformers_pipeline is None:
        return {}
    try:
        emotion_pipeline = transformers_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        emotions = emotion_pipeline(str(text)[:512])  # Limit text length
        return {e['label']: float(e['score']) for e in emotions}
    except Exception:
        return {}

def summarize_review(text):
    """Summarize review using transformers"""
    if not transformers_available or transformers_pipeline is None:
        return str(text)[:100] + "..."
    try:
        summarizer = transformers_pipeline("summarization", model="facebook/bart-large-cnn")
        if len(str(text).split()) > 50:
            summary = summarizer(str(text), max_length=50, min_length=20, do_sample=False)
            return summary[0]['summary_text']
        else:
            return str(text)
    except Exception:
        return str(text)[:100] + "..."

# ============ PHASE 2: MACHINE LEARNING TRAINING ============

def train_ml_models(texts, labels):
    """Train multiple ML models"""
    global ml_models, vectorizer, tfidf_vectorizer
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    
    # CountVectorizer for Naive Bayes
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_count = vectorizer.fit_transform(texts)
    
    # Split data
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
    X_train_count, X_test_count, _, _ = train_test_split(X_count, labels, test_size=0.2, random_state=42)
    
    ml_models = {}
    
    # Naive Bayes
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_count, y_train)
    ml_models['Naive Bayes'] = {'model': nb, 'vectorizer': vectorizer, 'accuracy': accuracy_score(y_test, nb.predict(X_test_count))}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    ml_models['Logistic Regression'] = {'model': lr, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, lr.predict(X_test_tfidf))}
    
    # SVM
    svm = LinearSVC(max_iter=2000, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    ml_models['SVM'] = {'model': svm, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, svm.predict(X_test_tfidf))}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)
    ml_models['Random Forest'] = {'model': rf, 'vectorizer': tfidf_vectorizer, 'accuracy': accuracy_score(y_test, rf.predict(X_test_tfidf))}
    
    return ml_models

# ============ PHASE 3: DEEP LEARNING TRAINING ============

def train_lstm_model(texts, labels, max_words=5000, max_len=100, epochs=10):
    """Train LSTM model for sentiment classification"""
    global lstm_model, tokenizer_lstm
    if not tf_available:
        return {'error': 'TensorFlow not available. Install tensorflow to enable deep learning.'}
    
    tokenizer_lstm = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer_lstm.fit_on_texts(texts)
    
    X = tokenizer_lstm.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=max_len, padding='post')
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Build LSTM model
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
    
    # Train
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                   validation_data=(X_test, y_test), verbose=0)
    
    test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
    return {'accuracy': float(test_accuracy), 'loss': float(test_loss)}

# ============ PHASE 4: DATA SCIENCE ANALYSIS ============

def analyze_trends(df, product_id):
    """Trend analysis and statistical insights"""
    product_df = df[df['ProductID'] == product_id]
    
    analysis = {
        'total_reviews': len(product_df),
        'avg_review_length': float(product_df['ReviewLength'].mean()),
        'avg_word_count': float(product_df['WordCount'].mean()),
        'sentiment_variance': float(product_df['SentimentScore_TextBlob'].var()),
        'positive_ratio': float((product_df['SentimentLabel'] == 1).sum() / len(product_df)),
        'textblob_vs_vader_correlation': float(product_df['SentimentScore_TextBlob'].corr(product_df['SentimentScore_VADER'])),
        'review_frequency': int(len(product_df) / ((product_df['ReviewDate'].max() - product_df['ReviewDate'].min()).days + 1) * 30)  # Per month
    }
    
    return analysis

def detect_fake_reviews(df, product_id, threshold=2.0):
    """Detect potentially fake reviews using statistical anomalies"""
    product_df = df[df['ProductID'] == product_id]
    
    # Calculate z-score for review length and word count
    product_df = product_df.copy()
    product_df['length_zscore'] = np.abs(stats.zscore(product_df['ReviewLength']))
    product_df['wordcount_zscore'] = np.abs(stats.zscore(product_df['WordCount']))
    
    # Flag reviews with extreme values
    fake_reviews = product_df[(product_df['length_zscore'] > threshold) | (product_df['wordcount_zscore'] > threshold)]
    
    return fake_reviews

# ============ FLASK ROUTES ============

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, ml_models, lstm_model
    
    # Reload data
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
                # ============ BASIC STATS ============
                sentiment_stats = {
                    'total': len(product_reviews),
                    'positive': (product_reviews['SentimentLabel'] == 1).sum(),
                    'negative': (product_reviews['SentimentLabel'] == 0).sum(),
                    'avg_score': round(product_reviews['SentimentScore_TextBlob'].mean(), 3),
                    'from_date': product_reviews['ReviewDate'].min().strftime('%Y-%m-%d'),
                    'to_date': product_reviews['ReviewDate'].max().strftime('%Y-%m-%d')
                }
                
                # ============ PHASE 4: DATA SCIENCE ANALYSIS ============
                detailed_analysis = analyze_trends(df, product_id)
                
                # ============ PHASE 1: NLP ANALYSIS ============
                top_phrases = extract_noun_phrases(' '.join(product_reviews['ReviewText'].head(10)))
                detailed_analysis['top_phrases'] = top_phrases

                # ============ UNIVARIATE ANALYSIS ============
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

                # ============ BIVARIATE ANALYSIS ============
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

                # ============ MULTIVARIATE ANALYSIS ============
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Monthly trend
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
                
                # Review length distribution
                ax = axes[0, 1]
                ax.hist(product_reviews['ReviewLength'], bins=30, color='green', alpha=0.7, edgecolor='black')
                ax.set_title("Review Length Distribution", fontweight='bold')
                ax.set_xlabel("Characters")
                ax.set_ylabel("Count")
                
                # Word count vs sentiment
                ax = axes[1, 0]
                scatter = ax.scatter(product_reviews['WordCount'], product_reviews['SentimentScore_TextBlob'], 
                                    c=product_reviews['SentimentLabel'], cmap='RdYlGn', alpha=0.6, s=50)
                ax.set_title("Word Count vs Sentiment", fontweight='bold')
                ax.set_xlabel("Word Count")
                ax.set_ylabel("Sentiment Score")
                plt.colorbar(scatter, ax=ax)
                
                # Sentiment pie chart
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

                # ============ WORD CLOUD ============
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

        # ============ PHASE 2: ML TRAINING & COMPARISON ============
        if 'train_ml' in request.form and len(df) > 50:
            ml_models = train_ml_models(df['ReviewText'].values, df['SentimentLabel'].values)
            ml_results = {model: data['accuracy'] for model, data in ml_models.items()}
            
            # Create ML comparison graph
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(ml_results.keys())
            accuracies = list(ml_results.values())
            bars = ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_title("ML Models Accuracy Comparison", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy Score")
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
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

        # ============ PHASE 3: DEEP LEARNING TRAINING ============
        if 'train_dl' in request.form and len(df) > 50:
            dl_results = train_lstm_model(df['ReviewText'].values, df['SentimentLabel'].values, epochs=5)

        # ============ MANUAL PREDICTION ============
        if 'manual_review' in request.form:
            review_text = request.form['manual_review'].strip()
            product_id_manual = request.form.get('product_id_manual', '').strip()
            
            # TextBlob
            textblob_score = TextBlob(review_text).sentiment.polarity
            
            # VADER (safe)
            vader_scores = get_vader_scores(review_text)
            
            # ML predictions
            ml_predictions = {}
            if ml_models and tfidf_vectorizer:
                for model_name, model_data in ml_models.items():
                    if model_name == 'Naive Bayes':
                        X = vectorizer.transform([review_text])
                    else:
                        X = tfidf_vectorizer.transform([review_text])
                    pred = model_data['model'].predict(X)[0]
                    ml_predictions[model_name] = 'Positive' if pred == 1 else 'Negative'
            
            # LSTM prediction
            lstm_prediction = None
            if lstm_model and tokenizer_lstm:
                X_lstm = tokenizer_lstm.texts_to_sequences([review_text])
                X_lstm = pad_sequences(X_lstm, maxlen=100, padding='post')
                lstm_pred = lstm_model.predict(X_lstm, verbose=0)[0][0]
                lstm_prediction = {'score': float(lstm_pred), 'sentiment': 'Positive' if lstm_pred > 0.5 else 'Negative'}
            
            # Emotions
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
            
            # Save to CSV if product ID provided
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
                    # Try to save, if locked use new file
                    try:
                        df_updated.to_csv("reviews.csv", index=False)
                    except PermissionError:
                        df_updated.to_csv("reviews_new.csv", index=False)
                    new_review_added = True
                    df = load_data()
                except Exception as e:
                    print(f"Error saving review: {e}")

    # Convert to dict for template
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
