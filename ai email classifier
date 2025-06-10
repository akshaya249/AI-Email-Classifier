# app.py - Complete AI Email Classifier Application

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Global variables for models
vectorizer = None
spam_model = None
sentiment_model = None
stop_words = set(stopwords.words('english'))

# Statistics tracking
stats = {
    'total_predictions': 0,
    'spam_count': 0,
    'ham_count': 0,
    'positive_sentiment': 0,
    'negative_sentiment': 0,
    'neutral_sentiment': 0,
    'high_priority': 0,
    'medium_priority': 0,
    'low_priority': 0
}

def preprocess_text(text):
    """Advanced text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_text)

def create_comprehensive_dataset():
    """Create a comprehensive dataset for training"""
    
    # Extended spam examples
    spam_emails = [
        "CONGRATULATIONS! You have won $1,000,000! Click here to claim your prize NOW! Limited time offer!",
        "URGENT: Your account will be suspended! Verify your identity immediately by clicking this link!",
        "Make $5000 per week working from home! No experience required! Start today!",
        "FREE MONEY! Get rich quick! No strings attached! Click here now!",
        "WINNER! You've been selected for a special offer! Claim your free iPhone now!",
        "ALERT: Suspicious activity detected! Update your password immediately!",
        "Get paid $500 daily! Work from home! No investment required! Join now!",
        "FINAL NOTICE: Your subscription expires today! Renew now to avoid charges!",
        "EXCLUSIVE DEAL: 90% off everything! Limited time! Buy now!",
        "URGENT: IRS notice! You owe $5000! Pay immediately to avoid arrest!",
        "FREE VIAGRA! No prescription needed! Order now! Discreet delivery!",
        "LOTTERY WINNER! You've won $50,000! Send us your bank details!",
        "BANKRUPTCY HELP! Eliminate all debt! Call now! Free consultation!",
        "WEIGHT LOSS MIRACLE! Lose 30 pounds in 30 days! Guaranteed results!",
        "DATING ALERT! Hot singles in your area! Meet them tonight!",
        "CREDIT REPAIR! Bad credit? No problem! We can help! Call now!",
        "CASH ADVANCE! Get $1000 today! No credit check! Fast approval!",
        "WORK FROM HOME! Earn $200/hour! No experience! Start immediately!",
        "FORECLOSURE HELP! Save your home! Call now! Free consultation!",
        "PENNY STOCKS! Get rich quick! Hot tips inside! Buy now!"
    ]
    
    # Extended ham (legitimate) examples
    ham_emails = [
        "Hi John, I hope you're doing well. Could we schedule a meeting for next week to discuss the project timeline?",
        "Thank you for your email. I'll review the documents and get back to you by Friday.",
        "The quarterly report has been completed and is ready for your review. Please let me know if you need any changes.",
        "Reminder: Team meeting tomorrow at 2 PM in Conference Room A. Please bring your project updates.",
        "I've attached the updated contract for your review. Please let me know if you have any questions.",
        "Great job on the presentation yesterday! The client was very impressed with your work.",
        "Could you please send me the latest version of the budget spreadsheet? I need it for the board meeting.",
        "The server maintenance is scheduled for this weekend. Please backup your important files.",
        "I'll be out of office next week. Please contact Sarah for any urgent matters.",
        "Congratulations on your promotion! Well deserved. Let's celebrate with lunch this Friday.",
        "The project deadline has been extended to next month. This gives us more time to perfect the deliverables.",
        "Please find attached the meeting minutes from yesterday's discussion. Let me know if I missed anything.",
        "I've reviewed your proposal and have a few suggestions. Can we discuss them over a call?",
        "The training session on Monday was very informative. Thank you for organizing it.",
        "Please submit your expense reports by the end of this week. HR needs them for processing.",
        "The new policy documents are available on the company portal. Please review them at your convenience.",
        "I'm working on the annual budget and need your department's input. Could you send it by Thursday?",
        "The client has approved the design mockups. We can proceed to the development phase.",
        "Thank you for the quick turnaround on the bug fixes. The application is working perfectly now.",
        "I've scheduled a code review session for next Tuesday. Please prepare your modules for review."
    ]
    
    # Create DataFrame
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1 = spam, 0 = ham
    
    df = pd.DataFrame({
        'text': emails,
        'label': labels
    })
    
    # Add more features
    df['length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['question_count'] = df['text'].apply(lambda x: x.count('?'))
    df['capital_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    
    return df

def train_models():
    """Train comprehensive ML models"""
    global vectorizer, spam_model, sentiment_model
    
    print("Creating comprehensive dataset...")
    df = create_comprehensive_dataset()
    
    # Preprocess texts
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    X_text = vectorizer.fit_transform(df['processed_text'])
    
    # Additional features
    X_features = df[['length', 'word_count', 'exclamation_count', 'question_count', 'capital_ratio']].values
    
    # Combine text and numerical features
    from scipy.sparse import hstack
    X = hstack([X_text, X_features])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train spam classifier
    spam_model = LogisticRegression(random_state=42)
    spam_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = spam_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Spam Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/spam_model.pkl', 'wb') as f:
        pickle.dump(spam_model, f)
    
    # Save model metrics
    metrics = {
        'accuracy': accuracy,
        'training_date': datetime.now().isoformat(),
        'model_type': 'Logistic Regression',
        'features': 2000,
        'training_samples': len(df)
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Models trained and saved successfully!")
    return accuracy

def load_models():
    """Load trained models"""
    global vectorizer, spam_model
    
    try:
        if os.path.exists('models/vectorizer.pkl') and os.path.exists('models/spam_model.pkl'):
            with open('models/vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('models/spam_model.pkl', 'rb') as f:
                spam_model = pickle.load(f)
            print("Models loaded successfully!")
            return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    
    return False

def analyze_sentiment(text):
    """Advanced sentiment analysis"""
    # Positive and negative word lists
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
        'perfect', 'brilliant', 'outstanding', 'superb', 'magnificent', 'marvelous',
        'terrific', 'fabulous', 'incredible', 'remarkable', 'exceptional', 'impressive',
        'love', 'like', 'enjoy', 'appreciate', 'pleased', 'happy', 'delighted',
        'satisfied', 'glad', 'thrilled', 'excited', 'grateful', 'thankful'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
        'poor', 'disappointing', 'unsatisfactory', 'unacceptable', 'disgusting',
        'pathetic', 'useless', 'worthless', 'annoying', 'frustrating', 'irritating',
        'angry', 'upset', 'sad', 'unhappy', 'disappointed', 'concerned', 'worried',
        'problem', 'issue', 'error', 'mistake', 'wrong', 'failed', 'broken'
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment
    if positive_score > negative_score:
        sentiment = 'Positive'
        confidence = min(90, 60 + (positive_score - negative_score) * 10)
    elif negative_score > positive_score:
        sentiment = 'Negative'
        confidence = min(90, 60 + (negative_score - positive_score) * 10)
    else:
        sentiment = 'Neutral'
        confidence = 70
    
    return sentiment, confidence

def analyze_priority(text):
    """Analyze email priority"""
    high_priority_words = [
        'urgent', 'asap', 'immediate', 'emergency', 'critical', 'important',
        'deadline', 'rush', 'priority', 'quickly', 'soon', 'today',
        'now', 'immediately', 'fast', 'quick', 'hurry', 'time-sensitive'
    ]
    
    medium_priority_words = [
        'meeting', 'schedule', 'discuss', 'review', 'feedback', 'update',
        'follow-up', 'reminder', 'please', 'could', 'would', 'request'
    ]
    
    text_lower = text.lower()
    
    high_count = sum(1 for word in high_priority_words if word in text_lower)
    medium_count = sum(1 for word in medium_priority_words if word in text_lower)
    
    if high_count > 0:
        return 'High', min(95, 70 + high_count * 10)
    elif medium_count > 0:
        return 'Medium', min(85, 60 + medium_count * 5)
    else:
        return 'Low', 60

def extract_features(text):
    """Extract comprehensive email features"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'digit_count': sum(1 for c in text if c.isdigit()),
        'special_char_count': sum(1 for c in text if not c.isalnum() and not c.isspace()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }
    
    return features

def create_visualization(text, prediction_data):
    """Create visualization charts"""
    try:
        # Create word cloud
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
        
        # Save word cloud to base64
        img_buffer = BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    except:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form.get('email_text', '').strip()
        
        if not email_text:
            return jsonify({'error': 'Please enter email text'})
        
        # Update statistics
        stats['total_predictions'] += 1
        
        # Preprocess text
        processed_text = preprocess_text(email_text)
        
        # Extract features
        email_features = extract_features(email_text)
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Additional features
        additional_features = np.array([[
            email_features['length'],
            email_features['word_count'],
            email_features['exclamation_count'],
            email_features['question_count'],
            email_features['capital_ratio']
        ]])
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([text_vector, additional_features])
        
        # Make spam prediction
        spam_prediction = spam_model.predict(X)[0]
        spam_probability = spam_model.predict_proba(X)[0]
        spam_confidence = max(spam_probability) * 100
        
        # Analyze sentiment
        sentiment, sentiment_confidence = analyze_sentiment(email_text)
        
        # Analyze priority
        priority, priority_confidence = analyze_priority(email_text)
        
        # Update statistics
        if spam_prediction == 1:
            stats['spam_count'] += 1
            classification = "Spam"
        else:
            stats['ham_count'] += 1
            classification = "Ham (Legitimate)"
        
        # Update sentiment stats
        if sentiment == 'Positive':
            stats['positive_sentiment'] += 1
        elif sentiment == 'Negative':
            stats['negative_sentiment'] += 1
        else:
            stats['neutral_sentiment'] += 1
        
        # Update priority stats
        if priority == 'High':
            stats['high_priority'] += 1
        elif priority == 'Medium':
            stats['medium_priority'] += 1
        else:
            stats['low_priority'] += 1
        
        # Risk assessment
        risk_factors = []
        if spam_prediction == 1:
            risk_factors.append("Classified as spam")
        if email_features['exclamation_count'] > 3:
            risk_factors.append("Excessive exclamation marks")
        if email_features['capital_ratio'] > 0.3:
            risk_factors.append("High ratio of capital letters")
        if any(word in email_text.lower() for word in ['click here', 'act now', 'limited time']):
            risk_factors.append("Contains urgency phrases")
        
        # Create word cloud
        wordcloud_base64 = create_visualization(email_text, {})
        
        response_data = {
            'classification': classification,
            'spam_probability': round(spam_confidence, 2),
            'sentiment': sentiment,
            'sentiment_confidence': round(sentiment_confidence, 2),
            'priority': priority,
            'priority_confidence': round(priority_confidence, 2),
            'features': email_features,
            'risk_factors': risk_factors,
            'wordcloud': wordcloud_base64,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', stats=stats)

@app.route('/api/stats')
def get_stats():
    return jsonify(stats)

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    try:
        emails = request.json.get('emails', [])
        results = []
        
        for email_text in emails:
            # Simplified batch processing
            processed_text = preprocess_text(email_text)
            text_vector = vectorizer.transform([processed_text])
            
            # Extract basic features
            features = extract_features(email_text)
            additional_features = np.array([[
                features['length'],
                features['word_count'],
                features['exclamation_count'],
                features['question_count'],
                features['capital_ratio']
            ]])
            
            from scipy.sparse import hstack
            X = hstack([text_vector, additional_features])
            
            prediction = spam_model.predict(X)[0]
            probability = spam_model.predict_proba(X)[0]
            
            results.append({
                'email': email_text[:100] + '...' if len(email_text) > 100 else email_text,
                'classification': 'Spam' if prediction == 1 else 'Ham',
                'confidence': round(max(probability) * 100, 2)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting AI Email Classifier...")
    print("📊 Initializing machine learning models...")
    
    # Load or train models
    if not load_models():
        print("📚 No existing models found. Training new models...")
        accuracy = train_models()
        print(f"✅ Model training completed with {accuracy:.2%} accuracy!")
    else:
        print("✅ Models loaded successfully!")
    
    print("\n🌐 Web application starting...")
    print("📧 AI Email Classifier Ready!")
    print("🔗 Open your browser and navigate to: http://127.0.0.1:5000")
    print("🎯 Features: Spam Detection | Sentiment Analysis | Priority Classification")
    print("-" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
