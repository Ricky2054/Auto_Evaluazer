from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    evaluations = db.relationship('Evaluation', backref='user', lazy=True)

class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    total_marks = db.Column(db.Float, nullable=False)
    obtained_marks = db.Column(db.Float, nullable=False)
    student_name = db.Column(db.String(100), nullable=True)
    student_id = db.Column(db.String(50), nullable=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    evaluations = user.evaluations
    # Clear any existing flash messages
    session.pop('_flashes', None)
    return render_template('dashboard.html', user=user, evaluations=evaluations)

@app.route('/delete_evaluation/<int:evaluation_id>', methods=['POST'])
def delete_evaluation(evaluation_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    evaluation = Evaluation.query.get_or_404(evaluation_id)
    
    if evaluation.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Permission denied'}), 403
    
    db.session.delete(evaluation)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Evaluation deleted successfully'})

# Set up stop words
stop_words = set(stopwords.words('english'))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Load DistilBERT model and tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

def tokenize_and_pad(text, word_to_idx, max_length=512):
    tokens = preprocess_text(text).split()
    sequence = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    if len(sequence) < max_length:
        sequence = sequence + [word_to_idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return torch.tensor(sequence, dtype=torch.long)

# Prepare vocabulary
all_texts = [line.strip() for line in open('dataset.txt')] + [line.strip() for line in open('student_answer_low.txt')]
all_tokens = [token for text in all_texts for token in preprocess_text(text).split()]
vocab = set(all_tokens)
word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = len(word_to_idx)

# Prepare LSTM model
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
vocab_size = len(word_to_idx)
lstm_model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM)

def train_model(model, texts, word_to_idx, epochs=5, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for text in texts:
            sequence = tokenize_and_pad(text, word_to_idx).unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(sequence)
            loss = criterion(outputs, outputs)  # Dummy target for autoencoding
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Train LSTM model
train_texts = [line.strip() for line in open('dataset.txt')] + [line.strip() for line in open('student_answer_low.txt')]
train_model(lstm_model, train_texts, word_to_idx)

def get_lstm_embedding(text, model):
    model.eval()
    with torch.no_grad():
        tokenized = tokenize_and_pad(text, word_to_idx).unsqueeze(0)
        embedding = model(tokenized)
    return embedding.squeeze().numpy()

def get_distilbert_embedding(text, model):
    model.eval()
    with torch.no_grad():
        inputs = distilbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_semantic_similarity(text1, text2):
    embedding1 = get_distilbert_embedding(text1, distilbert_model)
    embedding2 = get_distilbert_embedding(text2, distilbert_model)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def is_answer_irrelevant(question, answer, ref_answer):
    question_similarity = fuzz.ratio(question.lower(), answer.lower())
    if question_similarity > 80:
        return True
    ref_similarity = get_semantic_similarity(ref_answer, answer)
    if ref_similarity < 0.1:
        return True
    return False

def calculate_similarity(question, ref_answer, student_answer):
    if is_answer_irrelevant(question, student_answer, ref_answer):
        return 0
    
    ref_processed = preprocess_text(ref_answer)
    student_processed = preprocess_text(student_answer)
    
    ref_lstm = get_lstm_embedding(ref_processed, lstm_model)
    student_lstm = get_lstm_embedding(student_processed, lstm_model)
    lstm_similarity = cosine_similarity([ref_lstm], [student_lstm])[0][0]
    
    ref_bert = get_distilbert_embedding(ref_processed, distilbert_model)
    student_bert = get_distilbert_embedding(student_processed, distilbert_model)
    bert_similarity = cosine_similarity([ref_bert], [student_bert])[0][0]
    
    tfidf_matrix = tfidf_vectorizer.fit_transform([ref_processed, student_processed])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    ref_words = set(ref_processed.split())
    student_words = set(student_processed.split())
    word_overlap = len(ref_words.intersection(student_words)) / len(ref_words.union(student_words))
    
    combined_similarity = (
        0.4 * lstm_similarity +
        0.4 * bert_similarity +
        0.1 * tfidf_similarity +
        0.1 * word_overlap
    )
    
    return combined_similarity

def parse_file(file):
    content = file.read().decode('utf-8')
    questions = re.split(r'Question:', content)[1:]
    parsed_data = []
    
    for q in questions:
        parts = q.strip().split('Answer:', 1)
        question = parts[0].strip()
        answer_marks = parts[1].rsplit('Marks:', 1)
        answer = answer_marks[0].strip()
        marks = int(answer_marks[1].strip()) if len(answer_marks) > 1 else 0
        parsed_data.append({'question': question, 'answer': answer, 'marks': marks})
    
    return parsed_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        reference_file = request.files['reference']
        student_file = request.files['student']
        student_name = request.form['student_name']
        student_id = request.form['student_id']
        
        reference_data = parse_file(reference_file)
        student_data = parse_file(student_file)
        
        total_marks = sum(q['marks'] for q in reference_data)
        total_obtained = 0
        results = []
        
        for ref, student in zip(reference_data, student_data):
            similarity = calculate_similarity(ref['question'], ref['answer'], student['answer'])
            score = round(similarity * ref['marks'], 2)
            total_obtained += score
            results.append({
                'question': ref['question'],
                'score': score,
                'max_score': ref['marks']
            })
        
        # Save evaluation to database
        user = User.query.get(session['user_id'])
        new_evaluation = Evaluation(
            user=user, 
            total_marks=total_marks, 
            obtained_marks=round(total_obtained, 2),
            student_name=student_name,
            student_id=student_id
        )
        db.session.add(new_evaluation)
        db.session.commit()

        chart_data = {
            'labels': [f"Q{i+1}" for i in range(len(results))],
            'obtained': [result['score'] for result in results],
            'maximum': [result['max_score'] for result in results]
        }

        return render_template('result.html', 
                               total_marks=total_marks, 
                               total_obtained=round(total_obtained, 2),
                               results=results,
                               chart_data=chart_data,
                               student_name=student_name,
                               student_id=student_id)
    
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)