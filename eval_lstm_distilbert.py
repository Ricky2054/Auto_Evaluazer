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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
def prepare_vocab(data_paths):
    all_texts = []
    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as file:
            all_texts.extend([line.strip() for line in file.readlines()])
    all_tokens = [token for text in all_texts for token in preprocess_text(text).split()]
    vocab = set(all_tokens)
    word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = len(word_to_idx)
    return word_to_idx

data_paths = ['dataset.txt', 'student_answer_low.txt']
word_to_idx = prepare_vocab(data_paths)

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
train_texts = []
for path in data_paths:
    with open(path, 'r', encoding='utf-8') as file:
        train_texts.extend([line.strip() for line in file.readlines()])
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

def calculate_smape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
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

# Example usage
reference_file_path = 'dataset.txt'
student_file_path = 'student_answer_low 2.txt'

reference_data = parse_file(reference_file_path)
student_data = parse_file(student_file_path)

results = []
for ref, student in zip(reference_data, student_data):
    similarity = calculate_similarity(ref['question'], ref['answer'], student['answer'])
    score = round(similarity * ref['marks'], 2)
    results.append({
        'question': ref['question'],
        'score': score,
        'max_score': ref['marks']
    })

# Calculate metrics
mse = mean_squared_error([result['max_score'] for result in results], [result['score'] for result in results])
rmse = np.sqrt(mse)
mae = mean_absolute_error([result['max_score'] for result in results], [result['score'] for result in results])
smape = calculate_smape(np.array([result['max_score'] for result in results]), np.array([result['score'] for result in results]))

# Print metrics
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"sMAPE: {smape}")
