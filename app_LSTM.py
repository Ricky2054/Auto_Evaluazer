import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.lstm(embed)
        output = self.fc(h_n[-1])
        return output

# Preprocessing
def preprocess_lstm(text, vocab, device):
    tokens = [vocab.get(token, vocab["<unk>"]) for token in text.split()]
    tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    return tokens_tensor

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, model, vocab, device, max_seq_length=512):
    tokens_tensor1 = preprocess_lstm(answer1, vocab, device)
    tokens_tensor2 = preprocess_lstm(answer2, vocab, device)

    with torch.no_grad():
        embedding1 = model(tokens_tensor1)
        embedding2 = model(tokens_tensor2)

    similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]
    return similarity

# Assign marks based on evaluation score
def assign_marks(similarity_score):
    similarity_score = float(similarity_score)
    if similarity_score >= 0.9:
        return 5
    elif similarity_score >= 0.8:
        return 4
    elif similarity_score >= 0.6:
        return 3
    elif similarity_score >= 0.3:
        return 2
    else:
        return 1

def main():
    # Load the LSTM model and vocabulary
    vocab_size = 10000
    embedding_dim = 300
    hidden_dim = 256
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    vocab = {"<unk>": 0}  # Load your vocabulary here

    while True:
        print("\nEnter 'exit' to quit the program.")
        question = input("Please enter a question: ")
        if question.lower() == "exit":
            break
        sample_answer = input("Please enter a sample answer: ")
        user_answer = input("Please enter the user's answer: ")

        similarity_score = evaluate_long_answer(sample_answer, user_answer, model, vocab, device)
        marks = assign_marks(similarity_score)
        print(f"Similarity Score: {similarity_score}")
        print(f"Marks: {marks}")

if __name__ == "__main__":
    main()