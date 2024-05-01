import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        last_output = output[:, -1, :]  
        output = self.fc(last_output)
        return output

# Load pre-trained sentiment model
sentiment_model = LSTMModel(input_size=10000, hidden_size=256, num_layers=2, output_size=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

# Preprocessing
def preprocess_lstm(text, vocab):
    tokens = [vocab.get(word, vocab['<unk>']) for word in text.split()]
    tokens = tokens[:512]  # LSTM max input length
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    return tokens_tensor.unsqueeze(0)  # Add batch dimension

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, model, max_seq_length=512):
    tokens_tensor1 = preprocess_lstm(answer1, vocab)
    tokens_tensor2 = preprocess_lstm(answer2, vocab)
    max_length = max(tokens_tensor1.size(1), tokens_tensor2.size(1))
    padding_length = max_seq_length - max_length if max_length < max_seq_length else 0

    # Pad the tensors
    tokens_tensor1_padded = torch.nn.functional.pad(tokens_tensor1, (0, padding_length), 'constant', 0)
    tokens_tensor2_padded = torch.nn.functional.pad(tokens_tensor2, (0, padding_length), 'constant', 0)

    with torch.no_grad():
        output1 = model(tokens_tensor1_padded)
        last_hidden_states1 = output1[:, -1]  # Extract the last hidden state
        output2 = model(tokens_tensor2_padded)
        last_hidden_states2 = output2[:, -1]  # Extract the last hidden state

    embedding1 = last_hidden_states1.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    embedding2 = last_hidden_states2.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

    # Reshape embeddings to 2D arrays if they are scalar values
    embedding1 = np.atleast_2d(embedding1)
    embedding2 = np.atleast_2d(embedding2)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]  # Extract single value

    # Perform sentiment analysis
    sentiment_inputs1 = preprocess_lstm(answer1, vocab)
    sentiment_inputs2 = preprocess_lstm(answer2, vocab)
    sentiment_output1 = sentiment_model(sentiment_inputs1)
    sentiment_output2 = sentiment_model(sentiment_inputs2)
    sentiment1 = torch.softmax(sentiment_output1, dim=1).argmax().item()
    sentiment2 = torch.softmax(sentiment_output2, dim=1).argmax().item()

    # Adjust the similarity score based on sentiment analysis
    if sentiment1 != sentiment2:
        similarity *= 0.8  # Reduce the similarity score by 20% if sentiments don't match

    return similarity




# Assign marks based on evaluation score
def assign_marks(similarity_score):
    similarity_score = float(similarity_score)
    if similarity_score <= 0.5:
        return 0
    else:
        return int((similarity_score - 0.5) * 20)  # Scale to 0-10 range

def main():
    sample_question = input("Please enter a sample question: ")
    sample_answer = input("Please enter a sample answer: ")
    file_path = input("Please enter the path to the text file containing user answers: ")

    with open(file_path, 'r') as file:
        user_answers = file.read().split('\n\n')  # Split on double newline to separate answers

    num_answers = len(user_answers)
    marks_distribution = [[] for _ in range(11)]  # Initialize a list of lists to store the distribution of marks (0-10)
    similarity_scores = []
    answer_numbers = []

    for i, user_answer in enumerate(user_answers):
        if user_answer.strip():  # Check if the answer is not empty
            similarity_score = evaluate_long_answer(sample_answer, user_answer, sentiment_model)
            similarity_scores.append(similarity_score)
            answer_numbers.append(i + 1)
            marks = assign_marks(similarity_score)
            marks_distribution[marks].append(str(i + 1))  # Store the answer number as a string for each mark
            print(f"Marks for user answer {i + 1}: {marks}")

    print("\nMarks Distribution:")
    for m, answers in enumerate(marks_distribution):
        print(f"Marks {m}: {', '.join(answers)}")

    # Plot the bar graph
    marks = [i for i in range(11)]  # List of marks from 0 to 10
    counts = [len(answers) for answers in marks_distribution]  # Number of answers for each mark

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the bar graph
    bars = ax1.bar(marks, counts)

    # Add labels to the bars with answer numbers
    for bar, answers in zip(bars, marks_distribution):
        height = bar.get_height()
        if height > 0:
            ax1.annotate(','.join(answers),
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    # Set axis labels and title for the bar graph
    ax1.set_xlabel("Marks")
    ax1.set_ylabel("Number of Answers")
    ax1.set_title("Marks Distribution")

    # Calculate deviation from the reference answer
    reference_similarity = evaluate_long_answer(sample_answer, sample_answer, sentiment_model)
    deviations = [reference_similarity - score for score in similarity_scores]

    # Plot the scatter plot
    ax2.scatter(answer_numbers, deviations)

    # Add labels and annotations for the scatter plot
    ax2.set_xlabel("Answer Number")
    ax2.set_ylabel("Deviation from Reference Answer")
    ax2.set_title("Deviation of User Answers from Reference Answer")

    # Add annotations for each point in the scatter plot
    for i, deviation in enumerate(deviations):
        ax2.annotate(f"{similarity_scores[i]:.4f}", (answer_numbers[i], deviation), textcoords="offset points", xytext=(0, 10), ha='center')

    # Adjust the spacing between subplots to make room for the annotations
    plt.subplots_adjust(hspace=0.5, bottom=0.2)

    plt.show()

if __name__ == "__main__":
    # Create a simple vocabulary
    vocab = {word: idx for idx, word in enumerate(['<pad>', '<unk>'] + list('abcdefghijklmnopqrstuvwxyz '))}
    main()
