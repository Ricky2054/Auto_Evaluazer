import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
def preprocess_text(text, tokenizer, vocab, max_length):
    tokens = tokenizer(text)[:max_length]
    encoded = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0)
    attention_mask = (encoded != vocab['<pad>']).type(torch.LongTensor)
    return encoded.to(device), attention_mask.to(device)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, attention_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output[:, -1, :])
        return output

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, model, tokenizer, vocab, max_length=512):
    input_ids1, attention_mask1 = preprocess_text(answer1, tokenizer, vocab, max_length)
    input_ids2, attention_mask2 = preprocess_text(answer2, tokenizer, vocab, max_length)

    with torch.no_grad():
        embedding1 = model(input_ids1, attention_mask1)
        embedding2 = model(input_ids2, attention_mask2)

    similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]  # Extract single value

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

    # Load tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Build vocabulary
    with open(file_path, 'r') as file:
        user_answers = [sample_answer] + file.read().split('\n\n')  # Include sample_answer in the vocabulary

    def yield_tokens(answers):
        for answer in answers:
            yield tokenizer(answer)

    vocab = build_vocab_from_iterator(yield_tokens(user_answers), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])

    # Load or create the LSTM model
    vocab_size = len(vocab)
    embedding_dim = 300  # Adjust as needed
    hidden_dim = 256  # Adjust as needed
    num_layers = 2  # Adjust as needed
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

    num_answers = len(user_answers)
    marks_distribution = [[] for _ in range(11)]  # Initialize a list of lists to store the distribution of marks (0-10)
    similarity_scores = []
    answer_numbers = []

    for i, user_answer in enumerate(user_answers):
        if user_answer.strip():  # Check if the answer is not empty
            similarity_score = evaluate_long_answer(sample_answer, user_answer, model, tokenizer, vocab)
            similarity_scores.append(similarity_score)
            answer_numbers.append(i)
            marks = assign_marks(similarity_score)
            marks_distribution[marks].append(str(i))  # Store the answer number as a string for each mark
            print(f"Marks for user answer {i}: {marks}")

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
    reference_similarity = evaluate_long_answer(sample_answer, sample_answer, model, tokenizer, vocab)
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
    main()
