from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocessing
def preprocess_bert(text):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:512]  # BERT max input length
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    return tokens_tensor

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, model, max_seq_length=512):
    tokens_tensor1 = preprocess_bert(answer1)
    tokens_tensor2 = preprocess_bert(answer2)
    max_length = max(tokens_tensor1.size(1), tokens_tensor2.size(1))
    padding_length = max_seq_length - max_length if max_length < max_seq_length else 0
    # Pad the tensors
    tokens_tensor1_padded = torch.nn.functional.pad(tokens_tensor1, (0, padding_length), 'constant', 0)
    tokens_tensor2_padded = torch.nn.functional.pad(tokens_tensor2, (0, padding_length), 'constant', 0)
    # Create attention mask
    attention_mask1 = torch.ones_like(tokens_tensor1)
    attention_mask2 = torch.ones_like(tokens_tensor2)
    attention_mask1_padded = torch.nn.functional.pad(attention_mask1, (0, padding_length), 'constant', 0)
    attention_mask2_padded = torch.nn.functional.pad(attention_mask2, (0, padding_length), 'constant', 0)

    with torch.no_grad():
        output1 = model(tokens_tensor1_padded, attention_mask=attention_mask1_padded, token_type_ids=None)
        last_hidden_states1 = output1.last_hidden_state
        output2 = model(tokens_tensor2_padded, attention_mask=attention_mask2_padded, token_type_ids=None)
        last_hidden_states2 = output2.last_hidden_state

        embedding1 = last_hidden_states1[:, :max_length, :].squeeze(0)  # Remove batch dimension
        embedding2 = last_hidden_states2[:, :max_length, :].squeeze(0)  # Remove batch dimension

        similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]  # Extract single value

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
    sample_question = input("Please enter a sample question: ")
    sample_answer = input("Please enter a sample answer: ")

    num_answers = int(input("How many answers will the user give? "))

    marks_distribution = [0, 0, 0, 0, 0]  # Initialize a list to store the distribution of marks

    for i in range(num_answers):
        user_answer = input(f"Please enter the user's answer {i + 1}: ")
        similarity_score = evaluate_long_answer(sample_answer, user_answer, model)
        marks = assign_marks(similarity_score)
        marks_distribution[marks - 1] += 1  # Update the marks distribution

    print("\nMarks Distribution:")
    for m, count in enumerate(marks_distribution):
        print(f"Marks {m + 1}: {count}")

    # Plot the marks distribution
    marks = [1, 2, 3, 4, 5]
    plt.bar(marks, marks_distribution)
    plt.xlabel("Marks")
    plt.ylabel("Number of Answers")
    plt.title("Marks Distribution")
    plt.show()

if __name__ == "__main__":
    main()