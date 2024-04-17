from transformers import AlbertTokenizer, AlbertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained ALBERT model and tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocessing
def preprocess_albert(text):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:512]  # ALBERT max input length
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    return tokens_tensor

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, model, max_seq_length=512):
    tokens_tensor1 = preprocess_albert(answer1)
    tokens_tensor2 = preprocess_albert(answer2)
    max_length = max(tokens_tensor1.size(1), tokens_tensor2.size(1))
    padding_length = max_seq_length - max_length if max_length < max_seq_length else 0

    # Pad the tensors
    tokens_tensor1_padded = torch.nn.functional.pad(tokens_tensor1, (0, padding_length), 'constant', 0)
    tokens_tensor2_padded = torch.nn.functional.pad(tokens_tensor2, (0, padding_length), 'constant', 0)

    with torch.no_grad():
        output1 = model(tokens_tensor1_padded)[0][:, :max_length, :]  # Extract the last hidden states
        output2 = model(tokens_tensor2_padded)[0][:, :max_length, :]

    embedding1 = output1.mean(dim=1)  # Average the token embeddings
    embedding2 = output2.mean(dim=1)

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
    while True:
        print("\nEnter 'exit' to quit the program.")
        question = input("Please enter a question: ")
        if question.lower() == "exit":
            break
        sample_answer = input("Please enter a sample answer: ")
        user_answer = input("Please enter the user's answer: ")

        similarity_score = evaluate_long_answer(sample_answer, user_answer, model)
        marks = assign_marks(similarity_score)
        print(f"Similarity Score: {similarity_score}")
        print(f"Marks: {marks}")

if __name__ == "__main__":
    main()