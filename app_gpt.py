import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = "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"

# Define the GPT-3 model parameters
model_engine = "text-davinci-003"
max_tokens = 1024
temperature = 0.7
top_p = 1.0
n = 1
stop = None

# Preprocessing
def preprocess_gpt3(text):
    return text

# Evaluate long answer
def evaluate_long_answer(answer1, answer2, max_seq_length=1024):
    prompt = f"Compare the following two texts:\n\nText 1: {answer1}\n\nText 2: {answer2}\n\nReturn a number between 0 and 1 indicating the similarity between the two texts, where 0 means completely different and 1 means identical."

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature,
        top_p=top_p
    )

    similarity_score = float(response.choices[0].text.strip())
    return similarity_score

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

        similarity_score = evaluate_long_answer(sample_answer, user_answer)
        marks = assign_marks(similarity_score)
        print(f"Similarity Score: {similarity_score}")
        print(f"Marks: {marks}")

if __name__ == "__main__":
    main()