from textblob import TextBlob
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Evaluate long answer
def evaluate_long_answer(answer1, answer2):
    # Preprocess answers
    answer1_blob = TextBlob(answer1)
    answer2_blob = TextBlob(answer2)

    # Tokenize and remove stopwords
    answer1_tokens = [token.lower() for token in answer1_blob.words if token.lower() not in nlp.Defaults.stop_words]
    answer2_tokens = [token.lower() for token in answer2_blob.words if token.lower() not in nlp.Defaults.stop_words]

    # Calculate sentiment polarity
    sentiment1 = answer1_blob.sentiment.polarity
    sentiment2 = answer2_blob.sentiment.polarity

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(answer1_tokens), ' '.join(answer2_tokens)])

    # Calculate cosine similarity between TF-IDF vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Adjust the similarity score based on sentiment polarity
    if sentiment1 * sentiment2 < 0:
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
            similarity_score = evaluate_long_answer(sample_answer, user_answer)
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
    reference_similarity = evaluate_long_answer(sample_answer, sample_answer)
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