import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import spacy

# Step 1: Data Collection
# Example data
nltk.download('punkt')
nltk.download('stopwords')
questions = [
    "what is computer architecture?"
]
sample_answers = [
    "Computer architecture encompasses the design and organization of a computer system's components, comprising hardware and software elements that collaborate to execute instructions and accomplish tasks efficiently. This includes the central processing unit (CPU), memory modules, input/output (I/O) devices, and the communication pathways (such as buses) connecting them. Architectural decisions regarding instruction sets, addressing modes, data formats, and memory hierarchy profoundly influence a computer system's performance and capabilities. Various paradigms, like von Neumann and Harvard architectures, offer different approaches to organizing these components, while levels of abstraction from hardware design to software development provide different perspectives on the system's architecture. Overall, computer architecture dictates how effectively a system can execute programs and handle diverse computing demands."
]

user_answer = "Computer structure is like the plan of a computer's brain and body. Just like a plan outlines how a building is made, computer structure shows how all the parts of a computer—like the main processor (CPU), memory, and input/output tools—are organized and how they work together to process facts and do tasks. It's about finding the best way to design and connect these parts so that the computer can do things well and fast. Think of it as planning the most effective and efficient layout for a city so that everything runs well."

# Step 2: Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase conversion
    tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

preprocessed_sample_answers = [preprocess(answer) for answer in sample_answers]
preprocessed_user_answer = preprocess(user_answer)

# Step 3: Keyword Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sample_answers + [preprocessed_user_answer])

# Step 4: Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

sample_answer_sentiments = [analyze_sentiment(answer) for answer in sample_answers]
user_answer_sentiment = analyze_sentiment(user_answer)

# Step 5: Context Matching (Cosine Similarity)
def calculate_similarity(matrix):
    similarity_scores = cosine_similarity(matrix)
    return similarity_scores[:-1, -1]

similarity_scores = calculate_similarity(tfidf_matrix)

# Step 6: Scoring and Evaluation
def calculate_score(similarity_scores, sentiments):
    weighted_similarity = similarity_scores.mean()
    weighted_sentiment = sum(sentiments) / len(sentiments)
    # You can define your own weighting scheme based on the importance of each aspect
    return 0.6 * weighted_similarity + 0.4 * weighted_sentiment

evaluation_score = calculate_score(similarity_scores, sample_answer_sentiments)
def get_mark(score):
    if score >= 0.4:
        return 5
    elif score >= 0.3:
        return 4
    elif score >= 0.2:
        return 3
    elif score >= 0.1:
        return 2
    else:
        return 1

evaluation_mark = get_mark(evaluation_score)

print("Evaluation Score:", evaluation_score)
print("Evaluation Mark:", evaluation_mark)

#print("Evaluation Score:", evaluation_score)
