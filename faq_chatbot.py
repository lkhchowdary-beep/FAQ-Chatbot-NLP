from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_data = {
    "question": [
        "What is Python?",
        "What is Machine Learning?",
        "What is NLP?",
        "What is GitHub?"
    ],

    "answer": [
        "Python is a programming language.",
        "Machine Learning is a branch of AI.",
        "NLP stands for Natural Language Processing.",
        "GitHub is a platform for version control."
    ]
}

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(faq_data["question"])

print(question_vectors.shape)

user_query = input("Ask a question: ")

user_vector = vectorizer.transform([user_query])

similarity = cosine_similarity(user_vector, question_vectors)

print(similarity)

best_match = similarity.argmax()

print("Best Match Index:", best_match)
print("Answer:", faq_data["answer"][best_match])