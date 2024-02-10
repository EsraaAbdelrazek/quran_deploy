from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the dataset
dataset_path = 'Arabic-Original.csv'
df = pd.read_csv(dataset_path)

# Load the precomputed vectorized corpus
vectorized_corpus_path = 'corpus_vectorized.joblib'
vectorized_corpus = joblib.load(vectorized_corpus_path)

# Assuming the vectorization model needs to be applied to incoming queries
# You'll need to load or recreate the vectorization model
# For this example, let's assume it's a TF-IDF model trained on the dataset
tfidf_vectorizer = TfidfVectorizer().fit(df['text'])  # Placeholder; adjust 'text' to your actual text column

@app.route('/search', methods=['POST'])
def search():
    query_text = request.json.get('query', '')
    if not query_text:
        return jsonify({"error": "Query text is missing."}), 400

    # Vectorize the query using the same model used for creating the vectorized corpus
    query_vector = tfidf_vectorizer.transform([query_text])

    # Compute similarity
    cosine_similarities = linear_kernel(query_vector, vectorized_corpus).flatten()

    # Get indices of top matches; adjust the number of results as needed
    top_indices = cosine_similarities.argsort()[-10:][::-1]

    # Retrieve matching documents' details from the dataset
    results = df.iloc[top_indices].to_dict('records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
