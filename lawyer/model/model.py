import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("D:/lawyer/dataset/FIR_DATASET.csv")
print(df.shape)
print(df.head())
print(df.columns)
print(df.isnull().sum())
# DROPPING NULL VALUES
df=df.dropna()
print(df.isnull().sum())
print(df.shape)
print(df['Description'].values[0])
print(df['Offense'].values[0])
print(df['Punishment'].values[0])
print(df['Cognizable'].values[0])
print(df['Bailable'].values[0])
print(df['Court'].values[0])

# 2. Define a clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# 3. Apply cleaning to the offense column
df['cleaned_offense'] = df['Offense'].apply(clean_text)

# Display the first few rows of the cleaned data
print("--- Cleaned Data Sample ---")
print(df[['Offense', 'cleaned_offense']].head(2))
# --- 2. Load the S-BERT Model ---
# 'all-MiniLM-L6-v2' is a fast, efficient, yet highly effective model.
model = SentenceTransformer('all-MiniLM-L6-v2')
print("BERT-based Sentence Transformer Model loaded successfully.")

# --- 3. Generate Embeddings (Vectorization) ---
# S-BERT handles all pre-processing (tokenization, padding) internally.
# The result is a NumPy array where each row is the vector for one offense.
offense_vectors = model.encode(df['Offense'].tolist(), show_progress_bar=True)

print(f"Generated Vector Matrix Shape: {offense_vectors.shape}")
# Example: (5, 384) -> 5 offenses, each represented by a 384-dimensional vector.


# --- 4. The Retrieval Function (Using BERT Vectors) ---
def retrieve_legal_info_bert(user_query, df, model, offense_vectors):
    # Encode the user query into a vector using the SAME model
    query_vector = model.encode([user_query])
    
    # Calculate Cosine Similarity between the single query vector and all offense vectors
    similarity_scores = cosine_similarity(query_vector, offense_vectors)
    
    # Find the index of the highest score
    best_match_index = np.argsort(similarity_scores[0])[-2:][::-1]
    
    # Retrieve the full row data
    best_match_row = df.iloc[best_match_index]
    
    # Output includes the similarity score for reference
    output = {
        "Best Match Score": similarity_scores[0][best_match_index],
        "Best Match Offense": best_match_row['Offense'],
        "Cognizable": best_match_row['Cognizable'],
        "Bailable": best_match_row['Bailable'],
        "Punishment": best_match_row['Punishment'],
        "Court": best_match_row['Court'],
        "Description": best_match_row['Description']
    }
    
    return output

# --- 5. Example Usage ---
user_input_1 = input("Enter the legal offense you want information about: ")
result_1 = retrieve_legal_info_bert(user_input_1, df, model, offense_vectors)

print("\n--- BERT Retrieval Output (High Context Match) ---")
for key, value in result_1.items():
    print(f"| {key:<20} | {value}")
