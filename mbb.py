import streamlit as st
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_average_vector(words):
    embeddings = [get_embedding(x, model='text-embedding-3-small') for x in words]
    
    if len(words) == 0:
        return np.zeros(model.vector_size)
    
    # Compute average vector
    word_vectors = np.array(embeddings)
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector

def cosine_similarity(vec1, vec2):
    # Compute cosine similarity between two vectors
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim


# Streamlit UI
st.title('Mind-Body Map Comparison')

text1 = st.text_area("Before", "Enter each word separated by a comma.")
text2 = st.text_area("After","Enter each word separated by a comma.")

if st.button('Compare'):
    words1 = text1.split(',')
    words2 = text2.split(',')
    
    avg_vec1 = get_average_vector(words1)
    avg_vec2 = get_average_vector(words2)
    
    similarity = cosine_similarity(avg_vec1, avg_vec2)*100
    
    st.write(f'Similarity between before and after: {similarity:.4f}%')
