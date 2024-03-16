import streamlit as st
import faiss
import numpy as np
import fitz  # PyMuPDF for PDF processing
import textwrap
from langchain_openai import OpenAIEmbeddings
import os


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load the FAISS index from disk
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# Function to extract text from PDF and split into chunks
def extract_text_from_pdf_and_chunk(pdf_path, chunk_size=1000):
    document = fitz.open(pdf_path)
    texts = []
    for page in document:
        texts.append(page.get_text())
    full_text = ' '.join(texts)
    chunks = textwrap.wrap(full_text, width=chunk_size)
    return chunks

# Load your documents based on the FAISS index creation
def load_documents(pdf_path, chunk_size=1000):
    return extract_text_from_pdf_and_chunk(pdf_path, chunk_size)

# Embed a query and search the index
def search_index(query, index, documents):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")  # Ensure this matches the model used for index creation
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    D, I = index.search(query_embedding, k=5)

    # Handling the case where documents are chunks of a single PDF
    results = [documents[i] if i < len(documents) else "Document index out of bounds. Please check the FAISS index and documents alignment." for i in I[0]]
    return results

# Streamlit app interface
st.title('FAISS Index Search with Streamlit')

pdf_path = "Data/Datasheet3.pdf"
index_path = "your_faiss_index.faiss"

index = load_faiss_index(index_path)
documents = load_documents(pdf_path)

# User input
user_query = st.text_input('Enter your search query:', '')

if st.button('Search'):
    if user_query:
        # Perform the search
        results = search_index(user_query, index, documents)
        # Display results
        for result in results:
            st.write(result)
    else:
        st.write('Please enter a query to search.')
