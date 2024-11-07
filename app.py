import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import os

# Constants
EMBEDDINGS_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
CHAT_HISTORY_LIMIT = 5 
ROWS_PER_PAGE = 10

# 1. Load CSV Data
def load_csv_data(file):
    data = pd.read_csv(file)
    return data

# 2. Split Data into Chunks
def split_text_into_chunks(data, chunk_size=500, chunk_overlap=20):
    documents = []
    for _, row in data.iterrows():
        text = ' '.join(row.astype(str).values)
        document = Document(page_content=text)
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks

# 3. Create Embeddings
def create_embeddings(text_chunks, model_name=EMBEDDINGS_MODEL_NAME):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(text_chunks, embeddings)

# 5. Initialize Conversational Chain
def create_conversational_chain(docsearch):
    llm = CTransformers(model=LLM_MODEL_PATH, model_type="llama", max_new_tokens=512, temperature=0.1)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
    return qa

# 6. Handle User Interaction and Query Processing
def run_conversation(qa, chat_history, query):
    # Keep only the last few exchanges for performance
    if len(chat_history) > CHAT_HISTORY_LIMIT:
        chat_history = chat_history[-CHAT_HISTORY_LIMIT:]
        
    result = qa({"question": query + " according to the provided data", "chat_history": chat_history})
    chat_history.append((query, result['answer']))  # Update chat history
    return result['answer'], chat_history

# Main function to tie everything together
def main():
    st.title("CSV Data Query System")
    st.write("This app allows you to query data extracted from the CSV file.")

    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if csv_file is not None:
        data = load_csv_data(csv_file)
        st.write(f"Data loaded from {csv_file.name}, containing {len(data)} rows.")

        # Pagination setup for data display
        total_pages = len(data) // ROWS_PER_PAGE + (1 if len(data) % ROWS_PER_PAGE != 0 else 0)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        # Calculate the start and end rows for the current page
        start_row = (page - 1) * ROWS_PER_PAGE
        end_row = start_row + ROWS_PER_PAGE
        st.dataframe(data.iloc[start_row:end_row])

        text_chunks = split_text_into_chunks(data)
        st.write(f"Data split into {len(text_chunks)} chunks.")

        docsearch = create_embeddings(text_chunks)

        qa = create_conversational_chain(docsearch)

        chat_history = []

        query = st.text_input("Ask a question based on the data:", "")

        if st.button("Submit") and query:
            answer, chat_history = run_conversation(qa, chat_history, query)
            st.write("Answer:", answer) 
            print("Answer:", answer)

    else:
        st.write("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()