from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer

def initialize_llm(model_path, model_type="llama"):
    llm = CTransformers(model=model_path, model_type=model_type, max_new_tokens=512, temperature=0.1)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return llm, tokenizer

def create_conversational_chain(llm, docsearch):
    return ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())