import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings  # Updated import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ.get('GROQ_API_KEY')

# Initialize vector and embeddings in session state if not already done
if "vector" not in st.session_state:
    # Initialize embeddings with the specified model name
    st.session_state.embeddings = OllamaEmbeddings(model="llama2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    # Rest of the code...

    # Split documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create FAISS vector store from documents
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit title
st.title("ChatGroq Demo")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input prompt from user
prompt = st.text_input("Input your prompt here")

# Handle prompt and display response
if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    response_time = time.process_time() - start
    print("Response time:", response_time)
    st.write(response['answer'])

    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
