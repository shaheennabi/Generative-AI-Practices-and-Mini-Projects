import streamlit as st
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Initialize the NVIDIA language model
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Set up the Streamlit app title
st.title("Nvidia NIM Demo")  # Calling from Nvidia (Nvidia NIM)

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

# User input for questions
prompt1 = st.text_input("Enter your Question from PDFs")

# Button to create document embeddings
if st.button("Document Embeddings"):
    vector_embedding()
    st.write("Vector Store is Ready using Nvidia Embeddings")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.process_time()  
    response = retrieval_chain.invoke({"input": prompt1})  
    response_time = time.process_time() - start_time  

    # Display the response time
    print("Response time: ", response_time)
    st.write(response['answer'])

    # Expandable section for document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------")
