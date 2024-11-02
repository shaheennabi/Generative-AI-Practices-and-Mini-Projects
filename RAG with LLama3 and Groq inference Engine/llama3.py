import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()

# Get the API key from environment variables
groq_api_key = os.environ.get('GROQ_API_KEY')

# Streamlit app title
st.title("Chatgroq with Llama3 Demo")

# Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Define a function for vector embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize OllamaEmbeddings with the model name
        st.session_state.embeddings = OllamaEmbeddings(model="your_model_name_here")  # Replace with actual model name
        st.session_state.loader = PyPDFDirectoryLoader("./us-census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        print("Loaded documents:", len(st.session_state.docs))  # Debugging line
        
        # Check loaded document content
        for doc in st.session_state.docs:
            print(doc.page_content)  # Ensure documents contain text
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:2])  # Splitting
        print("Final documents after splitting:", len(st.session_state.final_documents))  # Debugging line
        
        if st.session_state.final_documents:
            # Generate embeddings from the final documents
            embeddings = st.session_state.embeddings.embed_documents(st.session_state.final_documents)
            print("Generated embeddings:", len(embeddings))  # Check how many embeddings were created
            
            if embeddings:  # Check if embeddings are generated
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            else:
                print("No embeddings generated.")
        else:
            print("No final documents to create embeddings.")

# Input box for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button for embedding documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Ensure that vectors is initialized before using it
if "vectors" in st.session_state:
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time :", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander for document similarity search
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.warning("Please embed the documents first.")
