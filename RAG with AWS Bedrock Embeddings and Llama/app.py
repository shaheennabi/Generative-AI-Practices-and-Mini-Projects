import json
import os
import sys
import boto3
import streamlit as st

# Updated imports
from langchain_community.llms import Bedrock
from langchain_aws import BedrockEmbeddings 

# Data Ingestion
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLm models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-text-lite-v1", client=bedrock)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Using text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, size_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs 

# Vector embeddings and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss")

def get_llama_llm():
    # Creating llama model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, 
                  model_kwargs={'max_gen_len': 512})
    return llm 

prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but at least summarize with
100 words with detailed explanation. If you don't know the answer, 
just say that you don't know, don't try to make up an answer
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": query}) 
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Llama Output"):
        faiss_index = FAISS.load_local("faiss", bedrock_embeddings)
        llm = get_llama_llm()
        st.write(get_response_llm(llm, faiss_index, user_question))
        st.success("Done")

if __name__ == "__main__":
    main()
