from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## defining prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user query."),
        ("user", "Question: {question}")
    ]
)

st.title("Langchain Demo with Ollama")
input_text = st.text_input("Search the topic you want")

## using llama2
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

if input_text:
    chain = prompt | llm | output_parser
    output = chain.invoke({"question": input_text})
    st.write(output)
