import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama  # Check if this is the correct import
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")  # Corrected variable name
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant. Please respond to the query"),
        ("user", "query:{text}")
    ]
)

# Streamlit template
st.title("LLM with Streamlit")
input_text = st.text_input("Please ask a query")

# Initialize the model
model = Ollama(model="gemma2:2b") 
output_parser = StrOutputParser()

# Create a chain
chain = prompt | model | output_parser  # Ensure this is the intended use

# Process user input
if input_text:
    result = chain.invoke({"text": input_text})
    st.write(result)