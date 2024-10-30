# Basic Utilities Libraries
import numpy as np
import os

# Streamlit Libraries
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# PDF Function Libraries
from PyPDF2 import PdfReader
from fpdf import FPDF

# Image Processing Libraries
import cv2
import pytesseract
from PIL import Image

# Langchain Models and Embeddings Libraries
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langchain Text Processing Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Vector sentence computaion library
import sentence_transformers

# Langchain Vector Storage processing Libraries
from langchain_community.vectorstores import FAISS

# Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Model ID
# ---"NousResearch/Llama-2-7b-chat-hf"
# ---"google/flan-t5-large" 
repo_id = "google/flan-t5-large" 

# Main Function
def main():
    
    st.header("Chat with PDF!")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

        # Extracting  text from PDF using PyPDF4 Library
    try:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

            # Creating and Storing Embeddings
        embeddings = HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

            # Displaying Results from Model's Query Engine
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = HuggingFaceHub(repo_id=repo_id,
                                model_kwargs={"temperature": 0.75, "max_length": 512})
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    except Exception as e:
        st.error("No PDF uploaded!")


# Running the App
if __name__ == '__main__':
    st.set_page_config(
        page_title="Document Chat App",
        page_icon="📚",
    )
    main()