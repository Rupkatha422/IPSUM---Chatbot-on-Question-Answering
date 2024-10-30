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

    # Setup Page Title & Icon
    st.header("Chat with PDF!")
    with st.sidebar:
        st.title('LLM Chat App 💬')
        st.markdown('''
                    ## About
                    This is a simple chat application using LLM (Large Language Model). The model used here uses:
                    - [Streamlit](https://streamlit.io/)
                    - [LangChain](https://python.langchain.com/)
        ''')
        add_vertical_space(5)
        st.write('Made by Aditya, Aditee :), Shaswata, Rupkatha and Sharanya')

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.divider()
    img = st.file_uploader("Upload your Image", type=['png','jpeg','jpg'])

    # Check if the user has uploaded a PDF file
    if pdf is not None:
        st.write("PDF Uploaded!")

        # Extracting  text from PDF using PyPDF4 Library
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

    # Check if User has uploaded a Image File
    if img is not None:        
        st.write("Image Uploaded!")

        # Image Preprocssing Functions
        def process(img):
   
            # Resizing the image
            def reshape_image(image_path):
                new_size = (800, 500) 
                resized_img = cv2.resize(image_path, new_size)
                return resized_img
            
            resized_img = reshape_image(img)
            cv2.imwrite("resized_img.jpg", resized_img)

            # Grayscale conversion and processing        
            def grayscale(image):
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            gray_image = grayscale(resized_img)
            cv2.imwrite("gray.jpg", gray_image)
            
            def thick_font(image):
                import numpy as np
                image = cv2.bitwise_not(image)
                kernel = np.ones((1,1),np.uint8)
                image = cv2.dilate(image, kernel, iterations=1)
                image = cv2.bitwise_not(image)
                return (image)
            
            dilated_image = thick_font(gray_image)
            cv2.imwrite("dilated_image.jpg", dilated_image)
            image = "dilated_image.jpg"
            img = Image.open(image)
            text = ''
            text += pytesseract.image_to_string(img)
            # print(text)

            # Convert processed and extracted text to PDF
            def text_to_pdf(text, output_file):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=text)
                pdf.output(output_file)
            output_file = "output.pdf"
            text_to_pdf(text, output_file)

            # Image Conversion to Numpy Array
        imag=Image.open(img)
        numpy_array = np.array(imag)
        process(numpy_array)

        # Accessing PDF File
        file_path = "C:\\Users\\KIIT\\Downloads\\Minor Project Documents\\output.pdf"
        pdf_reader = PdfReader(file_path)

        # Extracting Text from the PDF
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

if __name__ == '__main__':
    main()