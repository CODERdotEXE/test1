import streamlit as st
from transformers import pipeline
import PyPDF2

st.title("PDF Question Answering App")

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Load question answering model
qa_model = pipeline("question-answering")

# Process PDFs and answer questions
for pdf_file in uploaded_files:
    st.write(f"## {pdf_file.name}")
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()

    # Ask a question
    question = st.text_input("Ask a question:")
    if question:
        # Get the answer
        answer = qa_model({"context": text, "question": question})
        st.write(f"*Answer:* {answer['answer']}")
