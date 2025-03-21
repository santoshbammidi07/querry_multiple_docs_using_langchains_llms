import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from mistralai import Mistral
import json
import base64
from pathlib import Path

def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        for page in pages:
            text += page.page_content + "\n"
    return text

def extract_text_from_ppt(ppt_docs):
    text = ""
    for ppt in ppt_docs:
        loader = UnstructuredPowerPointLoader(ppt)
        slides = loader.load()
        for slide in slides:
            text += slide.page_content + "\n"
    return text

def perform_ocr_on_images(docs, client):
    ocr_text = ""
    for doc in docs:
        file_path = Path(doc.name)
        content = doc.read()
        encoded_file = base64.b64encode(content).decode()
        base64_url = f"data:application/octet-stream;base64,{encoded_file}"
        
        response = client.ocr.process(document=base64_url, model="mistral-ocr-latest")
        response_dict = json.loads(response.json())
        for page in response_dict.get("pages", []):
            ocr_text += page.get("markdown", "") + "\n"
    return ocr_text

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    return ConversationalRetrievalChain.from_llm(
        llm=OpenAIEmbeddings(), retriever=vectorstore.as_retriever()
    )

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        st.write(message.content)

def main():
    st.set_page_config(page_title="Chat with PDFs and PPTs", page_icon=":books:")
    st.header("Chat with PDFs & PPTs :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        uploaded_ppts = st.file_uploader("Upload PPTs", accept_multiple_files=True, type=['ppt', 'pptx'])
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                client = Mistral(api_key="your-mistral-api-key")
                
                pdf_text = extract_text_from_pdf(uploaded_pdfs) if uploaded_pdfs else ""
                ppt_text = extract_text_from_ppt(uploaded_ppts) if uploaded_ppts else ""
                ocr_text = perform_ocr_on_images(uploaded_pdfs + uploaded_ppts, client)
                
                combined_text = pdf_text + ppt_text + ocr_text
                text_chunks = combined_text.split("\n")
                
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
if __name__ == '__main__':
    main()
