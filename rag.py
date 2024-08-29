import os
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import requests

# Function to download and extract text from PDF from URL
def load_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    with open("downloaded_pdf.pdf", "wb") as f:
        f.write(response.content)

    text = ""
    pdf_reader = PdfReader("downloaded_pdf.pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="Hope_To_Skill AI Chatbot", page_icon=":robot_face:")
    
    # Display logo and title on the same line
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .logo {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            overflow: hidden;
            margin-right: 15px;
        }
        .logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .title {
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        <div class="header-container">
            <div class="logo">
                <img src="https://yt3.googleusercontent.com/G5iAGza6uApx12jz1CBkuuysjvrbonY1QBM128IbDS6bIH_9FvzniqB_b5XdtwPerQRN9uk1=s900-c-k-c0x00ffffff-no-rj" alt="Logo">
            </div>
            <div class="title">
                HopeToSkill AI Chatbot
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Description of the chatbot
    st.markdown("""
    **HopeToSkill AI Chatbot** is designed to assist you with all your questions related to the *Hope To Skill AI Advance* course. Powered by advanced AI models, this chatbot allows you to interact with the course content through a simple chat interface. Just ask your questions, and HopeToSkill AI Chatbot will provide precise answers based on the course material.
    """)
    st.subheader("Ask a question:")
    input_query = st.text_input("Type your question here...")

    # Sidebar for API Key
    st.sidebar.subheader("Google API Key")
    user_google_api_key = st.sidebar.text_input("Enter your Google API key (Optional)", type="password")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Use the direct download link for Google Drive PDF
    pdf_url = "https://drive.google.com/uc?id=1cTZoYuVeLDB7o9iEWlCwddunCMsjpK26"
    default_google_api_key = ""
    
    google_api_key = user_google_api_key if user_google_api_key else default_google_api_key

    # Process the PDF in the background (hidden from user)
    if st.session_state.processComplete is None:
        files_text = load_pdf_from_url(pdf_url)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = vectorstore
        st.session_state.processComplete = True

    # Chatbot functionality
    if input_query:
        response_text = rag(st.session_state.conversation, input_query, google_api_key)
        st.session_state.chat_history.append({"content": input_query, "is_user": True})
        st.session_state.chat_history.append({"content": response_text, "is_user": False})

    # Display chat history
    response_container = st.container()
    with response_container:
        for i, message_data in enumerate(st.session_state.chat_history):
            message(message_data["content"], is_user=message_data["is_user"], key=str(i))

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# Function to generate vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

# Function to perform question answering with Google Generative AI
def rag(vector_db, input_query, google_api_key):
    try:
        template = """You are an AI assistant that assists users by providing answers to their questions by extracting information from the provided context:
        {context}.
        If you do not find any relevant information from context for the given question, simply say 'I do not know'. Do not try to make up an answer.
        Answer should not be greater than 5 lines.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        response = rag_chain.invoke(input_query)
        return response
    except Exception as ex:
        return str(ex)

if __name__ == '__main__':
    main()
