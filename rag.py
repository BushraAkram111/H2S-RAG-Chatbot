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
from io import BytesIO

# Function to download and extract text from PDF from URL
def load_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure we got a successful response

        pdf_file = BytesIO(response.content)
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except requests.RequestException as e:
        st.error(f"Error downloading PDF: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return ""

def main():
    st.set_page_config(page_title="Hope_To_Skill AI Chatbot", page_icon=":robot_face:")
    
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

    st.markdown("""
    **HopeToSkill AI Chatbot** is designed to assist you with all your questions related to the *Hope To Skill AI Advance* course. Powered by advanced AI models, this chatbot allows you to interact with the course content through a simple chat interface. Just ask your questions, and HopeToSkill AI Chatbot will provide precise answers based on the course material.
    """)
    st.subheader("Ask a question:")
    input_query = st.text_input("Type your question here...")

    st.sidebar.subheader("Google API Key")
    user_google_api_key = st.sidebar.text_input("Enter your Google API key (Optional)", type="password")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    pdf_url = "https://drive.google.com/uc?id=1cTZoYuVeLDB7o9iEWlCwddunCMsjpK26"
    default_google_api_key = ""
    google_api_key = user_google_api_key if user_google_api_key else default_google_api_key

    if st.session_state.processComplete is None:
        files_text = load_pdf_from_url(pdf_url)
        if files_text:  # Proceed only if PDF was successfully read
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = vectorstore
            st.session_state.processComplete = True
        else:
            st.error("Failed to process the PDF file.")

    if input_query:
        response_text = rag(st.session_state.conversation, input_query, google_api_key)
        st.session_state.chat_history.append({"content": input_query, "is_user": True})
        st.session_state.chat_history.append({"content": response_text, "is_user": False})

    response_container = st.container()
    with response_container:
        for i, message_data in enumerate(st.session_state.chat_history):
            message(message_data["content"], is_user=message_data["is_user"], key=str(i))

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

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
