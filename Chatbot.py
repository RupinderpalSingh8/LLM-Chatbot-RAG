import streamlit as st
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    model = Ollama(model="llama2")

    question_prompt = PromptTemplate(
        input_variables=["history", "context"],
        template="Given the following conversation history and context, generate a question:\n\nHistory:\n{history}\n\nContext:\n{context}\n\nQuestion:",
    )

    question_generator = LLMChain(
        llm=model,
        prompt=question_prompt
    )

    qa_chain = load_qa_chain(model)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=qa_chain,
        memory=memory,
        question_generator=question_generator
    )
    return conversation_chain

def handle_userinput(user_question):
    # Retrieve the conversation history from memory
    if st.session_state.conversation and st.session_state.conversation.memory:
        history = st.session_state.conversation.memory.load_memory_variables({}).get("chat_history", "")
    else:
        history = ""

    # Combine the context, question, and history into a single input
    consolidated_input = f"Question: {user_question}\nContext: {user_question}\nHistory: {history}"

    # Pass this single input to the conversation chain
    response = st.session_state.conversation({'question': consolidated_input})

    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=':books:')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Chat with multiple books :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your files here and click on 'Process'", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_texts(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.text("Processing completed.")

if __name__ == '__main__':
    main()
