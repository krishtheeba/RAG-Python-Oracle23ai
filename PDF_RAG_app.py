import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import tempfile
import os

# Streamlit page setup
st.set_page_config(page_title=" Chat with Your PDF ", layout="wide")

st.title(" Chat with Your PDF using Groq + Ollama + LangChain")
st.write("Upload a PDF, embed it with Ollama, and chat using Groq LLM.")

# Step 1: PDF Upload
uploaded_file = st.file_uploader(" Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(" PDF uploaded successfully!")

    # Step 2: Load PDF
    with st.spinner(" Loading and processing PDF..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Step 3: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Step 4: Create embeddings
        embedding_obj = OllamaEmbeddings(model="nomic-embed-text:latest")

        # Step 5: Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embedding_obj)
        retriever_obj = vectorstore.as_retriever()

    st.success("Document indexed and ready for questions!")

    # Step 6: Setup Groq LLM
    groq_api_key = st.text_input(" Enter your Groq API Key:", type="password")

    if groq_api_key:
        llm_obj = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

        # Step 7: Create prompt and chain
        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion:\n{input}"
        )

        qa_chain = create_stuff_documents_chain(llm_obj, prompt)
        rag_chain = create_retrieval_chain(retriever_obj, qa_chain)

        # Step 8: Question input
        st.subheader(" Ask a question about your document")
        user_query = st.text_input("Enter your question:")

        if st.button("Get Answer") and user_query.strip():
            with st.spinner(" Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"] if "answer" in response else response
            st.markdown("###  Answer:")
            st.write(answer)
    else:
        st.info("Please enter your Groq API key to continue.")
else:
    st.info("Upload a PDF to start.")
