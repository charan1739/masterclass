import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")
st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about company HR policies!")

@st.cache_resource
def build_rag():
    loader = PyPDFDirectoryLoader("./")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    RAG_PROMPT = ChatPromptTemplate.from_template(
        """You are an HR assistant for Zyro Dynamics. Answer based ONLY on the provided HR policy context.

Context:
{context}

Question: {question}

Answer clearly and concisely."""
    )
    OOS_PROMPT = ChatPromptTemplate.from_template(
        """Is this question related to HR, company policies, or employee benefits? Reply YES or NO only.
Question: {question}
Answer:"""
    )
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    REFUSAL = "I can only answer Zyro Dynamics HR policy questions. Please contact HR for other queries."

    def ask_bot(question):
        verdict = (OOS_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
        if verdict.startswith("NO"):
            return REFUSAL, []
        retrieved = retriever.invoke(question)
        answer = rag_pipeline.invoke(question)
        sources = list({d.metadata.get("source", "Unknown") for d in retrieved})
        return answer, sources

    return ask_bot

ask_bot = build_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            answer, sources = ask_bot(prompt)
        st.markdown(answer)
        if sources:
            st.caption("📄 Sources: " + ", ".join(sources))
        st.session_state.messages.append({"role": "assistant", "content": answer})
