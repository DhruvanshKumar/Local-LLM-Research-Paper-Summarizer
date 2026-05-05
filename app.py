import os 
import tempfile
import streamlit as st
from ingestion import ingest_pdfs
from vectorstore import load_vectorstore, add_documents_to_vectorstore
from rag_pipeline import build_rag_chain, ask
st.set_page_config(page_title="Research Paper Summarizer",page_icon="📚", layout="wide")
st.title("Local Research Paper Summarizer 📚")
st.caption("Powered by LangChain - Ollama (LLaMA 3) - FAISS - sentence-transformers - 100% local, zero API cost")
with st.sidebar:
    st.header("Upload Research Papers")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("ingest PDFs", type="primary"):
            with st.spinner("Loading and chunking PDFs..."):
                tmp_paths = []
                try:
                    for uf in uploaded_files:
                        suffix = ".pdf"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp.write(uf.read())
                        tmp.flush()
                        tmp_paths.append(tmp.name)
                    chunks = ingest_pdfs(tmp_paths)
                    vectorstore = add_documents_to_vectorstore(chunks)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["rag_chain"] = build_rag_chain(vectorstore)
                    st.session_state["chat_history"] = []
                    st.success(f"Ingested {len(uploaded_files)} PDFs with {len(chunks)} chunks indexed")
                finally:
                    for p in tmp_paths:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass
    st.divider()
    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []
        if "rag_chain" in st.session_state:
            st.session_state["rag_chain"] = build_rag_chain(st.session_state["vectorstore"])
        st.rerun()
if "vectorstore" not in st.session_state:
    vs = load_vectorstore()
    if vs is not None:
        st.session_state["vectorstore"] = vs
        st.session_state["rag_chain"] = build_rag_chain(vs)
    else:
        st.session_state["vectorstore"] = None
        st.session_state["rag_chain"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.session_state["rag_chain"] is None:
    st.info("Upload and ingest at least one PDF using the sidebar to get started.")
else:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask a question about the research papers...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                result = ask(st.session_state["rag_chain"], user_input)
                answer = result["answer"]
                sources = list(dict.fromkeys(result["sources"]))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.write(f"-{os.path.basename(src)}")
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})