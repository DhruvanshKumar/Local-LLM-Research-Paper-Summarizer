import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
VECTORSTORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
def build_vectorstore(chunks: list) -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore
def load_vectorstore() -> FAISS | None:
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
def add_documents_to_vectorstore(chunks: list) -> FAISS:
    existing = load_vectorstore()
    embeddings = get_embeddings()
    if existing is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        new_store = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new_store)
        vectorstore = existing
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore