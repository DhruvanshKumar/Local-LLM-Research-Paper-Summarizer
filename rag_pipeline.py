from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
OLLAMA_MODEL = "llama3"
def build_rag_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    return chain
def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": [
            doc.metadata.get("source", "Unknown")
            for doc in result.get("source_documents", [])
        ],
    }