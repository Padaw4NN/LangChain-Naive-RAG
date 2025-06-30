"""
Author: Alexander Guimarães
GitHub: Padaw4NN
"""
import os
import argparse
import embedding
import naive_rag
import streamlit as st

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


# A flag to insert a new document to be retrieved
parser = argparse.ArgumentParser(prog="RAG", description="Naive RAG with Groq API.")
parser.add_argument("--use_new_document", type=str, help="Store new data from a new document.")
args = parser.parse_args()


# From Docker - QDrant env
QDRANT_CONFIG = {
    "host": os.environ.get("QDRANT_HOST", "localhost"), # Docker service name
    "port": int(os.environ.get("QDRANT_PORT", 6333))
}

# Environment
LLM_CONFIG = {
    "model": "llama3-8b-8192",
    "temperature": 0.1,
    "streaming": True
}

# Objects
embeddings    = HuggingFaceEmbeddings(model_name=embedding.EMBEDDING_MODEL)
qdrant_client = QdrantClient(**QDRANT_CONFIG)


# Check if the collection exists before call the vector store. If the
# collection not exists, will be stored the current file on data path.
if not qdrant_client.collection_exists(f"{embedding.QDRANT_COLLECTION}"):
    embedding.store_chunks_from_document(
        qdrant_client=qdrant_client,
        embedding=embeddings,
        metadata=embedding.load_metadata()
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    embedding=embeddings,
    collection_name=embedding.QDRANT_COLLECTION
)

llm = ChatGroq(**LLM_CONFIG)

# Front-end chat from Streamlit
st.title("Naive RAG v1.0")

def main() -> None:
    if args.use_new_document is not None:
        embedding.store_chunks_from_document(
            qdrant_client=qdrant_client,
            embedding=embeddings,
            metadata=embedding.load_metadata()
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if prompt := st.chat_input("Input"):
        st.session_state.messages.append(f"{prompt.lower()}")

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chat_response = naive_rag.perform_rag(prompt, llm, vector_store)
            st.write(chat_response.get("text"))

if __name__ == "__main__":
    main()
