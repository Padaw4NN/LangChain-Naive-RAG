# Utils
import os
import json
import warnings
from types import FunctionType

# LangChain
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# QDrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

warnings.simplefilter("ignore")

# Environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
QDRANT_COLLECTION = "geoproc-RAG"


def load_metadata(fname: str = "metadata.json") -> dict:
    """
    """
    metadata_path = os.path.join("./data", fname)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("\n[*] Metadata not fount. This file must be loaded with document together.")
    
    with open(metadata_path, mode='r') as f:
        metadata = json.load(f)

    return metadata

# Insert chunks in the QDrant Collection
def store_chunks_from_document(
    fname: str = "document.txt",
    metadata: dict = None,
    chunk_size: int  = 1000,
    chunk_overlap: int  = 0,
    len_function: FunctionType = len,
    qdrant_client: QdrantClient = None,
    embedding: HuggingFaceEmbeddings = None
) -> None:
    """
    Create a list of documents (chunks) from a text file
    and store in the collection.

    args:
        fname: str: text file name.
        metadata: dict: a metadata to make a better search.
        chunk_size: int: number of character per chunk.
        chunk_overlap: int: number of character overlapping the next chunk.
        len_function: FunctionType: will be used for calculate the length.
    """

    doc_path = os.path.join("./data", fname)
    if not os.path.exists(doc_path):
        raise FileNotFoundError("\n[*] The text document not exists.")

    with open(doc_path, mode='r') as f:
        text = f.read()

    documents = [Document(page_content=text, metadata=metadata)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_function
    )
    chunks = text_splitter.split_documents(documents)

    if qdrant_client is None:
        raise ValueError("\n[*] The QDrant client must not be None.")

    if not qdrant_client.collection_exists(f"{QDRANT_COLLECTION}"):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=768, # 3072,
                distance=Distance.COSINE
            )
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        embedding=embedding,
        collection_name=QDRANT_COLLECTION
    )

    result = vector_store.add_documents(chunks)
    if len(result) == 0:
        raise Exception("\n[*] No vector stored.")
