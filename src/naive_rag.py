# Utils
import os
import dotenv
import warnings

# LangChain
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.simplefilter("ignore")


# Environment
dotenv.load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# --- Prompt Template --- #

PROMPT_TEMPLATE = """
Answer the query based on the context. If you don't know, just say you don't know.
All answers must be in pt-BR.

### CONTEXT:
{context}

### QUERY:
{query}
"""

template = PromptTemplate.from_template(PROMPT_TEMPLATE)

def perform_rag(
    query: str,
    llm:   ChatGroq,
    vector_store: QdrantVectorStore
) -> dict:
    """
    Performs the Naive RAG.

    args:
        query: User query.
        llm: A LLM from Groq API.
        vector_store: QDrant store from LangChain.

    returns:
        A dict containing the `text` and the context `score`.
    """
    context = vector_store.similarity_search_with_relevance_scores(query, k=3)
    context = list(filter(lambda x: x[1] >= 0.7, context))

    if len(context) == 0:
        return {
            "text": "\nEu não sou capaz de responder a isso.\n",
            "score": 0.0
        }
    
    chain = template | llm | StrOutputParser()
    response = chain.invoke({"context": context, "query": query})

    return {
        "text": "\n" + response + "\n",
        "score": context[0][1]
    }
