from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

#RAG Prerequisites
reader = SimpleDirectoryReader(input_dir = "C:\Users\arjun\OneDrive\Documents\Agents\os.pdf")
document = reader.load_data()
db = chromadb.PersistentClient(
    path = "C:\Users\arjun\OneDrive\Documents\Agents\chroma_db"
)
chroma_collection = db.create_collection(name = "os_collection")
vector_store = ChromaVectorStore(chroma_collection = chroma_collection)

#RAG Implementation
pipeline = IngestionPipeline(
         transformation = [
             SentenceSplitter(chunk_overlap = 0),
             HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
)

#LLM interface
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature = 1,
    max_tokens = 100,
    token = hf_token
)

response = llm.complete("How are llamas born?")
print(response)
