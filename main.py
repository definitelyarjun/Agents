from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct",
    temprature = 1,
    max_tokens = 100,
    token = hf_token
)

response = llm.complete("Hey! Good morning")
print(response)