from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
import gradio as gr

#LLM interface
def qwencoder(query):
  hf_token = os.getenv("HF_TOKEN")
  llm = HuggingFaceInferenceAPI(
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature = 1,
    max_tokens = 100,
    token = hf_token
  )
  response = llm.complete(query)
  return response

Interface = gr.Interface(fn=qwencoder,inputs=gr.Text(label="input query"),outputs=gr.Text(label="llm response"))

Interface.launch(share=True)
