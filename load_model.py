import torch
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@st.cache_resource
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def load_llm(model_name: str, config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )
    return HuggingFacePipeline(pipeline=model_pipeline)


