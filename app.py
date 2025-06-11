
import streamlit as st
from process import process_pdf
from load_model import load_embeddings, load_llm

# Make sure model are loaded each interaction
if "rag_chain" not in st.session_state: # Rag chain build from pdf
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG PDF Assistant")

st.markdown("""
**AI allows you to directly ask questions abd get answers from the contecnt of PDF documents in Vietnamese**
            
**Simple use:**
1. **Upload PDF:** Choose a PDF file to upload and click "Process PDF"
2. **Question:** Type your question about the content of file that you've just uploaded
""")

if not st.session_state.models_loaded:
    st.info("Downloading models...")
    st.session_state.embeddings = load_embeddings("bkai-foundation-models/vietnamese-bi-encoder")
    st.session_state.llm = load_llm("lmsys/vicuna-7b-v1.5")
    st.session_state.models_loaded = True
    st.success("Models loaded successfully!")
    st.rerun()

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file, st.session_state)
        st.success(f"PDF processed successfully! Number of chunks: {num_chunks}")

if st.session_state.rag_chain:
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Answering..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer: ")[1].strip() if "Answer:" in output else output
            st.write("Answer: ", answer)
            