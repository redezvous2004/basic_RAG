import tempfile
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain import hub

def remove_invalid_surrogates(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

def process_pdf(uploaded_file, session_state):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmpf.write(uploaded_file.getvalue())
        tmpf_path = tmpf.name
    
    loader = PyPDFLoader(tmpf_path)
    documents = loader.load()

    semanctic_splitter = SemanticChunker(
        embeddings=session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semanctic_splitter.split_documents(documents=documents)

    for doc in docs:
        doc.page_content = remove_invalid_surrogates(doc.page_content)

    vec_db = Chroma.from_documents(
        documents=docs,
        embedding=session_state.embeddings
    )
    retriever = vec_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmpf_path)  # Clean up the temporary file
    return rag_chain, len(docs)
