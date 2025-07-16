import streamlit as st
import os
import tempfile
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import CrossEncoder
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from handwrittingocr import HandwritingOCR

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


st.set_page_config(page_title="RAG Question Answer")

def process_document(uploaded_file: UploadedFile, use_handwriting_ocr: bool = False) -> list[Document]:
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()  # Properly close the file before using it

    if use_handwriting_ocr:
        # Use handwriting OCR pipeline
        st.info("🔍 Processing with handwriting recognition...")
        ocr = HandwritingOCR()
        extracted_text = ocr.process_pdf(temp_file.name)

        # Create a document from the extracted text
        docs = [Document(page_content=extracted_text, metadata={"source": uploaded_file.name})]
    else:
        # Use regular PDF loader
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

    os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        # Only add non-empty documents
        if split.page_content.strip():
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

    if documents:  # Only upsert if we have documents
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        st.success("Data added to the vector store!")
    else:
        st.warning("No text content found in the document. Please check if the file contains readable text.")

def reset_vector_database():
    """Reset/clear the entire ChromaDB vector database"""
    try:
        # Method 1: Delete the collection
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
        try:
            chroma_client.delete_collection(name="rag_app")
            st.success("✅ Collection 'rag_app' deleted successfully!")
        except Exception as e:
            if "does not exist" in str(e).lower():
                st.info("ℹ️ Collection 'rag_app' doesn't exist (already empty)")
            else:
                st.warning(f"Could not delete collection: {e}")

        # Method 2: Remove the entire ChromaDB directory
        if os.path.exists("./demo-rag-chroma"):
            shutil.rmtree("./demo-rag-chroma")
            st.success("🗑️ ChromaDB directory completely removed!")
        else:
            st.info("ℹ️ ChromaDB directory doesn't exist")

        st.success("🎉 Vector database reset complete! You can now upload new documents.")

    except Exception as e:
        st.error(f"❌ Error resetting database: {e}")

def get_collection_info():
    """Get information about the current collection"""
    try:
        collection = get_vector_collection()
        count = collection.count()
        return count
    except Exception as e:
        return f"Error: {e}"


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    #st.write(ranks) #commented out for now as it is not needed
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    with st.sidebar:
        st.header("📄 RAG Question Answer")

        # Show current database status
        doc_count = get_collection_info()
        if isinstance(doc_count, int):
            st.info(f"📊 Current documents in database: {doc_count}")
        else:
            st.info("📊 Database status: Empty or not initialized")

        uploaded_files = st.file_uploader(
            "**📂 Upload PDF files for QnA**",
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader_main"
        )

        # Add handwriting recognition option
        use_handwriting_ocr = st.checkbox(
            "🖋️ Enable Handwriting Recognition",
            value=False,
            help="Use CRAFT + TrOCR for handwritten text detection and recognition"
        )

        process = st.button("⚙️ Process")

        st.divider()

        # Database management section
        st.subheader("🗄️ Database Management")

        # Reset button with confirmation
        if st.button("🗑️ Reset Vector Database", type="secondary"):
            reset_vector_database()
            st.rerun()  # Refresh the app to update the document count
    
    if uploaded_files and process:
        for uploaded_file in uploaded_files:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file, use_handwriting_ocr)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

        processing_method = "handwriting recognition" if use_handwriting_ocr else "standard PDF processing"
        st.success(f"Successfully processed {len(uploaded_files)} file(s) using {processing_method}!")

    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
    "🔥 Ask",
    )
    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)