import streamlit as st
import os
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
import google.generativeai as genai
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini API Configuration
genai.configure(api_key="AIzaSyBrWpabmaYMSpSJsM9RrDKPd1JKIr6quyM")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Initialize LangChain embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyBrWpabmaYMSpSJsM9RrDKPd1JKIr6quyM")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

st.set_page_config(page_title="Parallel PDF Analyzer with Gemini", layout="wide")
st.title("📚 Bank Document Compliance Analyzer")

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Initialize session state
if 'central_bank_uploaders' not in st.session_state:
    st.session_state.central_bank_uploaders = 1
if 'private_bank_uploaders' not in st.session_state:
    st.session_state.private_bank_uploaders = 1
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {}  # {file_key: (text, file_path, doc_id, visible, keywords)}
if 'central_bank_doc_ids' not in st.session_state:
    st.session_state.central_bank_doc_ids = []
if 'private_bank_doc_ids' not in st.session_state:
    st.session_state.private_bank_doc_ids = []

# -------- Utility Functions -------- #

def extract_pdf_text(pdf_path, txt_path):
    try:
        client = LLMWhispererClientV2(
            base_url="https://llmwhisperer-api.us-central.unstract.com/api/v2",
            api_key="OoidxZhSSdqG5Tqr-FZkYaJNaKKlVYH4NLHLEu6NLtE"
        )
        whisper = client.whisper(
            file_path=pdf_path,
            wait_for_completion=True,
            wait_timeout=200
        )
        text = whisper['extraction']['result_text'].strip()
        if not text:
            text = "No text extracted."
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except LLMWhispererClientException as e:
        raise RuntimeError(f"LLM Whisperer text extraction failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during text extraction: {e}")

def gemini_prompt(prompt_text):
    try:
        return model.generate_content(prompt_text).text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {e}")

def store_in_vector_db(txt_path, metadata, doc_id):
    try:
        if not os.path.exists(txt_path):
            st.error(f"Failed to store document: Text file {txt_path} does not exist.")
            return False
        if os.path.getsize(txt_path) == 0:
            st.error(f"Failed to store document: Text file {txt_path} is empty.")
            return False

        loader = TextLoader(txt_path, encoding="utf-8")
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200))
        
        if not documents:
            st.error(f"Failed to store document: No content extracted from {txt_path}.")
            return False

        valid_documents = []
        for i, doc in enumerate(documents):
            if not doc.page_content.strip() or doc.page_content == "No text extracted.":
                continue
            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **metadata,
                    'id': doc_id,
                    'chunk_index': i,
                    'source': txt_path
                }
            )
            valid_documents.append(new_doc)
        
        if not valid_documents:
            st.error(f"Failed to store document: No valid content to store.")
            return False

        vector_store.add_documents(valid_documents)
        results = vector_store.get(where={"id": doc_id})
        if results and results.get('documents') and len(results['documents']) > 0:
            return True
        else:
            st.error(f"Failed to store document in vector DB for ID: {doc_id}")
            return False
    except Exception as e:
        st.error(f"Failed to store document in vector DB: {e}")
        return False

def retrieve_from_vector_db(doc_id):
    try:
        results = vector_store.get(where={"id": doc_id})
        if not results or 'documents' not in results or not results['documents']:
            st.error(f"No document found in vector DB for ID: {doc_id}")
            return None, None
        combined_text = "\n\n".join(results['documents'])
        metadata = results['metadatas'][0] if results['metadatas'] else {}
        return combined_text, metadata
    except Exception as e:
        st.error(f"Failed to retrieve document from vector DB: {e}")
        return None, None

# -------- Main Logic -------- #

col1, col2 = st.columns(2)

def process_file(uploaded_file, column, column_label, doc_type, uploader_index, file_key):
    if uploaded_file:
        # Check if file has already been processed
        if file_key in st.session_state.processed_docs:
            text, file_path, doc_id, visible, keywords = st.session_state.processed_docs[file_key]
            column.markdown(f"### 📘 {column_label} {uploader_index + 1}: `{uploaded_file.name}`")
            visibility = st.selectbox(f"Visibility for Document {uploader_index + 1}", ["Show", "Hide"], key=f"visibility_{file_key}", index=0 if visible else 1)
            if visibility == "Show":
                st.session_state.processed_docs[file_key] = (text, file_path, doc_id, True, keywords)
                file_status = column.empty()
                file_status.success(f"✅ Document already processed.")
                with column:
                    st.subheader(f"📝 Extracted Text (Document {uploader_index + 1})")
                    st.text_area("Extracted Text", text, height=250, key=f"extracted_{file_key}")
                    wc = len(text.split())
                    pc = len([p for p in text.split('\n') if p.strip()])
                    st.markdown(f"**Words**: {wc}, **Paragraphs**: {pc}")

                    st.subheader(f"🧠 Summary (Document {uploader_index + 1})")
                    summary = gemini_prompt(f"Summarize this banking/compliance-related document:\n{text}")
                    st.write(summary)

                    st.subheader(f"🔑 Keywords")
                    search_query = st.text_input(f"Search keywords for {uploaded_file.name}", key=f"search_{file_key}")
                    filtered_keywords = [kw.strip() for kw in keywords if search_query.lower() in kw.lower()] if search_query else keywords
                    st.subheader("Tags:")
                    if filtered_keywords:
                        for keyword in filtered_keywords:
                            st.write(keyword)
                    else:
                        st.write("No keywords match the search.")

                    txt_path = os.path.splitext(file_path)[0] + ".txt"
                    with open(txt_path, "rb") as f:
                        st.download_button(f"⬇️ Download Extracted Text (Document {uploader_index + 1})", f, file_name=os.path.basename(txt_path), mime="text/plain")
            else:
                st.session_state.processed_docs[file_key] = (text, file_path, doc_id, False, keywords)
            return text, file_path, doc_id

        column.markdown(f"### 📘 {column_label} {uploader_index + 1}: `{uploaded_file.name}`")
        file_status = column.empty()

        if uploaded_file.size > MAX_FILE_SIZE:
            file_status.warning("⚠️ Skipped: File exceeds 10MB limit.")
            return None, None, None

        with column:
            with st.spinner(f"Processing document {uploaded_file.name}..."):
                try:
                    file_path = os.path.join("data", f"{uuid.uuid4()}_{uploaded_file.name}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    txt_path = os.path.splitext(file_path)[0] + ".txt"
                    extracted_text = extract_pdf_text(file_path, txt_path)

                    visibility = st.selectbox(f"Visibility for Document {uploader_index + 1}", ["Show", "Hide"], key=f"visibility_{file_key}", index=0)
                    if visibility == "Show":
                        st.subheader(f"📝 Extracted Text (Document {uploader_index + 1})")
                        st.text_area("Extracted Text", extracted_text, height=250, key=f"extracted_{file_key}")
                        wc = len(extracted_text.split())
                        pc = len([p for p in extracted_text.split('\n') if p.strip()])
                        st.markdown(f"**Words**: {wc}, **Paragraphs**: {pc}")

                        st.subheader(f"🧠 Summary (Document {uploader_index + 1})")
                        summary = gemini_prompt(f"Summarize this banking/compliance-related document:\n{extracted_text}")
                        st.write(summary)

                        st.subheader(f"🔑 Keywords")
                        # Use LLM to generate keywords
                        keyword_prompt = f"Extract the top 10 most relevant keywords from this Vietnamese banking/compliance-related document in Vietnamese:\n{extracted_text}\nReturn the keywords as a comma-separated list."
                        keywords_raw = gemini_prompt(keyword_prompt)
                        # Clean and split the keywords
                        keywords = [kw.strip() for kw in keywords_raw.split(',') if kw.strip()]
                        search_query = st.text_input(f"Search keywords for {uploaded_file.name}", key=f"search_{file_key}")
                        filtered_keywords = [kw.strip() for kw in keywords if search_query.lower() in kw.lower()] if search_query else keywords
                        st.subheader("Tags:")
                        if filtered_keywords:
                            for keyword in filtered_keywords:
                                st.write(keyword)
                        else:
                            st.write("No keywords match the search.")

                        with open(txt_path, "rb") as f:
                            st.download_button(f"⬇️ Download Extracted Text (Document {uploader_index + 1})", f, file_name=os.path.basename(txt_path), mime="text/plain")

                    doc_id = f"{doc_type}_{uuid.uuid4()}"
                    metadata = {"file_name": uploaded_file.name, "doc_type": doc_type}
                    if store_in_vector_db(txt_path, metadata, doc_id):
                        file_status.success(f"✅ Processed and stored document {uploader_index + 1} successfully.")
                        # Cache the result with keywords
                        st.session_state.processed_docs[file_key] = (extracted_text, file_path, doc_id, visibility == "Show", keywords)
                        return extracted_text, file_path, doc_id
                    else:
                        file_status.error(f"❌ Failed to store document {uploader_index + 1} in vector DB.")
                        return None, None, None
                except Exception as e:
                    file_status.error(f"❌ Error processing document {uploader_index + 1}: {str(e)}")
                    return None, None, None
    else:
        column.info(f"Please upload PDF file {uploader_index + 1}.")
        return None, None, None

# File uploaders for Central Bank
with col1:
    st.markdown("### Central Bank Documents")
    for i in range(st.session_state.central_bank_uploaders):
        file_key = f"central_{i}"
        uploaded_file = st.file_uploader(f"Upload Central Bank PDF {i + 1}", type=["pdf"], key=file_key)
        text, file_path, doc_id = process_file(uploaded_file, col1, "Central Bank Document", "central_bank", i, file_key)
        if doc_id and doc_id not in st.session_state.central_bank_doc_ids:
            st.session_state.central_bank_doc_ids.append(doc_id)
    if st.button("➕ Add Central Bank Document", key="add_central"):
        st.session_state.central_bank_uploaders += 1
        st.rerun()

# File uploaders for Private Bank
with col2:
    st.markdown("### Private Bank Documents")
    for i in range(st.session_state.private_bank_uploaders):
        file_key = f"private_{i}"
        uploaded_file = st.file_uploader(f"Upload Private Bank PDF {i + 1}", type=["pdf"], key=file_key)
        text, file_path, doc_id = process_file(uploaded_file, col2, "Private Bank Document", "private_bank", i, file_key)
        if doc_id and doc_id not in st.session_state.private_bank_doc_ids:
            st.session_state.private_bank_doc_ids.append(doc_id)
    if st.button("➕ Add Private Bank Document", key="add_private"):
        st.session_state.private_bank_uploaders += 1
        st.rerun()

# Compare button (unchanged)
st.markdown("---")
if st.session_state.central_bank_doc_ids and st.session_state.private_bank_doc_ids:
    if st.button("🔍 Compare All Documents"):
        status_message = st.empty()
        status_message.info("📊 Performing compliance gap analysis for all documents...")

        with st.spinner("Analyzing compliance gaps..."):
            try:
                # Retrieve all texts
                central_texts = []
                private_texts = []
                for doc_id in st.session_state.central_bank_doc_ids:
                    text, metadata = retrieve_from_vector_db(doc_id)
                    if text:
                        central_texts.append((text, metadata))
                for doc_id in st.session_state.private_bank_doc_ids:
                    text, metadata = retrieve_from_vector_db(doc_id)
                    if text:
                        private_texts.append((text, metadata))

                if not central_texts or not private_texts:
                    status_message.error("❌ Failed to retrieve all documents from vector DB.")
                    st.stop()

                # Combine texts for comparison
                central_combined = "\n\n".join([text for text, _ in central_texts])
                private_combined = "\n\n".join([text for text, _ in private_texts])

                comparison_prompt = (
                    "You are a document analyser."
                    "The *Central Bank* documents contain rules to be followed by the *Private Bank*."
                    "The *Private Bank* documents are bank policy documents from a private bank in Vietnam."
                    "The *Central Bank* documents contain the regulatory terms defined by the Vietnamese central bank."
                    "All banks in Vietnam must comply with the terms in the *Central Bank* documents."
                    "The *Private Bank* documents contain the policies defined by one of the private banks in Vietnam."
                    "The task is to analyze whether the policies of the private bank are in full compliance with the Central Bank regulations.\n\n"
                    "Please perform the following and provide the details in VIETNAMESE:\n"
                    "* Give me the comparative result in a tabular format with the following columns: **Requirement**, **Private Bank Policy**, **Central Bank Regulation**, **Gap Category**, **Confidence Score**, **Reasoning** and make sure these attributes are in vietnamese\n"
                    "1. Categorize gaps between the documents into: **Missing**, **Partially compliant**, **Outdated**, and **Conflicting**.\n"
                    "2. For each Central Bank requirement, determine and label the gap category.\n"
                    "3. Where numerical thresholds or limits exist, compare them explicitly.\n"
                    "4. Generate **confidence scores (0–100)** indicating how certain you are about each categorization.\n"
                    "5. Provide detailed reasoning for each identified gap, explaining what is missing, conflicting, or needs update.\n\n"
                    f"Central Bank Documents Combined:\n{central_combined}\n\n"
                    f"Private Bank Documents Combined:\n{private_combined}\n\n"
                    "Return your results in a structured and easy-to-read format."
                )

                comparison_result = gemini_prompt(comparison_prompt)
                st.subheader("🔍 Compliance Gap Report")
                st.markdown(comparison_result)

                comparison_file_path = os.path.join("data", f"compliance_gap_analysis_{uuid.uuid4()}.txt")
                with open(comparison_file_path, "w", encoding="utf-8") as f:
                    f.write(comparison_result)
                with open(comparison_file_path, "rb") as f:
                    st.download_button("⬇️ Download Compliance Gap Report", f, file_name=os.path.basename(comparison_file_path), mime="text/plain")

                status_message.success("✅ Compliance gap analysis completed!")
            except Exception as e:
                status_message.error(f"❌ Comparison Error: {str(e)}")
else:
    st.info("Please upload at least one PDF for both Central Bank and Private Bank and ensure they are processed successfully to enable comparison.")