import streamlit as st
import os
import tempfile
from ingestion import ingest_document
from rag import answer_question

# ── Page config ──
st.set_page_config(
    page_title="Offline RAG — Private Document Assistant",
    page_icon="🔒",
    layout="wide"
)

# ── Header ──
st.title("🔒 Private Document Assistant")
st.caption("100% offline · Nothing leaves your device · Powered by Llama 3.2")

# ── Session state for chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# ── Sidebar ──
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Your file never leaves this device"
    )

    if uploaded_file is not None:
        if st.button("🔒 Ingest & Encrypt Document", type="primary"):
            with st.spinner("Reading, chunking, encrypting and embedding..."):
                # Save uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run ingestion pipeline
                ingest_document(temp_path)
                st.session_state.document_loaded = True

            st.success("✅ Document ready! Ask questions below.")
            st.info("🔒 Original file encrypted on disk")

    st.divider()

    # Status indicator
    if st.session_state.document_loaded:
        st.success("📚 Document loaded")
    else:
        st.warning("⚠️ No document loaded yet")

    st.divider()

    # Audit log viewer
    st.header("📋 Audit Log")
    if st.button("View Recent Queries"):
        if os.path.exists("rag_audit.log"):
            with open("rag_audit.log", "r") as f:
                log_content = f.read()
            st.text_area("Log", log_content[-2000:], height=200)
        else:
            st.info("No queries yet")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built by Vijay Gaddi · Artifact 1")

# ── Main chat area ──
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.document_loaded:
        st.error("⚠️ Please upload and ingest a document first using the sidebar!")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                answer = answer_question(prompt)
            st.markdown(answer)

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })