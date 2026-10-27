import streamlit as st
import fitz
import numpy as np
import uuid
from google import genai
from sentence_transformers import SentenceTransformer

# =========================
# 1. PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

# =========================
# 2. CUSTOM DESIGN
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1020, #111827, #172033);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca, #2563eb, #0891b2);
    margin-bottom: 24px;
}
.hero h1, .hero p {
    color: white;
}
div[data-testid="stMetric"] {
    background: #182235;
    border: 1px solid #2d3b52;
    padding: 15px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 3. LOAD EMBEDDING MODEL
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =========================
# 4. SESSION STATE
# =========================
defaults = {
    "chunks": [],
    "indexed": False,
    "document_names": [],
    "chunk_count": 0,
    "chat_history": [],
    "last_answer": "",
    "last_sources": [],
    "last_question": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================
# 5. TEXT CHUNKING
# =========================
def split_text(text, size=1000, overlap=150):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# =========================
# 6. READ PDF FILES
# =========================
def process_pdfs(files):
    documents = []

    for file in files:
        try:
            pdf = fitz.open(
                stream=file.getvalue(),
                filetype="pdf"
            )

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")

                for chunk in split_text(text):
                    documents.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_number
                    })

            pdf.close()

        except Exception as error:
            st.warning(f"Could not read {file.name}: {error}")

    return documents


# =========================
# 7. SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📚 StudyMate AI")
    st.caption("Your academic study companion")

    answer_style = st.selectbox(
        "Answer style",
        [
            "Short answer",
            "Simple explanation",
            "Exam-oriented answer",
            "Detailed explanation",
            "Bullet points"
        ]
    )

    uploaded_files = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_clicked = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True
    )

    clear_clicked = st.button(
        "🗑️ Clear Documents and Chat",
        use_container_width=True
    )

    if clear_clicked:
        st.session_state.chunks = []
        st.session_state.indexed = False
        st.session_state.document_names = []
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []
        st.session_state.last_question = ""
        st.rerun()


# =========================
# 8. WELCOME BANNER
# =========================
st.markdown("""
<div class="hero">
    <h1>📚 StudyMate AI</h1>
    <p>
        Your intelligent academic companion.
        Upload your notes, ask questions, and learn from
        answers grounded in your own study material.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# 9. PROCESS DOCUMENTS
# =========================
if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        try:
            with st.spinner("Reading PDFs and creating embeddings..."):
                documents = process_pdfs(uploaded_files)

                if not documents:
                    st.error(
                        "No readable text was found in the PDFs. "
                        "Scanned PDFs may require OCR."
                    )

                else:
                    model = load_embedding_model()

                    texts = [item["text"] for item in documents]

                    embeddings = model.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    # Store text and embeddings in session memory.
                    # No ChromaDB or database is required.
                    for i, item in enumerate(documents):
                        item["embedding"] = np.asarray(
                            embeddings[i], dtype=np.float32
                        )

                    st.session_state.chunks = documents
                    st.session_state.indexed = True
                    st.session_state.document_names = sorted(
                        set(item["source"] for item in documents)
                    )
                    st.session_state.chunk_count = len(documents)
                    st.session_state.chat_history = []
                    st.session_state.last_answer = ""
                    st.session_state.last_sources = []
                    st.session_state.last_question = ""

                    st.success(
                        f"Successfully processed "
                        f"{len(st.session_state.document_names)} PDF(s) "
                        f"into {len(documents)} text chunks."
                    )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


# =========================
# 10. DASHBOARD
# =========================
col1, col2, col3 = st.columns(3)

col1.metric(
    "📄 Documents",
    len(st.session_state.document_names)
)

col2.metric(
    "🧩 Text Chunks",
    st.session_state.chunk_count
)

col3.metric(
    "💬 Questions Asked",
    len(st.session_state.chat_history)
)

if st.session_state.document_names:
    st.caption(
        "Processed PDFs: "
        + ", ".join(st.session_state.document_names)
    )


# =========================
# 11. QUESTION ANSWERING
# =========================
st.markdown("## 💬 Ask Your Study Materials")

if not st.session_state.indexed:
    st.info(
        "Upload your academic PDFs and click "
        "'Process Documents' to begin."
    )

else:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

    question = st.text_input(
        "Your question",
        placeholder="Example: Explain the main concepts in my notes."
    )

    if st.button("🚀 Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Searching your notes and generating an answer..."):
                    model = load_embedding_model()

                    query_embedding = model.encode(
                        [question],
                        normalize_embeddings=True
                    )[0]

                    # Cosine similarity: embeddings are normalized.
                    similarities = []

                    for item in st.session_state.chunks:
                        score = float(
                            np.dot(query_embedding, item["embedding"])
                        )
                        similarities.append((score, item))

                    # Retrieve the four most relevant chunks.
                    similarities.sort(
                        key=lambda pair: pair[0],
                        reverse=True
                    )

                    top_results = similarities[:4]

                    sources = []

                    for score, item in top_results:
                        sources.append({
                            "source": item["source"],
                            "page": item["page"],
                            "text": item["text"],
                            "similarity": score
                        })

                    context = "\n\n".join(
                        f"Source: {item['source']}, "
                        f"Page: {item['page']}\n{item['text']}"
                        for item in sources
                    )

                    instructions = {
                        "Short answer":
                            "Answer concisely in a few sentences.",
                        "Simple explanation":
                            "Explain in simple, student-friendly language.",
                        "Exam-oriented answer":
                            "Give an exam-ready answer with a definition "
                            "and relevant points.",
                        "Detailed explanation":
                            "Explain the topic in detail using the context.",
                        "Bullet points":
                            "Present the answer as clear bullet points."
                    }

                    prompt = f"""
You are StudyMate AI, an academic assistant.

Answer the student's question using the document context below.

Rules:
1. Use the supplied context as the main source of information.
2. Do not invent facts or claim unsupported information is in the PDF.
3. If the context does not contain the answer, say so clearly.
4. Follow this answer style: {instructions[answer_style]}
5. Do not invent citations.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

                    # Gemini API key must be configured in Streamlit Secrets.
                    client = genai.Client(
                        api_key=st.secrets["GEMINI_API_KEY"]
                    )
                    preferred_models = [
                        "gemini-3.8-flash",
                        "gemini-2.5-flash",
                        "gemini-2.0-flash"
                    ]
                    answer = ""
                    errors = []
                    for model_name in preferred_models:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            answer = (response.text or "").strip()
                            if answer:
                                break
                        except Exception as model_error:
                            errors.append(f"{model_name}: {model_error}")
                            if not answer:
                                st.error(
                                    "Gemini models are temporarily unavailable. "
                                    "Please try again later. Details: " + " | ".join(errors)
                                )
                                st.stop()

                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )

                    answer = (response.text or "").strip()

                    if not answer:
                        st.error("The AI model returned an empty answer.")

                    else:
                        st.session_state.chat_history.append({
                            "id": uuid.uuid4().hex,
                            "question": question,
                            "answer": answer,
                            "sources": sources
                        })

                        st.session_state.last_answer = answer
                        st.session_state.last_sources = sources
                        st.session_state.last_question = question

                        st.rerun()

            except Exception as error:
                st.error(f"Could not generate the answer: {error}")


# =========================
# 12. DISPLAY SOURCES AND DOWNLOAD
# =========================
if st.session_state.last_answer:
    st.markdown("---")
    st.markdown("## 📝 Latest Answer")

    st.code(
        st.session_state.last_answer,
        language="text"
    )

    st.download_button(
        "⬇️ Download Answer as TXT",
        data=(
            f"Question: {st.session_state.last_question}\n\n"
            f"Answer:\n{st.session_state.last_answer}\n\n"
            "Sources:\n"
            + "\n".join(
                f"- {source['source']}, Page {source['page']}"
                for source in st.session_state.last_sources
            )
        ),
        file_name="studymate_answer.txt",
        mime="text/plain"
    )

    st.markdown("### 📚 Retrieved Sources")

    for source in st.session_state.last_sources:
        with st.expander(
            f"{source['source']} — Page {source['page']}"
        ):
            st.write(source["text"])
            st.caption(
                f"Cosine similarity: {source['similarity']:.3f}"
            )


# =========================
# 13. FOOTER
# =========================
st.markdown("---")
st.caption(
    "StudyMate AI | RAG Academic Assistant | "
    "Python, Streamlit, Sentence Transformers and Gemini"
)
