
import streamlit as st
import pymupdf
import numpy as np
import uuid
from google import genai
from sentence_transformers import SentenceTransformer

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="StudyMate AI | Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. CUSTOM DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1020, #111827, #172033);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca, #2563eb, #0891b2);
    margin-bottom: 24px;
    box-shadow: 0 10px 35px rgba(37, 99, 235, .20);
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
.info-card {
    background: #182235;
    border: 1px solid #2d3b52;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}
.info-card h4 {
    color: #93c5fd;
}
.info-card p {
    color: #cbd5e1;
    font-size: 14px;
}
.stButton > button {
    border-radius: 10px;
    min-height: 40px;
    font-weight: 600;
    border: 1px solid #354b70;
}
.stButton > button:hover {
    border-color: #67e8f9;
    color: #67e8f9;
}
div[data-testid="stMetric"] {
    background: #182235;
    border: 1px solid #2d3b52;
    padding: 15px;
    border-radius: 14px;
}
[data-testid="stExpander"] {
    background: #151f30;
    border: 1px solid #344258;
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding: 25px 0 8px 0;
}
#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. LOAD EMBEDDING MODEL
# ==================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================================
# 4. SESSION STATE
# ==================================================
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


# ==================================================
# 5. TEXT CHUNKING
# ==================================================
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


# ==================================================
# 6. EXTRACT TEXT FROM PDF FILES
# ==================================================
def process_pdfs(files):
    documents = []

    for file in files:
        try:
            pdf = pymupdf.open(
                stream=file.getvalue(),
                filetype="pdf"
            )

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                for chunk in split_text(page_text):
                    documents.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_number
                    })

            pdf.close()

        except Exception as error:
            st.warning(
                f"Could not process {file.name}: {error}"
            )

    return documents


# ==================================================
# 7. GEMINI MODEL FALLBACK
# ==================================================
def generate_gemini_answer(prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it under "
            "Streamlit Cloud Settings > Secrets."
        )

    client = genai.Client(api_key=api_key)

    # Preferred models are tried first if available.
    preferred_models = [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]

    available_models = []

    # Ask the API which models support content generation.
    try:
        for model_info in client.models.list():
            name = (model_info.name or "").replace("models/", "")
            actions = getattr(
                model_info, "supported_actions", []
            ) or []

            if (
                "generateContent" in actions
                and "flash" in name.lower()
            ):
                available_models.append(name)

    except Exception:
        # If model listing fails, try the known candidates below.
        pass

    # Prefer models that the API actually listed.
    candidates = [
        name for name in preferred_models
        if name in available_models
    ]

    # If listing did not return the preferred names, still try them.
    if not candidates:
        candidates = preferred_models.copy()

    # Add other available Flash models as additional fallbacks.
    for name in available_models:
        if name not in candidates:
            candidates.append(name)

    errors = []

    for model_name in candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            answer = (response.text or "").strip()

            if answer:
                return answer

            errors.append(
                f"{model_name}: empty response"
            )

        except Exception as error:
            errors.append(
                f"{model_name}: {str(error)}"
            )

    details = "\n".join(errors[-5:])

    raise RuntimeError(
        "No attempted Gemini model returned an answer. "
        "The models may be temporarily unavailable, "
        "unsupported by your API key, or rate-limited.\n"
        + details
    )


# ==================================================
# 8. SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 📚 StudyMate AI")
    st.caption("Your personal academic study companion")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

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

    st.markdown("---")
    st.markdown("### 📄 Your Documents")

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

    if st.session_state.document_names:
        st.markdown("---")
        st.markdown("**Indexed PDFs**")

        for name in st.session_state.document_names:
            st.caption(f"📄 {name}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🕘 Recent Questions")

        for item in reversed(
            st.session_state.chat_history[-5:]
        ):
            st.caption(item["question"])


# ==================================================
# 9. WELCOME BANNER
# ==================================================
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


# ==================================================
# 10. PROCESS DOCUMENTS
# ==================================================
if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        try:
            with st.spinner(
                "Reading PDFs and creating document embeddings..."
            ):
                documents = process_pdfs(uploaded_files)

                if not documents:
                    st.error(
                        "No readable text was found. "
                        "Scanned PDFs may require OCR."
                    )

                else:
                    embedding_model = load_embedding_model()

                    texts = [
                        item["text"] for item in documents
                    ]

                    # Process in batches to reduce memory use.
                    all_embeddings = []

                    for start in range(0, len(texts), 64):
                        batch = texts[start:start + 64]

                        batch_embeddings = embedding_model.encode(
                            batch,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        all_embeddings.extend(batch_embeddings)

                    for index, item in enumerate(documents):
                        item["embedding"] = np.asarray(
                            all_embeddings[index],
                            dtype=np.float32
                        )

                    # Store indexed chunks in Streamlit session memory.
                    st.session_state.chunks = documents
                    st.session_state.indexed = True

                    st.session_state.document_names = sorted(
                        set(
                            item["source"]
                            for item in documents
                        )
                    )

                    st.session_state.chunk_count = len(documents)
                    st.session_state.chat_history = []
                    st.session_state.last_answer = ""
                    st.session_state.last_sources = []
                    st.session_state.last_question = ""

                    st.success(
                        f"Successfully indexed "
                        f"{len(st.session_state.document_names)} PDF(s) "
                        f"into {len(documents)} text chunks."
                    )

        except Exception as error:
            st.error(
                f"Document processing failed: {error}"
            )


# ==================================================
# 11. DASHBOARD METRICS
# ==================================================
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


# ==================================================
# 12. QUICK STUDY PROMPTS
# ==================================================
st.markdown("## ✨ Quick Study Prompts")

st.caption(
    "Choose a prompt to use it as your question."
)

prompt_cols = st.columns(3)

quick_prompts = [
    (
        "📖 Explain a concept",
        "Explain the main concepts in simple terms."
    ),
    (
        "🎯 Exam key points",
        "Give the important exam points from the documents."
    ),
    (
        "⚖️ Compare topics",
        "Compare the relevant topics found in the documents."
    )
]

for column, (label, prompt_text) in zip(
    prompt_cols, quick_prompts
):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state["suggested_question"] = prompt_text


# ==================================================
# 13. CHAT AND QUESTION ANSWERING
# ==================================================
st.markdown("## 💬 Ask Your Study Materials")

if not st.session_state.indexed:
    st.info(
        "Upload your academic PDFs in the sidebar and click "
        "'Process Documents' to begin."
    )

    st.markdown("""
    <div class="info-card">
        <h4>🔎 Semantic Search</h4>
        <p>
            Find passages related to your question, even when
            the wording differs from your notes.
        </p>
    </div>
    <div class="info-card">
        <h4>🧠 AI-Powered Answers</h4>
        <p>
            Get explanations based on retrieved academic content.
        </p>
    </div>
    <div class="info-card">
        <h4>📌 Verifiable Sources</h4>
        <p>
            Inspect the original filename, page number, and
            retrieved passage.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Display previous messages.
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            for source in item["sources"]:
                with st.expander(
                    f"{source['source']} — Page {source['page']}"
                ):
                    st.write(source["text"])
                    st.caption(
                        f"Cosine similarity: "
                        f"{source['similarity']:.3f}"
                    )

    default_question = st.session_state.pop(
        "suggested_question", ""
    )

    question = st.text_input(
        "Your question",
        value=default_question,
        key="current_question",
        placeholder="Example: Explain the challenges in traditional ML."
    )

    if st.button(
        "🚀 Get Answer",
        type="primary",
        use_container_width=True
    ):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching your notes and generating an answer..."
                ):
                    embedding_model = load_embedding_model()

                    query_embedding = embedding_model.encode(
                        [question],
                        normalize_embeddings=True
                    )[0]

                    # Retrieve the four most similar chunks.
                    similarities = []

                    for item in st.session_state.chunks:
                        score = float(
                            np.dot(
                                query_embedding,
                                item["embedding"]
                            )
                        )

                        similarities.append((score, item))

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
                        f"Page: {item['page']}\n"
                        f"{item['text']}"
                        for item in sources
                    )

                    style_instructions = {
                        "Short answer":
                            "Answer concisely in a few sentences.",

                        "Simple explanation":
                            "Explain in simple, student-friendly language.",

                        "Exam-oriented answer":
                            "Give an exam-ready answer with a definition "
                            "and relevant points from the context.",

                        "Detailed explanation":
                            "Explain the topic in detail using the context.",

                        "Bullet points":
                            "Present the answer as clear bullet points."
                    }

                    prompt = f"""
You are StudyMate AI, an academic assistant.

Answer the student's question using the supplied document context.

Rules:
1. Use the supplied context as your main source.
2. Answer directly when the information is present.
3. Do not invent facts or unsupported information.
4. If the answer is not supported by the context, say so.
5. Follow this response style:
   {style_instructions[answer_style]}
6. Do not invent source citations. The application displays
   filenames and page numbers separately.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

                    # Try available Gemini Flash models in sequence.
                    answer = generate_gemini_answer(prompt)

                    chat_item = {
                        "id": uuid.uuid4().hex,
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }

                    st.session_state.chat_history.append(chat_item)
                    st.session_state.last_answer = answer
                    st.session_state.last_sources = sources
                    st.session_state.last_question = question

                    st.rerun()

            except Exception as error:
                st.error(
                    f"Could not generate the answer: {error}"
                )


# ==================================================
# 14. LATEST ANSWER: COPY AND DOWNLOAD
# ==================================================
if st.session_state.last_answer:
    st.markdown("---")
    st.markdown("## 📝 Save Your Answer")

    st.code(
        st.session_state.last_answer,
        language="text"
    )

    download_text = (
        f"Question: {st.session_state.last_question}\n\n"
        f"Answer:\n{st.session_state.last_answer}\n\n"
        "Sources:\n"
    )

    for source in st.session_state.last_sources:
        download_text += (
            f"- {source['source']}, Page {source['page']}\n"
        )

    st.download_button(
        "⬇️ Download Answer as TXT",
        data=download_text,
        file_name="studymate_answer.txt",
        mime="text/plain",
        use_container_width=True
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


# ==================================================
# 15. FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    StudyMate AI · Retrieval-Augmented Generation (RAG)<br>
    Built with Python, Streamlit, Sentence Transformers,
    NumPy and Gemini.
</div>
""", unsafe_allow_html=True)
