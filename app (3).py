
import streamlit as st
import fitz
import chromadb
from google import genai
import uuid

from sentence_transformers import SentenceTransformer
from datetime import datetime

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
# 2. CUSTOM CSS — SAME DARK BLUE DESIGN
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
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    margin: 22px 0 12px 0;
    color: #e2e8f0;
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
    margin-top: 0;
}

.info-card p {
    color: #cbd5e1;
    margin-bottom: 0;
    font-size: 14px;
}

.stButton > button {
    border-radius: 10px;
    min-height: 40px;
    font-weight: 600;
    border: 1px solid #354b70;
    transition: all .2s ease;
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

div[data-testid="stChatMessage"] {
    border: 1px solid #26374f;
    border-radius: 14px;
    padding: 12px;
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
# 3. EMBEDDING MODEL
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
    "client": None,
    "collection": None,
    "indexed": False,
    "document_names": [],
    "chunk_count": 0,
    "chat_history": [],
    "question_input": "",
    "pending_question": "",
    "last_answer": "",
    "last_sources": [],
    "last_question": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.client is None:
    st.session_state.client = chromadb.Client()

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
# 6. PDF EXTRACTION
# ==================================================
def process_pdfs(files):
    documents = []
    metadata = []

    for file in files:
        try:
            pdf = fitz.open(
                stream=file.getvalue(),
                filetype="pdf"
            )

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")

                for chunk in split_text(text):
                    documents.append(chunk)
                    metadata.append({
                        "source": file.name,
                        "page": page_number
                    })

            pdf.close()

        except Exception as error:
            st.warning(
                f"Could not process {file.name}: {error}"
            )

    return documents, metadata

# ==================================================
# 7. QUESTION PROMPTS
# ==================================================
def choose_prompt(question):
    st.session_state.pending_question = question
    st.session_state.question_input = question

# ==================================================
# 8. SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 📚 StudyMate AI")
    st.caption("Your personal academic study companion")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "AI model",
        ["Gemini Flash (Cloud)"],
        key="selected_model"
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Simple explanation",
            "Exam-oriented answer",
            "Detailed explanation",
            "Short answer",
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

    if st.button(
        "🗑️ Clear Documents and Chat",
        use_container_width=True
    ):
        old_collection = st.session_state.collection

        if old_collection is not None:
            st.session_state.client.delete_collection(
                old_collection.name
            )

        st.session_state.collection = None
        st.session_state.indexed = False
        st.session_state.document_names = []
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []
        st.session_state.last_question = ""
        st.session_state.question_input = ""
        st.session_state.pending_question = ""

        st.rerun()

    st.markdown("---")
    st.markdown("### ✨ Study tools")
    st.caption("Ask questions, revise concepts, and prepare for exams.")

    if st.session_state.document_names:
        st.markdown("**Indexed PDFs**")
        for name in st.session_state.document_names:
            st.caption(f"📄 {name}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🕘 Recent Questions")

        recent = st.session_state.chat_history[-5:][::-1]

        for index, item in enumerate(recent):
            label = item["question"]
            if len(label) > 42:
                label = label[:39] + "..."

            st.button(
                label,
                key=f"recent_{index}_{item['id']}",
                use_container_width=True,
                on_click=choose_prompt,
                args=(item["question"],)
            )

# ==================================================
# 9. MAIN WELCOME BANNER
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
        st.warning("Please upload at least one PDF first.")

    else:
        try:
            with st.spinner(
                "Reading PDFs and creating document embeddings..."
            ):
                docs, metadata = process_pdfs(uploaded_files)

                if not docs:
                    st.error(
                        "No readable text was found. "
                        "Scanned PDFs may require OCR."
                    )

                else:
                    client = st.session_state.client
                    old_collection = st.session_state.collection

                    if old_collection is not None:
                        client.delete_collection(
                            old_collection.name
                        )

                    collection = client.create_collection(
                        name="study_" + uuid.uuid4().hex,
                        metadata={"hnsw:space": "cosine"}
                    )

                    model = load_embedding_model()

                    for start in range(0, len(docs), 64):
                        batch_docs = docs[start:start + 64]
                        batch_meta = metadata[start:start + 64]

                        vectors = model.encode(
                            batch_docs,
                            normalize_embeddings=True
                        ).tolist()

                        collection.add(
                            ids=[
                                f"chunk_{i}"
                                for i in range(
                                    start,
                                    start + len(batch_docs)
                                )
                            ],
                            documents=batch_docs,
                            metadatas=batch_meta,
                            embeddings=vectors
                        )

                    st.session_state.collection = collection
                    st.session_state.indexed = True
                    st.session_state.document_names = sorted(
                        set(m["source"] for m in metadata)
                    )
                    st.session_state.chunk_count = len(docs)
                    st.session_state.chat_history = []
                    st.session_state.last_answer = ""
                    st.session_state.last_sources = []

                    st.success(
                        f"Successfully indexed "
                        f"{len(st.session_state.document_names)} PDF(s) "
                        f"into {len(docs)} text chunks."
                    )

        except Exception as error:
            st.error(f"Document processing failed: {error}")

# ==================================================
# 11. DASHBOARD METRICS
# ==================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📄 Documents",
        len(st.session_state.document_names)
    )

with col2:
    st.metric(
        "🧩 Text Chunks",
        st.session_state.chunk_count
    )

with col3:
    st.metric(
        "💬 Questions Asked",
        len(st.session_state.chat_history)
    )

# ==================================================
# 12. QUICK STUDY PROMPTS
# ==================================================
st.markdown(
    '<div class="section-title">✨ Quick Study Prompts</div>',
    unsafe_allow_html=True
)

st.caption(
    "Choose a prompt to fill the question box. "
    "You can edit it before asking."
)

prompt_cols = st.columns(3)

quick_prompts = [
    ("📖 Explain a concept", "Explain the main concepts in simple terms."),
    ("🎯 Exam key points", "Give the important exam points from the documents."),
    ("⚖️ Compare topics", "Compare the relevant topics found in the documents.")
]

for column, (label, prompt_text) in zip(
    prompt_cols, quick_prompts
):
    with column:
        st.button(
            label,
            key="quick_" + label,
            use_container_width=True,
            on_click=choose_prompt,
            args=(prompt_text,)
        )

# ==================================================
# 13. CHAT AND QUESTION ANSWERING
# ==================================================
st.markdown(
    '<div class="section-title">💬 Ask Your Study Materials</div>',
    unsafe_allow_html=True
)

if not st.session_state.indexed:
    st.info(
        "Upload your academic PDFs in the sidebar and click "
        "'Process Documents' to begin."
    )

    st.markdown("""
    <div class="info-card">
        <h4>🔎 Semantic Search</h4>
        <p>Find passages related to your question, even when
        the wording differs from your notes.</p>
    </div>
    <div class="info-card">
        <h4>🧠 AI-Powered Answers</h4>
        <p>Get explanations based on the retrieved academic
        content, with an option to use different answer styles.</p>
    </div>
    <div class="info-card">
        <h4>📌 Verifiable Sources</h4>
        <p>Inspect the original filename, page number, and
        retrieved passage.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show previous chat messages
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            for source in item["sources"]:
                with st.expander(
                    f"📄 {source['source']} — Page {source['page']}"
                ):
                    st.write(source["text"])
                    st.caption(
                        f"Cosine distance: {source['distance']:.3f}"
                    )

    # Question entry
    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="E.g. What are the challenges in traditional ML?"
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
                    collection = st.session_state.collection
                    model = load_embedding_model()

                    query_vector = model.encode(
                        [question],
                        normalize_embeddings=True
                    ).tolist()

                    results = collection.query(
                        query_embeddings=query_vector,
                        n_results=min(4, collection.count()),
                        include=[
                            "documents",
                            "metadatas",
                            "distances"
                        ]
                    )

                    retrieved_docs = results["documents"][0]
                    retrieved_meta = results["metadatas"][0]
                    retrieved_distances = results["distances"][0]

                    sources = []

                    for doc, meta, distance in zip(
                        retrieved_docs,
                        retrieved_meta,
                        retrieved_distances
                    ):
                        sources.append({
                            "source": meta["source"],
                            "page": meta["page"],
                            "text": doc,
                            "distance": float(distance)
                        })

                    # Do not discard all retrieved passages using
                    # an arbitrary similarity threshold.
                    context = "\n\n".join(
                        f"Source: {source['source']}, "
                        f"Page: {source['page']}\n"
                        f"{source['text']}"
                        for source in sources
                    )

                    style_instructions = {
                        "Simple explanation":
                            "Explain in simple student-friendly language.",
                        "Exam-oriented answer":
                            "Write an exam-ready answer with a definition "
                            "and relevant points from the context.",
                        "Detailed explanation":
                            "Explain the topic in detail using the context.",
                        "Short answer":
                            "Answer concisely in a few sentences.",
                        "Bullet points":
                            "Present the answer as clear bullet points."
                    }

                    prompt = f"""
You are StudyMate AI, an academic assistant.

Answer the student's question using the supplied document context.

Rules:
1. Read all supplied context carefully.
2. Answer directly when the information is present.
3. If asked to list items, include all relevant items present
   in the context.
4. Do not claim that information is missing if it is clearly present.
5. Do not invent facts or add unsupported information.
6. If the answer genuinely is not supported by the context,
   say that you could not find it in the uploaded documents.
7. Use this response style: {style_instructions[answer_style]}
8. Do not invent source citations. The application displays
   retrieved source filenames and page numbers separately.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

                    # Generate the answer using Gemini (cloud API).
                    # GEMINI_API_KEY is stored securely in Streamlit Secrets.
                    client = genai.Client(
                        api_key=st.secrets["GEMINI_API_KEY"]
                    )

                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )

                    answer = (response.text or "").strip()

                    if not answer:
                        st.error("The model returned an empty answer.")

                    else:
                        chat_item = {
                            "id": uuid.uuid4().hex,
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                            "time": datetime.now().strftime("%H:%M")
                        }

                        st.session_state.chat_history.append(chat_item)
                        st.session_state.last_answer = answer
                        st.session_state.last_sources = sources
                        st.session_state.last_question = question

                        st.rerun()

            except Exception as error:
                st.error(f"An unexpected error occurred: {error}")

# ==================================================
# 14. LATEST ANSWER: COPY AND DOWNLOAD
# ==================================================
if st.session_state.last_answer:
    st.markdown("---")
    st.markdown(
        '<div class="section-title">📝 Save Your Answer</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Your latest answer is available below for copying or revision."
    )

    # st.code includes a built-in copy control in Streamlit.
    st.code(
        st.session_state.last_answer,
        language="text"
    )

    download_text = (
        f"Question: {st.session_state.last_question}\n\n"
        f"Answer:\n{st.session_state.last_answer}\n\n"
        f"Sources:\n"
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
            f"📄 {source['source']} — Page {source['page']}"
        ):
            st.write(source["text"])
            st.caption(
                f"Cosine distance: {source['distance']:.3f}"
            )

# ==================================================
# 15. FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    StudyMate AI · Retrieval-Augmented Generation (RAG)<br>
    Built with Python, Streamlit, ChromaDB, Sentence Transformers
    and Gemini.
</div>
""", unsafe_allow_html=True)