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
                        model="gemini-2.5-flash",
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
