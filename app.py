from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import get_config
from models import ChatMessage, LLMError, chat

from utils import (
    build_rag_index,
    format_context,
    format_web_context,
    get_chat_messages_state,
    load_local_documents,
    reset_chat,
    retrieve,
    search_web,
    should_web_search,
)


APP_TITLE = "AI Career Mentor Chatbot"


def instructions_page(cfg) -> None:
    st.title("Use Case: The Chatbot Blueprint")
    st.markdown(
        """
## Installation

```bash
pip install -r requirements.txt
```

## API keys

- **OpenAI**: set `OPENAI_API_KEY`
- **Groq**: set `GROQ_API_KEY`
- **Gemini**: set `GEMINI_API_KEY`

Optional:
- **Web search (Serper)**: set `SERPER_API_KEY` (otherwise DuckDuckGo fallback is used)
- **Docs folder**: set `DOCS_PATH` (default: `docs/`)
"""
    )

    st.markdown("## Tasks to complete")
    st.markdown(
        """
1. **RAG Integration**: local documents, vector retrieval, chunking.
2. **Live Web Search Integration**: real-time results as a tool.
3. **Response Modes**: concise vs detailed responses.
"""
    )

    st.markdown("## RAG (local documents)")
    st.write(f"Docs folder: `{cfg.docs_path}`")
    st.markdown(
        """
Supported file types:
- `.txt`, `.md`, `.pdf`

Drop files into your docs folder and use the sidebar button to (re)index them.
"""
    )


def _select_provider(cfg) -> tuple[str, str, str | None]:
    provider = cfg.llm_provider.strip().lower()
    if provider == "openai":
        return "openai", cfg.openai_model, cfg.openai_api_key
    if provider == "groq":
        return "groq", cfg.groq_model, cfg.groq_api_key
    if provider == "gemini":
        return "gemini", cfg.gemini_model, cfg.gemini_api_key
    return "openai", cfg.openai_model, cfg.openai_api_key


def chat_page(cfg) -> None:
    st.title("🤖 AI Career Mentor")
    st.caption("Your personal mentor for AI, Machine Learning, and Data Science careers.")     

    provider, model, api_key = _select_provider(cfg)

    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
    if "rag_docs_count" not in st.session_state:
        st.session_state.rag_docs_count = 0

    with st.sidebar:
        st.subheader("Controls")
        response_mode = st.radio("Response mode", options=["Concise", "Detailed"], index=0)
        use_rag = st.toggle("Use local docs (RAG)", value=True)
        use_web = st.toggle("Enable web search tool", value=True)
        force_web = st.toggle("Always web search", value=False, disabled=not use_web)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        system_prompt = st.text_area(
    "System prompt",
    value="""
You are an AI Career Mentor helping students build successful careers in Artificial Intelligence,
Machine Learning, and Data Science.

Your role is to:
• Guide students on AI/ML learning roadmaps
• Suggest projects for AI portfolios
• Help with ML interview preparation
• Explain AI concepts clearly
• Recommend resources and skills required for AI jobs

Give practical, motivating, and clear advice.
""",
    height=160,
)

        st.divider()
        st.caption(f"Provider: `{provider}`  |  Model: `{model}`")
        if not api_key:
            st.warning("Missing API key for the selected provider.")

        st.divider()
        st.subheader("RAG indexing")
        st.caption(f"Docs path: `{cfg.docs_path}`")
        st.caption(f"Embeddings backend: `{cfg.embeddings_backend}`")
        if st.button("Build / refresh index", use_container_width=True):
            with st.spinner("Indexing local documents..."):
                loaded = load_local_documents(cfg.docs_path)
                docs = [(d.doc_id, d.source, d.text) for d in loaded]
                st.session_state.rag_index = build_rag_index(
                    docs=docs,
                    embeddings_backend=cfg.embeddings_backend,
                    openai_api_key=cfg.openai_api_key,
                    openai_embeddings_model=cfg.openai_embeddings_model,
                )
                st.session_state.rag_docs_count = len(loaded)
            st.success(f"Indexed {len(loaded)} document(s).")

        if st.button("Reset chat", use_container_width=True):
            reset_chat()
            st.rerun()

    history = get_chat_messages_state()
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your message here…")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            context_parts: list[str] = []

            if use_rag and st.session_state.rag_index is not None:
                hits = retrieve(
                    index=st.session_state.rag_index,
                    query=prompt,
                    k=4,
                    embeddings_backend=cfg.embeddings_backend,
                    openai_api_key=cfg.openai_api_key,
                    openai_embeddings_model=cfg.openai_embeddings_model,
                )
                ctx = format_context(hits)
                if ctx:
                    context_parts.append(ctx)

            if use_web and (force_web or should_web_search(prompt)):
                with st.spinner("Searching the web..."):
                    results = search_web(prompt, serper_api_key=cfg.serper_api_key, num_results=5)
                wctx = format_web_context(results)
                if wctx:
                    context_parts.append(wctx)

            mode_instruction = (
                "Answer briefly and directly. Use short bullets when helpful."
                if response_mode == "Concise"
                else "Answer with more detail, including rationale and step-by-step guidance when useful."
            )

            messages: list[ChatMessage] = []
            sys_parts = [system_prompt.strip(), mode_instruction]
            if context_parts:
                sys_parts.append("\n\n".join(context_parts))
            sys_text = "\n\n".join([p for p in sys_parts if p]).strip()
            if sys_text:
                messages.append(ChatMessage(role="system", content=sys_text))

            for m in history:
                if m["role"] in ("user", "assistant"):
                    messages.append(ChatMessage(role=m["role"], content=m["content"]))

            answer = chat(
                provider=provider,  # type: ignore[arg-type]
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
            ).strip()
            if not answer:
                answer = "(No response.)"
            placeholder.markdown(answer)
            history.append({"role": "assistant", "content": answer})
        except LLMError as e:
            err = f"**LLM error:** {e}"
            placeholder.markdown(err)
            history.append({"role": "assistant", "content": err})
        except Exception as e:
            err = f"**Error:** {e}"
            placeholder.markdown(err)
            history.append({"role": "assistant", "content": err})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide", initial_sidebar_state="expanded")
    cfg = get_config()

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

    if page == "Instructions":
        instructions_page(cfg)
    else:
        chat_page(cfg)


if __name__ == "__main__":
    main()