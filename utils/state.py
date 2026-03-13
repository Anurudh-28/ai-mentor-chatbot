from __future__ import annotations

import streamlit as st


def get_chat_messages_state(key: str = "messages") -> list[dict[str, str]]:
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def reset_chat(key: str = "messages") -> None:
    st.session_state[key] = []

