from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Provider = Literal["openai", "groq", "gemini"]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LLMError(message)


def chat(
    *,
    provider: Provider,
    model: str,
    api_key: str | None,
    messages: list[ChatMessage],
    temperature: float = 0.2,
) -> str:
    provider = provider.strip().lower()  # type: ignore[assignment]
    if provider == "openai":
        return _chat_openai(model=model, api_key=api_key, messages=messages, temperature=temperature)
    if provider == "groq":
        return _chat_groq(model=model, api_key=api_key, messages=messages, temperature=temperature)
    if provider == "gemini":
        return _chat_gemini(model=model, api_key=api_key, messages=messages, temperature=temperature)
    raise LLMError(f"Unknown provider: {provider!r}. Use openai, groq, or gemini.")


def _chat_openai(*, model: str, api_key: str | None, messages: list[ChatMessage], temperature: float) -> str:
    _require(bool(api_key), "Missing OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise LLMError("OpenAI package missing. Install: pip install openai") from e

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _chat_groq(*, model: str, api_key: str | None, messages: list[ChatMessage], temperature: float) -> str:
    _require(bool(api_key), "Missing GROQ_API_KEY.")
    try:
        from groq import Groq
    except Exception as e:  # pragma: no cover
        raise LLMError("Groq package missing. Install: pip install groq") from e

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        temperature=temperature,
    )
    # groq response shape mirrors OpenAI
    return (resp.choices[0].message.content or "").strip()


def _chat_gemini(*, model: str, api_key: str | None, messages: list[ChatMessage], temperature: float) -> str:
    _require(bool(api_key), "Missing GEMINI_API_KEY.")
    try:
        import google.generativeai as genai
    except Exception as e:  # pragma: no cover
        raise LLMError("Gemini package missing. Install: pip install google-generativeai") from e

    genai.configure(api_key=api_key)

    system_parts: list[str] = []
    chat_parts: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            chat_parts.append({"role": "user", "parts": [m.content]})
        elif m.role == "assistant":
            chat_parts.append({"role": "model", "parts": [m.content]})

    system_instruction = "\n\n".join(system_parts).strip() or None
    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config={"temperature": temperature},
    )
    resp = gm.generate_content(chat_parts)
    text = getattr(resp, "text", None)
    return (text or "").strip()

