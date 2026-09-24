"""
Data Dashboard Connector — Expt 2

A Streamlit client that talks to Groq's LLM API. The LLM has one tool,
get_current_weather (backed by wttr.in), and the UI makes the client/server
interaction visible: you can see the model decide to call the tool, the
tool's raw result, and the model's final summarized answer.
"""

import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from weather_tool import GET_CURRENT_WEATHER_TOOL, get_current_weather

load_dotenv()

TOOLS = [GET_CURRENT_WEATHER_TOOL]
TOOL_IMPLS = {
    "get_current_weather": lambda args: get_current_weather(args["location"]),
}

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a get_current_weather tool "
    "that fetches real-time data. Use it whenever the user asks about "
    "current weather conditions anywhere. After getting the tool result, "
    "summarize it for the user in a friendly, concise way."
)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

st.set_page_config(page_title="Data Dashboard Connector", page_icon="🌤️", layout="centered")


def render_trace(trace, animate=False):
    """Render one assistant turn's trace: thoughts, tool calls, tool
    results, and the final answer, in order."""
    for step in trace:
        kind = step["type"]

        if kind == "thought" and step["text"].strip():
            st.markdown(f"🤔 _{step['text']}_")

        elif kind == "tool_call":
            with st.status(f"🔧 Calling tool: `{step['tool']}`", state="complete", expanded=True):
                st.code(json.dumps(step["input"], indent=2), language="json")

        elif kind == "tool_result":
            icon = "❌" if step.get("isError") else "📦"
            with st.expander(f"{icon} Tool result: `{step['tool']}`", expanded=True):
                st.json(step["output"])

        elif kind == "final_answer" and step["text"].strip():
            st.markdown(step["text"])

        if animate:
            time.sleep(0.35)


def run_agent_turn(client, model, user_message):
    """Runs the tool-use loop for one user message and returns the trace."""
    st.session_state.conversation.append({"role": "user", "content": user_message})
    trace = []

    for _ in range(4):  # cap on tool round-trips to avoid runaway loops
        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.conversation,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )
        msg = response.choices[0].message

        if msg.content and msg.content.strip():
            trace.append({"type": "thought", "text": msg.content})

        tool_calls = msg.tool_calls or []

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        st.session_state.conversation.append(assistant_msg)

        if not tool_calls:
            trace.append({"type": "final_answer", "text": msg.content or ""})
            return trace

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            trace.append({"type": "tool_call", "tool": tc.function.name, "input": args})

            is_error = False
            try:
                impl = TOOL_IMPLS.get(tc.function.name)
                if impl is None:
                    raise ValueError(f"Unknown tool: {tc.function.name}")
                output = impl(args)
            except Exception as exc:  # noqa: BLE001 - surface any tool failure to the UI
                is_error = True
                output = {"error": str(exc)}

            trace.append(
                {"type": "tool_result", "tool": tc.function.name, "output": output, "isError": is_error}
            )

            st.session_state.conversation.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output)}
            )

    trace.append({"type": "final_answer", "text": "(Stopped after too many tool-use turns.)"})
    return trace


st.title("🌤️ Data Dashboard Connector")
st.caption(
    "Ask about the weather anywhere. Watch the AI decide to call a live "
    "data tool, fetch the result, and summarize it — the client/server "
    "interaction, made visible."
)

with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input(
        "Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password"
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("Get a free key at [console.groq.com/keys](https://console.groq.com/keys)")
    if st.button("Clear conversation"):
        st.session_state.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display_history = []
        st.rerun()

if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []

for turn in st.session_state.display_history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.write(turn["content"])
    else:
        with st.chat_message("assistant"):
            render_trace(turn["trace"])

user_input = st.chat_input("What's the weather in Tokyo?")

if user_input:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            client = Groq(api_key=api_key)
            with st.spinner("Thinking..."):
                trace = run_agent_turn(client, model, user_input)
            render_trace(trace, animate=True)
        except Exception as exc:  # noqa: BLE001 - show API/network errors in the UI
            st.error(f"Something went wrong: {exc}")
            trace = [{"type": "final_answer", "text": f"Error: {exc}"}]

    st.session_state.display_history.append({"role": "assistant", "trace": trace})
