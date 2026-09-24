# Data Dashboard Connector (Expt 2)

Connects an LLM (via Groq) to a live weather API (wttr.in) through a single
tool, `get_current_weather(location)`. The Streamlit UI shows the LLM's
tool-use decision, the tool's raw result, and the final summarized answer.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env   # then add your GROQ_API_KEY
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys.

## How it works

- `weather_tool.py` — the "MCP server" side: defines the tool's schema
  (name/description/parameters) and its implementation, which fetches
  structured JSON from `wttr.in`.
- `app.py` — the client: sends the user's message plus the tool definition
  to Groq's chat-completions API, runs the tool-call loop (model decides to
  call the tool → we execute it → feed the result back → model summarizes),
  and renders each step in the chat UI.
