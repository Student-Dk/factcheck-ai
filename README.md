# 🔍 FactCheck AI

An AI-powered fact-checking web app that automatically verifies claims in PDF documents against live web data.

## What it does

1. **Upload** any PDF document
2. **Extracts** all specific claims (stats, dates, financial figures, technical facts)
3. **Searches** the live web to verify each claim (via Tavily API)
4. **Reports** each claim as `Verified`, `Inaccurate`, or `False` with sources

## Tech Stack

- **Frontend + Backend:** Streamlit (Python)
- **AI/LLM:** Google Gemini 1.5 Flash
- **Web Search:** Tavily API
- **PDF Parsing:** PyMuPDF

## Setup Locally

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud).

API keys are stored as Streamlit Secrets (not in `.env`) for production.
