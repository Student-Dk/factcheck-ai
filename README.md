# FactCheck Pro

AI-powered fact-checking web application that automatically verifies claims from PDF documents using live web search and Large Language Models (LLMs).

## Overview

FactCheck Pro helps users identify inaccurate, outdated, or false information in documents. Users can upload a PDF, and the application extracts factual claims, verifies them against live web sources, and generates a structured verification report.

This project was developed as part of the Product Management Trainee Assessment.

---

## Features

### PDF Claim Extraction

* Upload PDF documents through a simple web interface.
* Automatically extracts important factual claims, including:

  * Statistics
  * Dates
  * Financial figures
  * Technical statements

### Live Web Verification

* Searches the web in real time using Tavily Search API.
* Retrieves relevant evidence from multiple online sources.

### Intelligent Fact Checking

Each claim is classified as:

* **Verified** – Claim matches current evidence.
* **Inaccurate** – Claim contains outdated or partially incorrect information.
* **False** – No supporting evidence found or claim is factually incorrect.

### Detailed Reporting

For every claim, the application provides:

* Verification Status
* Confidence Level
* Correct Fact
* Explanation
* Supporting Sources

### Summary Dashboard

Displays:

* Total Claims
* Verified Claims
* Inaccurate Claims
* False Claims
* Overall Accuracy Score

---

## System Workflow

```text
PDF Upload
      ↓
Claim Extraction
      ↓
Live Web Search
      ↓
Evidence Analysis
      ↓
Fact Verification
      ↓
Structured Report Generation
```

---

## Tech Stack

| Component         | Technology                |
| ----------------- | ------------------------- |
| Frontend          | Streamlit                 |
| Backend           | Python                    |
| LLM               | OpenRouter (GPT-OSS 120B) |
| Web Search        | Tavily API                |
| PDF Processing    | PyMuPDF                   |
| Report Generation | ReportLab                 |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Student-Dk/factcheck-ai.git
cd factcheck-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run Application

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using Streamlit Cloud.

### Live Application

https://factcheck-ai-mfddz3uzklhn628qttutzh.streamlit.app/

### Deployment Steps

1. Push code to GitHub.
2. Connect repository to Streamlit Cloud.
3. Add API keys in Streamlit Secrets.
4. Deploy the application.

---

## Project Structure

```text
factcheck-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
```

---

## Example Output

For each uploaded PDF, the application generates:

* Verification Summary
* Claim-by-Claim Analysis
* Supporting Evidence
* Corrected Facts
* Confidence Scores

---

## Future Improvements

* Multi-language support
* Batch document verification
* Source credibility scoring
* Export reports in multiple formats
* Historical claim tracking


