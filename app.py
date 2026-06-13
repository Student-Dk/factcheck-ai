import streamlit as st
import fitz  # PyMuPDF
from tavily import TavilyClient
import json
import os
import time
import requests
from dotenv import load_dotenv
from fpdf import FPDF
import re

load_dotenv()

# -------------------------------------------------------------------
# Configuration & API Setup
# -------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API keys. Please set OPENROUTER_API_KEY and TAVILY_API_KEY in your environment.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b:free"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# -------------------------------------------------------------------
# Page Config & Dark Theme (same as before, no emojis)
# -------------------------------------------------------------------
st.set_page_config(page_title="FactCheck Pro", page_icon="", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        text-align: center;
        color: #8fa8c0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .verified-card {
        background: #0a2e1a;
        border-left: 4px solid #00d48a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .inaccurate-card {
        background: #2e1f0a;
        border-left: 4px solid #ff9500;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .false-card {
        background: #2e0a0a;
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .claim-text {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .verdict-badge-verified {
        display: inline-block;
        background: #00d48a22;
        color: #00d48a;
        border: 1px solid #00d48a;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .verdict-badge-inaccurate {
        display: inline-block;
        background: #ff950022;
        color: #ff9500;
        border: 1px solid #ff9500;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .verdict-badge-false {
        display: inline-block;
        background: #ff444422;
        color: #ff4444;
        border: 1px solid #ff4444;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .explanation-text {
        color: #b0c4d8;
        font-size: 0.9rem;
        line-height: 1.4;
        margin: 0.3rem 0;
    }
    .correct-fact-text {
        color: #ffd966;
        font-size: 0.9rem;
        line-height: 1.4;
        margin: 0.3rem 0;
        font-style: italic;
    }
    .source-link {
        color: #00c6ff;
        font-size: 0.8rem;
        text-decoration: none;
        margin-right: 0.8rem;
    }
    .stat-box {
        background: #1a2535;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { color: #8fa8c0; font-size: 0.85rem; }
    hr { border-color: #2a3a4a; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def call_openrouter(prompt, system_message=None, max_retries=3):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                if attempt == max_retries - 1:
                    st.error(f"OpenRouter API error: {response.text}")
                    return None
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"OpenRouter exception: {str(e)}")
                return None
            time.sleep(2 ** attempt)
    return None

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No extractable text found. The PDF may be scanned or image-based.")
        return text.strip()
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return None

def extract_claims(pdf_text, max_claims=15):
    prompt = f"""
You are a fact-checking assistant. Extract the most important, verifiable claims from the text below.

Focus on:
- Statistics and percentages
- Dates and years
- Financial figures
- Named facts (e.g., company founded in year Y)
- Technical claims

Return ONLY a valid JSON array. No markdown, no backticks, no extra text.
Format: [{{"claim": "exact claim text", "category": "statistic|date|financial|technical|other"}}]

Limit to the {max_claims} most significant claims.

Text to analyze:
{pdf_text[:10000]}
"""
    response = call_openrouter(prompt, system_message="You are a helpful assistant that extracts claims as JSON.")
    if not response:
        return []
    try:
        raw = response['choices'][0]['message']['content'].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        if not isinstance(claims, list):
            raise ValueError("Not a list")
        return claims[:max_claims]
    except Exception:
        return []

def verify_claim(claim_text):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        search_results = tavily.search(query=claim_text, search_depth="basic", max_results=4)
    except Exception as e:
        return {"verdict": "False", "correct_fact": "Search failed.", "explanation": str(e), "confidence": "Low", "sources": []}
    
    search_context = ""
    sources = []
    for r in search_results.get("results", []):
        content = r.get("content", "")
        url = r.get("url", "")
        title = r.get("title", "")
        search_context += f"Source: {url}\nTitle: {title}\nContent: {content}\n\n"
        sources.append({"title": title, "url": url})
    
    prompt = f"""
You are a strict fact-checker. Evaluate the claim against the web search results.
The web search results are the primary source of truth. Do not use your own knowledge.

CLAIM: "{claim_text}"

WEB SEARCH RESULTS:
{search_context}

Based ONLY on the search results above, respond with ONLY valid JSON (no markdown, no backticks):
{{
  "verdict": "Verified" or "Inaccurate" or "False",
  "correct_fact": "If verdict is Verified, restate the claim as correct. If Inaccurate or False, provide the corrected fact based on search results.",
  "explanation": "2-3 sentence explanation supporting the verdict.",
  "confidence": "High" or "Medium" or "Low"
}}
"""
    response = call_openrouter(prompt, system_message="You are a strict fact-checker. Output only JSON.")
    if not response:
        return {"verdict": "False", "correct_fact": "API error.", "explanation": "Could not get response.", "confidence": "Low", "sources": sources[:2]}
    try:
        raw = response['choices'][0]['message']['content'].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result.setdefault("verdict", "False")
        result.setdefault("correct_fact", "No correction provided.")
        result.setdefault("explanation", "No explanation.")
        result.setdefault("confidence", "Low")
        result["sources"] = sources[:2]
        return result
    except Exception:
        return {"verdict": "False", "correct_fact": "Parsing error.", "explanation": "Invalid JSON from model.", "confidence": "Low", "sources": sources[:2]}

# -------------------------------------------------------------------
# PDF Report Generation
# -------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(15, 17, 23)
        self.cell(0, 10, "FactCheck Pro - Verification Report", ln=True, align="C", fill=True)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(143, 168, 192)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
    
    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(26, 37, 53)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)
    
    def stat_box(self, label, value, color_hex):
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        self.set_font("Arial", "B", 14)
        self.set_text_color(r, g, b)
        self.cell(35, 10, str(value), ln=0, align="C")
        self.set_font("Arial", "", 8)
        self.set_text_color(143, 168, 192)
        self.cell(0, 10, label, ln=True, align="C")
        self.ln(2)
    
    def claim_card(self, claim, verdict, correct_fact, explanation, confidence, sources):
        # Verdict color and label
        if verdict == "Verified":
            badge = "VERIFIED"
            color = (0, 212, 138)
        elif verdict == "Inaccurate":
            badge = "INACCURATE"
            color = (255, 149, 0)
        else:
            badge = "FALSE"
            color = (255, 68, 68)
        
        self.set_font("Arial", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 8, badge, ln=True)
        self.set_font("Arial", "I", 8)
        self.set_text_color(143, 168, 192)
        self.cell(0, 6, f"Confidence: {confidence}", ln=True)
        self.set_font("Arial", "B", 10)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 6, f'Claim: "{claim}"')
        self.set_font("Arial", "I", 9)
        self.set_text_color(255, 217, 102)
        self.multi_cell(0, 6, f"Correct fact: {correct_fact}")
        self.set_font("Arial", "", 9)
        self.set_text_color(176, 196, 216)
        self.multi_cell(0, 6, f"Explanation: {explanation}")
        self.set_font("Arial", "U", 8)
        self.set_text_color(0, 198, 255)
        for src in sources:
            if src.get("url"):
                self.cell(0, 6, f"Source: {src['url']}", ln=True)
        self.ln(4)

def generate_pdf_report(results, total, verified_count, inaccurate_count, false_count, accuracy_score):
    pdf = PDF()
    pdf.add_page()
    # Summary metrics
    pdf.section_title("Summary Metrics")
    pdf.stat_box("Total Claims", total, "#ffffff")
    pdf.stat_box("Verified", verified_count, "#00d48a")
    pdf.stat_box("Inaccurate", inaccurate_count, "#ff9500")
    pdf.stat_box("False", false_count, "#ff4444")
    pdf.stat_box("Accuracy Score", f"{accuracy_score}%", "#00c6ff")
    pdf.ln(6)
    
    # Detailed results
    pdf.section_title("Detailed Results")
    for r in results:
        pdf.claim_card(
            claim=r['claim'],
            verdict=r['verdict'],
            correct_fact=r.get('correct_fact', 'Not available'),
            explanation=r['explanation'],
            confidence=r.get('confidence', 'Low'),
            sources=r.get('sources', [])
        )
    return pdf.output(dest='S').encode('latin1')

# -------------------------------------------------------------------
# Main UI (fixed button width + PDF download)
# -------------------------------------------------------------------
st.markdown('<div class="hero-title">FactCheck Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a PDF → Extract claims → Verify against live web data → Professional report</div>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"], help="PDF files containing factual statements, statistics, or claims.")

if uploaded_file:
    # Better button layout: reduced width and centered
    col_space1, col_btn, col_space2 = st.columns([2, 1.5, 2])
    with col_btn:
        analyze_btn = st.button("Start Fact-Checking", use_container_width=True, type="primary")
    
    if analyze_btn:
        with st.spinner("Reading PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text is None:
                st.stop()
        st.success(f"PDF processed — {len(pdf_text.split())} words")
        
        with st.spinner("Extracting claims..."):
            claims = extract_claims(pdf_text)
        if not claims:
            st.warning("No verifiable claims found.")
            st.stop()
        st.info(f"Found {len(claims)} claims. Searching web...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, claim_obj in enumerate(claims):
            claim_text = claim_obj.get("claim", "")
            status_text.text(f"Verifying {i+1}/{len(claims)}: {claim_text[:80]}...")
            try:
                verification = verify_claim(claim_text)
                results.append({"claim": claim_text, "category": claim_obj.get("category", "other"), **verification})
            except Exception as e:
                results.append({"claim": claim_text, "category": "other", "verdict": "False", "correct_fact": "Error", "explanation": str(e), "confidence": "Low", "sources": []})
            progress_bar.progress((i+1)/len(claims))
            time.sleep(0.05)
        status_text.empty()
        progress_bar.empty()
        
        # Compute metrics
        total = len(results)
        verified = [r for r in results if r["verdict"] == "Verified"]
        inaccurate = [r for r in results if r["verdict"] == "Inaccurate"]
        false = [r for r in results if r["verdict"] == "False"]
        accuracy_score = int((len(verified)/total)*100) if total>0 else 0
        
        # Display on screen (same as before)
        st.markdown("## Fact-Check Report")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Claims</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#00d48a">{len(verified)}</div><div class="stat-label">Verified</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff9500">{len(inaccurate)}</div><div class="stat-label">Inaccurate</div></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff4444">{len(false)}</div><div class="stat-label">False</div></div>', unsafe_allow_html=True)
        with col5: st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#00c6ff">{accuracy_score}%</div><div class="stat-label">Accuracy Score</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display detailed cards on screen
        sections = [("False Claims", false, "false"), ("Inaccurate Claims", inaccurate, "inaccurate"), ("Verified Claims", verified, "verified")]
        for section_title, section_results, card_type in sections:
            if not section_results: continue
            st.markdown(f"### {section_title}")
            for r in section_results:
                badge_label = {"verified":"VERIFIED","inaccurate":"INACCURATE","false":"FALSE"}[card_type]
                sources_html = "".join([f'<a class="source-link" href="{src["url"]}" target="_blank">{src.get("title", src["url"])[:60]}</a>' for src in r.get("sources",[]) if src.get("url")])
                st.markdown(f"""
                <div class="{card_type}-card">
                    <span class="verdict-badge-{card_type}">{badge_label}</span>
                    <span style="color:#8fa8c0; font-size:0.7rem; margin-left:8px">Confidence: {r.get('confidence','?')}</span>
                    <div class="claim-text">"{r['claim']}"</div>
                    <div class="correct-fact-text">Correct fact: {r.get('correct_fact','Not available')}</div>
                    <div class="explanation-text">{r['explanation']}</div>
                    <div style="margin-top:0.5rem">{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # PDF DOWNLOAD BUTTON (instead of JSON)
        st.markdown("---")
        pdf_bytes = generate_pdf_report(results, total, len(verified), len(inaccurate), len(false), accuracy_score)
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name="factcheck_report.pdf",
            mime="application/pdf",
            use_container_width=False
        )
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #8fa8c0;">
        <div style="font-size: 3rem;">📄</div>
        <div style="font-size: 1.2rem; margin-top: 1rem;">Upload a PDF to begin verification</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">The system extracts factual claims and verifies them against live web data.</div>
    </div>
    """, unsafe_allow_html=True)
