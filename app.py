import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from tavily import TavilyClient
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# Configuration & API Setup
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API keys. Please set GEMINI_API_KEY and TAVILY_API_KEY in your environment.")
    st.stop()

# Configure Gemini with a supported model (gemini-1.5-pro)
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------------------------
# Page Configuration & Professional Dark Theme
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FactCheck Pro",
    page_icon="",
    layout="wide"
)

# Dark theme CSS (no emojis)
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
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF with error handling for empty or scanned docs."""
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
        st.error(f"Failed to extract text from PDF: {str(e)}")
        return None

def extract_claims(pdf_text, max_claims=15):
    """Use Gemini to extract verifiable claims. Limit to max_claims. Robust JSON parsing."""
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
    try:
        response = model.generate_content(prompt, request_options={"timeout": 60})
        raw = response.text.strip()
        # Clean markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        if not isinstance(claims, list):
            raise ValueError("Response is not a JSON array")
        # Limit to max_claims
        return claims[:max_claims]
    except json.JSONDecodeError as e:
        st.warning(f"Gemini returned malformed JSON. Attempting repair... {str(e)}")
        # Fallback: try to extract array with regex
        import re
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            try:
                claims = json.loads(match.group())
                return claims[:max_claims]
            except:
                pass
        return []
    except Exception as e:
        st.error(f"Claim extraction failed: {str(e)}")
        return []

def verify_claim(claim_text):
    """
    Search web via Tavily, then use Gemini to evaluate claim.
    Gemini is used ONLY for reasoning over search results, not as source of truth.
    Returns dict with verdict, correct_fact, explanation, confidence, sources.
    """
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    
    try:
        # Web search with timeout handling
        search_results = tavily.search(
            query=claim_text,
            search_depth="basic",
            max_results=4
        )
    except Exception as e:
        return {
            "verdict": "False",
            "correct_fact": "Unable to search web due to API error.",
            "explanation": f"Tavily search failed: {str(e)}",
            "confidence": "Low",
            "sources": []
        }
    
    # Format search results
    search_context = ""
    sources = []
    for r in search_results.get("results", []):
        content = r.get("content", "")
        url = r.get("url", "")
        title = r.get("title", "")
        search_context += f"Source: {url}\nTitle: {title}\nContent: {content}\n\n"
        sources.append({"title": title, "url": url})
    
    # Gemini prompt: must output correct_fact
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

Rules:
- "Verified": Search results confirm the claim exactly.
- "Inaccurate": Claim has wrong numbers/dates but the topic exists.
- "False": Claim is contradicted by evidence OR no supporting evidence found.
- Always include a clear correct_fact.
"""
    try:
        response = model.generate_content(prompt, request_options={"timeout": 60})
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # Ensure all required fields exist
        result.setdefault("verdict", "False")
        result.setdefault("correct_fact", "No correction provided.")
        result.setdefault("explanation", "No explanation available.")
        result.setdefault("confidence", "Low")
        result["sources"] = sources[:2]
        return result
    except Exception as e:
        return {
            "verdict": "False",
            "correct_fact": "Verification failed due to an internal error.",
            "explanation": f"Gemini evaluation error: {str(e)}",
            "confidence": "Low",
            "sources": sources[:2]
        }

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
st.markdown('<div class="hero-title">FactCheck Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a PDF → Extract claims → Verify against live web data → Professional report</div>', unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload PDF document",
    type=["pdf"],
    help="PDF files containing factual statements, statistics, or claims."
)

if uploaded_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("Start Fact-Checking", use_container_width=True, type="primary")
    
    if analyze_btn:
        # Step 1: Extract text
        with st.spinner("Reading PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text is None:
                st.stop()
        
        st.success(f"PDF processed successfully — {len(pdf_text.split())} words extracted")
        
        # Step 2: Extract claims
        with st.spinner("Extracting verifiable claims..."):
            claims = extract_claims(pdf_text)
        
        if not claims:
            st.warning("No verifiable claims found in this document. Try a different PDF.")
            st.stop()
        
        st.info(f"Found {len(claims)} claims to verify. Searching the web...")
        
        # Step 3: Verify each claim
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, claim_obj in enumerate(claims):
            claim_text = claim_obj.get("claim", "")
            status_text.text(f"Verifying claim {i+1}/{len(claims)}: {claim_text[:80]}...")
            
            try:
                verification = verify_claim(claim_text)
                results.append({
                    "claim": claim_text,
                    "category": claim_obj.get("category", "other"),
                    **verification
                })
            except Exception as e:
                results.append({
                    "claim": claim_text,
                    "category": claim_obj.get("category", "other"),
                    "verdict": "False",
                    "correct_fact": "Verification process encountered an error.",
                    "explanation": f"Error: {str(e)}",
                    "confidence": "Low",
                    "sources": []
                })
            
            progress_bar.progress((i + 1) / len(claims))
            time.sleep(0.1)  # Smooth UI
        
        status_text.empty()
        progress_bar.empty()
        
        # -------------------------------------------------------------------
        # Professional Report Layout
        # -------------------------------------------------------------------
        st.markdown("## Fact-Check Report")
        
        # Calculate metrics
        total = len(results)
        verified = [r for r in results if r["verdict"] == "Verified"]
        inaccurate = [r for r in results if r["verdict"] == "Inaccurate"]
        false = [r for r in results if r["verdict"] == "False"]
        accuracy_score = int((len(verified) / total) * 100) if total > 0 else 0
        
        # Summary metrics (no emojis)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#ffffff">{total}</div>
                <div class="stat-label">Total Claims</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#00d48a">{len(verified)}</div>
                <div class="stat-label">Verified</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#ff9500">{len(inaccurate)}</div>
                <div class="stat-label">Inaccurate</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#ff4444">{len(false)}</div>
                <div class="stat-label">False</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#00c6ff">{accuracy_score}%</div>
                <div class="stat-label">Accuracy Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed results sections
        sections = [
            ("False Claims", false, "false"),
            ("Inaccurate Claims", inaccurate, "inaccurate"),
            ("Verified Claims", verified, "verified")
        ]
        
        for section_title, section_results, card_type in sections:
            if not section_results:
                continue
            st.markdown(f"### {section_title}")
            for r in section_results:
                badge_class = f"verdict-badge-{card_type}"
                card_class = f"{card_type}-card"
                badge_label = {"verified": "VERIFIED", "inaccurate": "INACCURATE", "false": "FALSE"}[card_type]
                
                # Build sources HTML
                sources_html = ""
                for src in r.get("sources", []):
                    if src.get("url"):
                        sources_html += f'<a class="source-link" href="{src["url"]}" target="_blank">{src.get("title", src["url"])[:60]}</a>'
                
                st.markdown(f"""
                <div class="{card_class}">
                    <span class="{badge_class}">{badge_label}</span>
                    <span style="color:#8fa8c0; font-size:0.7rem; margin-left:8px">Confidence: {r.get('confidence', '?')}</span>
                    <div class="claim-text">"{r['claim']}"</div>
                    <div class="correct-fact-text">Correct fact: {r.get('correct_fact', 'Not available')}</div>
                    <div class="explanation-text">{r['explanation']}</div>
                    <div style="margin-top:0.5rem">{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Download JSON report
        st.markdown("---")
        st.download_button(
            label="Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="factcheck_report.json",
            mime="application/json"
        )

else:
    # Empty state (no emojis)
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #8fa8c0;">
        <div style="font-size: 3rem;">📄</div>
        <div style="font-size: 1.2rem; margin-top: 1rem;">Upload a PDF to begin verification</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
            The system extracts factual claims and verifies them against live web data.
        </div>
    </div>
    """, unsafe_allow_html=True)
