import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from tavily import TavilyClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ── Page Setup ───────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00c6ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #8fa8c0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .verified-card {
        background: linear-gradient(135deg, #0a2e1a, #0d3b22);
        border: 1px solid #00d48a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .inaccurate-card {
        background: linear-gradient(135deg, #2e1f0a, #3b2a0d);
        border: 1px solid #ff9500;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .false-card {
        background: linear-gradient(135deg, #2e0a0a, #3b0d0d);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .claim-text {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    .verdict-badge-verified {
        display: inline-block;
        background: #00d48a22;
        color: #00d48a;
        border: 1px solid #00d48a;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .verdict-badge-inaccurate {
        display: inline-block;
        background: #ff950022;
        color: #ff9500;
        border: 1px solid #ff9500;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .verdict-badge-false {
        display: inline-block;
        background: #ff444422;
        color: #ff4444;
        border: 1px solid #ff4444;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .explanation-text {
        color: #b0c4d8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .source-link {
        color: #00c6ff;
        font-size: 0.8rem;
        text-decoration: none;
    }
    .stat-box {
        background: #1a2535;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { color: #8fa8c0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    """Extract all text from uploaded PDF"""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_claims(pdf_text):
    """Use Gemini to extract specific verifiable claims from PDF"""
    prompt = f"""
You are a fact-checking assistant. Extract ALL specific, verifiable claims from the text below.

Focus on:
- Statistics and percentages (e.g., "X% of people...")
- Dates and years (e.g., "In 2023, X happened...")
- Financial figures (e.g., "revenue of $X billion")
- Named facts (e.g., "Company X was founded in year Y")
- Technical claims (e.g., "Model X achieves Y% accuracy")

Return ONLY a valid JSON array. No explanation, no markdown, no backticks.
Format: [{{"claim": "exact claim text", "category": "statistic/date/financial/technical/other"}}]

Text to analyze:
{pdf_text[:8000]}
"""
    response = model.generate_content(prompt)
    raw = response.text.strip()
    # clean any markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    claims = json.loads(raw)
    return claims


def search_and_verify(claim_text):
    """Search web for claim and get Gemini to judge it"""
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    # Web search
    search_results = tavily.search(
        query=claim_text,
        search_depth="basic",
        max_results=4
    )

    # Format search results for Gemini
    search_context = ""
    sources = []
    for r in search_results.get("results", []):
        search_context += f"Source: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}\n\n"
        sources.append({"title": r.get("title", ""), "url": r.get("url", "")})

    # Gemini judges the claim
    prompt = f"""
You are a strict fact-checker. Evaluate this claim against the web search results.

CLAIM: "{claim_text}"

WEB SEARCH RESULTS:
{search_context}

Based ONLY on the search results above, respond with ONLY valid JSON (no markdown, no backticks):
{{
  "verdict": "Verified" or "Inaccurate" or "False",
  "explanation": "2-3 sentence explanation. If Inaccurate, state the correct fact.",
  "confidence": "High" or "Medium" or "Low"
}}

Rules:
- "Verified" = search results confirm the claim
- "Inaccurate" = claim has wrong numbers/dates but the topic exists (e.g., outdated stat)
- "False" = claim is contradicted by evidence OR no evidence found at all
"""
    response = model.generate_content(prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["sources"] = sources[:2]
    return result


# ── UI ───────────────────────────────────────────────────

st.markdown('<div class="hero-title">🔍 FactCheck AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a PDF → AI extracts claims → Live web verification → Instant report</div>', unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"],
    help="Upload any PDF containing claims, statistics, or factual assertions"
)

if uploaded_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 Start Fact-Checking", use_container_width=True, type="primary")

    if analyze_btn:
        # Step 1: Extract text
        with st.spinner("📖 Reading PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if not pdf_text.strip():
                st.error("Could not extract text from PDF. Make sure it's not a scanned image.")
                st.stop()

        st.success(f"✅ PDF read successfully — {len(pdf_text.split())} words extracted")

        # Step 2: Extract claims
        with st.spinner("🤖 Gemini is extracting claims..."):
            try:
                claims = extract_claims(pdf_text)
            except Exception as e:
                st.error(f"Could not parse claims: {e}")
                st.stop()

        if not claims:
            st.warning("No specific verifiable claims found in this document.")
            st.stop()

        st.info(f"🎯 Found **{len(claims)} claims** to verify. Starting web search...")

        # Step 3: Verify each claim
        results = []
        progress = st.progress(0)
        status_text = st.empty()

        for i, claim_obj in enumerate(claims):
            claim_text = claim_obj.get("claim", "")
            status_text.text(f"🔎 Verifying claim {i+1}/{len(claims)}: {claim_text[:60]}...")

            try:
                verdict = search_and_verify(claim_text)
                results.append({
                    "claim": claim_text,
                    "category": claim_obj.get("category", "other"),
                    **verdict
                })
            except Exception as e:
                results.append({
                    "claim": claim_text,
                    "category": claim_obj.get("category", "other"),
                    "verdict": "False",
                    "explanation": f"Could not verify: {str(e)}",
                    "confidence": "Low",
                    "sources": []
                })

            progress.progress((i + 1) / len(claims))

        status_text.empty()
        progress.empty()

        # ── Results ──────────────────────────────────────

        st.markdown("## 📊 Fact-Check Report")

        verified = [r for r in results if r["verdict"] == "Verified"]
        inaccurate = [r for r in results if r["verdict"] == "Inaccurate"]
        false = [r for r in results if r["verdict"] == "False"]

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number" style="color:#00d48a">{len(verified)}</div>
                <div class="stat-label">✅ Verified</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number" style="color:#ff9500">{len(inaccurate)}</div>
                <div class="stat-label">⚠️ Inaccurate</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number" style="color:#ff4444">{len(false)}</div>
                <div class="stat-label">❌ False</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            score = int((len(verified) / len(results)) * 100) if results else 0
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number" style="color:#00c6ff">{score}%</div>
                <div class="stat-label">📈 Accuracy Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed results
        verdict_order = [("❌ False Claims", false, "false"),
                         ("⚠️ Inaccurate Claims", inaccurate, "inaccurate"),
                         ("✅ Verified Claims", verified, "verified")]

        for section_title, section_results, card_type in verdict_order:
            if not section_results:
                continue
            st.markdown(f"### {section_title}")
            for r in section_results:
                badge_class = f"verdict-badge-{card_type}"
                card_class = f"{card_type}-card"
                badge_label = {"verified": "✅ VERIFIED", "inaccurate": "⚠️ INACCURATE", "false": "❌ FALSE"}[card_type]

                sources_html = ""
                for src in r.get("sources", []):
                    if src.get("url"):
                        sources_html += f'<a class="source-link" href="{src["url"]}" target="_blank">🔗 {src.get("title", src["url"])[:60]}</a><br>'

                st.markdown(f"""
                <div class="{card_class}">
                    <span class="{badge_class}">{badge_label}</span>
                    <span style="color:#8fa8c0; font-size:0.75rem; margin-left:8px">Confidence: {r.get('confidence','?')}</span>
                    <div class="claim-text">"{r['claim']}"</div>
                    <div class="explanation-text">{r['explanation']}</div>
                    <div style="margin-top:0.5rem">{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)

        # Download JSON report
        st.markdown("---")
        st.download_button(
            label="⬇️ Download Full Report (JSON)",
            data=json.dumps(results, indent=2),
            file_name="factcheck_report.json",
            mime="application/json"
        )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #8fa8c0;">
        <div style="font-size: 4rem;">📄</div>
        <div style="font-size: 1.2rem; margin-top: 1rem;">Upload a PDF to get started</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
            The AI will extract all factual claims and verify them against live web data
        </div>
    </div>
    """, unsafe_allow_html=True)
