import streamlit as st
import fitz  # PyMuPDF
from tavily import TavilyClient
import json
import os
import time
import requests
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv()

# -------------------------------------------------------------------
# Configuration & API Setup
# -------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")

if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API keys. Please set OPENROUTER_API_KEY and TAVILY_API_KEY.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME     = "openai/gpt-oss-120b:free"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# -------------------------------------------------------------------
# Page Config & Styles
# -------------------------------------------------------------------
st.set_page_config(page_title="FactCheck Pro", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .hero-title {
        font-size: 2.8rem; font-weight: 700; color: #ffffff;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .hero-sub { text-align: center; color: #8fa8c0; font-size: 1rem; margin-bottom: 2rem; }
    .verified-card   { background:#0a2e1a; border-left:4px solid #00d48a; border-radius:8px; padding:1rem 1.2rem; margin:0.8rem 0; }
    .inaccurate-card { background:#2e1f0a; border-left:4px solid #ff9500; border-radius:8px; padding:1rem 1.2rem; margin:0.8rem 0; }
    .false-card      { background:#2e0a0a; border-left:4px solid #ff4444; border-radius:8px; padding:1rem 1.2rem; margin:0.8rem 0; }
    .claim-text { font-size:1rem; font-weight:600; color:#ffffff; margin-bottom:0.5rem; }
    .verdict-badge-verified   { display:inline-block; background:#00d48a22; color:#00d48a; border:1px solid #00d48a; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; margin-bottom:0.6rem; }
    .verdict-badge-inaccurate { display:inline-block; background:#ff950022; color:#ff9500; border:1px solid #ff9500; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; margin-bottom:0.6rem; }
    .verdict-badge-false      { display:inline-block; background:#ff444422; color:#ff4444; border:1px solid #ff4444; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; margin-bottom:0.6rem; }
    .explanation-text  { color:#b0c4d8; font-size:0.9rem; line-height:1.4; margin:0.3rem 0; }
    .correct-fact-text { color:#ffd966; font-size:0.9rem; line-height:1.4; margin:0.3rem 0; font-style:italic; }
    .source-link { color:#00c6ff; font-size:0.8rem; text-decoration:none; margin-right:0.8rem; }
    .stat-box { background:#1a2535; border-radius:8px; padding:1rem; text-align:center; }
    .stat-number { font-size:2rem; font-weight:800; }
    .stat-label { color:#8fa8c0; font-size:0.85rem; }
    .methodology-box { background:#111827; border:1px solid #1e3a5f; border-radius:10px; padding:1rem 1.4rem; margin:1rem 0; color:#8fa8c0; font-size:0.85rem; }
    hr { border-color: #2a3a4a; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper: OpenRouter call with retries
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
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if attempt == max_retries - 1:
                st.error(f"OpenRouter error ({resp.status_code}): {resp.text}")
                return None
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                st.error("OpenRouter API timed out.")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"OpenRouter exception: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

# -------------------------------------------------------------------
# Helper: PDF text extraction
# -------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No extractable text found. PDF may be scanned or image-based.")
        return text.strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return None

# -------------------------------------------------------------------
# Helper: Extract ALL claims via LLM
# -------------------------------------------------------------------
def extract_claims(pdf_text):
    prompt = f"""
You are a fact-checking assistant. Extract ALL specific, verifiable claims from the text below.

Focus on:
- Statistics and percentages
- Dates and years
- Financial figures
- Named facts (company founded in year Y, etc.)
- Technical claims

Extract every verifiable claim you find — do not skip any.

Return ONLY a valid JSON array. No markdown, no backticks, no extra text.
Format: [{{"claim": "exact claim text", "category": "statistic|date|financial|technical|other"}}]

Text:
{pdf_text[:10000]}
"""
    response = call_openrouter(prompt, system_message="You extract claims as JSON only.")
    if not response:
        return []

    try:
        raw = response["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        if not isinstance(claims, list):
            raise ValueError("Not a JSON array")
        return claims
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        st.warning("Could not parse claims from model response.")
        return []
    except Exception as e:
        st.error(f"Claim extraction failed: {e}")
        return []

# -------------------------------------------------------------------
# Helper: Verify single claim via Tavily + LLM
# -------------------------------------------------------------------
def verify_claim(claim_text):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        search_results = tavily.search(
            query=claim_text,
            search_depth="basic",
            max_results=4
        )
    except Exception as e:
        return {
            "verdict": "False",
            "correct_fact": "Unable to search — API error.",
            "explanation": f"Tavily search failed: {e}",
            "confidence": "Low",
            "sources": []
        }

    search_context = ""
    sources = []
    for r in search_results.get("results", []):
        search_context += (
            f"Source: {r.get('url','')}\n"
            f"Title: {r.get('title','')}\n"
            f"Content: {r.get('content','')}\n\n"
        )
        sources.append({"title": r.get("title",""), "url": r.get("url","")})

    prompt = f"""
You are a strict fact-checker. Evaluate the claim using ONLY the web search results below.
Do NOT use your own training knowledge as evidence.

CLAIM: "{claim_text}"

WEB SEARCH RESULTS:
{search_context}

Respond with ONLY valid JSON (no markdown, no backticks):
{{
  "verdict": "Verified" or "Inaccurate" or "False",
  "correct_fact": "If Verified, restate the claim correctly. If Inaccurate or False, give the corrected fact from search results.",
  "explanation": "2-3 sentence explanation based on search evidence.",
  "confidence": "High" or "Medium" or "Low"
}}

Rules:
- Verified   = search results clearly confirm the claim
- Inaccurate = right topic but wrong number/date/figure
- False      = directly contradicted OR no supporting evidence found
"""
    response = call_openrouter(prompt, system_message="You are a strict fact-checker. Output only JSON.")
    if not response:
        return {
            "verdict": "False",
            "correct_fact": "Verification failed — API error.",
            "explanation": "No response from model.",
            "confidence": "Low",
            "sources": sources[:2]
        }

    try:
        raw = response["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result.setdefault("verdict",      "False")
        result.setdefault("correct_fact", "No correction provided.")
        result.setdefault("explanation",  "No explanation available.")
        result.setdefault("confidence",   "Low")
        result["sources"] = sources[:2]
        return result
    except Exception as e:
        return {
            "verdict": "False",
            "correct_fact": "Parsing error.",
            "explanation": f"Error: {e}",
            "confidence": "Low",
            "sources": sources[:2]
        }

# -------------------------------------------------------------------
# PDF Report Generator
# -------------------------------------------------------------------
def S(name, **kw):
    return ParagraphStyle(name, **kw)

C_DARK = colors.HexColor("#0f1117")
C_MID  = colors.HexColor("#1a2535")
C_BLUE = colors.HexColor("#00c6ff")
C_GREY = colors.HexColor("#8fa8c0")
C_TEXT = colors.HexColor("#b0c4d8")

def generate_pdf_report(results):
    buf = BytesIO()
    W   = A4[0] - 36*mm
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm)

    sTitle   = S("t",  fontSize=26, leading=32, textColor=colors.white,   fontName="Helvetica-Bold", alignment=TA_LEFT)
    sSub     = S("s",  fontSize=10, leading=14, textColor=C_GREY,          fontName="Helvetica",      alignment=TA_LEFT)
    sSection = S("se", fontSize=13, leading=18, textColor=C_BLUE,          fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    sClaim   = S("cl", fontSize=10, leading=14, textColor=colors.white,    fontName="Helvetica-Bold")
    sBody    = S("bo", fontSize=9,  leading=13, textColor=C_TEXT,          fontName="Helvetica")
    sCorrect = S("co", fontSize=9,  leading=13, textColor=colors.HexColor("#7dd3fc"), fontName="Helvetica-Oblique")
    sMeta    = S("me", fontSize=8,  leading=12, textColor=C_GREY,          fontName="Helvetica")
    sFooter  = S("fo", fontSize=8,  leading=11, textColor=C_GREY,          fontName="Helvetica", alignment=TA_CENTER)

    story = []

    # Header
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("FactCheck Pro", sTitle))
    story.append(Paragraph(
        f"Automated Fact-Check Report  |  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
        sSub))
    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width=W, thickness=1, color=C_BLUE, spaceAfter=6*mm))

    # Summary stats
    total      = len(results)
    n_verified = sum(1 for r in results if r["verdict"] == "Verified")
    n_inaccur  = sum(1 for r in results if r["verdict"] == "Inaccurate")
    n_false    = sum(1 for r in results if r["verdict"] == "False")
    score      = int((n_verified / total) * 100) if total else 0

    def big(val, col):
        return Paragraph(f"<b>{val}</b>",
            S("x", fontName="Helvetica-Bold", fontSize=22, leading=28,
              textColor=colors.HexColor(col), alignment=TA_CENTER))
    def small(lbl):
        return Paragraph(lbl,
            S("y", fontName="Helvetica", fontSize=8, leading=11,
              textColor=C_GREY, alignment=TA_CENTER))

    stat_data = [
        [big(total,"#ffffff"), big(n_verified,"#00d48a"), big(n_inaccur,"#ff9500"), big(n_false,"#ff4444"), big(f"{score}%","#00c6ff")],
        [small("Total Claims"), small("Verified"), small("Inaccurate"), small("False"), small("Accuracy")],
    ]
    stat_tbl = Table(stat_data, colWidths=[W/5]*5, rowHeights=[18*mm, 7*mm])
    stat_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_MID),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LINEAFTER",    (0,0),(3,1),   0.5, colors.HexColor("#2a3a4a")),
    ]))
    story.append(stat_tbl)
    story.append(Spacer(1, 8*mm))

    # Verdict sections — worst first
    sections = [
        ("False Claims",      [r for r in results if r["verdict"]=="False"],      "#ff4444","#2e0a0a"),
        ("Inaccurate Claims", [r for r in results if r["verdict"]=="Inaccurate"], "#ff9500","#2e1f0a"),
        ("Verified Claims",   [r for r in results if r["verdict"]=="Verified"],   "#00d48a","#0a2e1a"),
    ]

    for sec_title, sec_results, hex_col, hex_bg in sections:
        if not sec_results:
            continue
        story.append(Paragraph(sec_title, sSection))
        story.append(HRFlowable(width=W, thickness=0.5,
            color=colors.HexColor(hex_col), spaceAfter=3*mm))

        for r in sec_results:
            verdict  = r.get("verdict","?")
            claim    = r.get("claim","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            category = r.get("category","other").capitalize()
            conf     = r.get("confidence","?")
            expl     = r.get("explanation","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            correct  = r.get("correct_fact","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            sources  = r.get("sources",[])

            badge_map  = {"Verified":"VERIFIED","Inaccurate":"INACCURATE","False":"FALSE"}
            badge_text = badge_map.get(verdict, verdict.upper())

            inner = []
            inner.append(Paragraph(
                f'<font color="{hex_col}"><b>[{badge_text}]</b></font>'
                f'  <font color="#8fa8c0" size="8">Confidence: {conf}  |  Category: {category}</font>',
                S("bm", fontName="Helvetica-Bold", fontSize=9, leading=13,
                  textColor=colors.HexColor(hex_col))))
            inner.append(Spacer(1, 2*mm))
            inner.append(Paragraph(f'"{claim}"', sClaim))
            inner.append(Spacer(1, 1.5*mm))
            inner.append(Paragraph(expl, sBody))

            if correct and correct.strip().lower() not in ("n/a","","not available"):
                inner.append(Spacer(1, 1.5*mm))
                inner.append(Paragraph(f"Correct fact: {correct}", sCorrect))

            if sources:
                src_parts = []
                for src in sources[:2]:
                    url   = src.get("url","")
                    title = src.get("title", url)[:60]
                    if url:
                        src_parts.append(f'<a href="{url}" color="#00c6ff">{title}</a>')
                if src_parts:
                    inner.append(Spacer(1, 1.5*mm))
                    inner.append(Paragraph("Sources: " + "  |  ".join(src_parts), sMeta))

            card = Table([[inner]], colWidths=[W - 6*mm])
            card.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor(hex_bg)),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LINEAFTER",    (0,0),(-1,-1), 4, colors.HexColor(hex_col)),
                ("VALIGN",       (0,0),(-1,-1), "TOP"),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 4*mm))

        story.append(Spacer(1, 2*mm))

    # Footer
    story.append(HRFlowable(width=W, thickness=0.5, color=C_GREY,
        spaceBefore=4*mm, spaceAfter=3*mm))
    story.append(Paragraph(
        "Generated by FactCheck Pro  |  Claims verified via live web search (Tavily API)  |  "
        "AI reasoning by OpenRouter  |  Results are indicative and should be independently verified.",
        sFooter))

    def dark_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_DARK)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=dark_bg, onLaterPages=dark_bg)
    return buf.getvalue()

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
st.markdown('<div class="hero-title">FactCheck Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a PDF → Extract claims → Verify against live web data → Professional report</div>', unsafe_allow_html=True)

st.markdown("""
<div class="methodology-box">
  <b style="color:#00c6ff">How it works:</b><br>
  1. LLM extracts ALL verifiable claims (stats, dates, figures) from your PDF.<br>
  2. Each claim is searched on the <b>live web</b> using Tavily API — real-time results.<br>
  3. The model reasons <b>only over those search results</b> to classify each claim.<br>
  Verdicts are grounded in live web evidence, not model hallucination.
</div>
""", unsafe_allow_html=True)

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
        # Step 1: Read PDF
        with st.spinner("Reading PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text is None:
                st.stop()
        st.success(f"PDF processed — {len(pdf_text.split())} words extracted")

        # Step 2: Extract ALL claims
        with st.spinner("Extracting all verifiable claims..."):
            claims = extract_claims(pdf_text)
        if not claims:
            st.warning("No verifiable claims found. Try a different PDF.")
            st.stop()

        st.info(f"Found **{len(claims)} claims** to verify. Searching the web for each...")

        # Step 3: Verify each claim
        results      = []
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, claim_obj in enumerate(claims):
            claim_text = claim_obj.get("claim", "")
            status_text.text(f"Verifying {i+1}/{len(claims)}: {claim_text[:80]}...")

            try:
                verification = verify_claim(claim_text)
                results.append({
                    "claim":    claim_text,
                    "category": claim_obj.get("category", "other"),
                    **verification
                })
            except Exception as e:
                results.append({
                    "claim":        claim_text,
                    "category":     claim_obj.get("category", "other"),
                    "verdict":      "False",
                    "correct_fact": "Verification error.",
                    "explanation":  f"Error: {e}",
                    "confidence":   "Low",
                    "sources":      []
                })

            progress_bar.progress((i + 1) / len(claims))
            time.sleep(0.1)

        status_text.empty()
        progress_bar.empty()

        # -------------------------------------------------------------------
        # Results
        # -------------------------------------------------------------------
        st.markdown("## Fact-Check Report")

        total      = len(results)
        verified   = [r for r in results if r["verdict"] == "Verified"]
        inaccurate = [r for r in results if r["verdict"] == "Inaccurate"]
        false_     = [r for r in results if r["verdict"] == "False"]
        score      = int((len(verified) / total) * 100) if total else 0

        # Summary stats
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#fff">{total}</div><div class="stat-label">Total Claims</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#00d48a">{len(verified)}</div><div class="stat-label">Verified</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff9500">{len(inaccurate)}</div><div class="stat-label">Inaccurate</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff4444">{len(false_)}</div><div class="stat-label">False</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#00c6ff">{score}%</div><div class="stat-label">Accuracy Score</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Claim cards — worst first
        for sec_title, sec_results, card_type in [
            ("False Claims",      false_,     "false"),
            ("Inaccurate Claims", inaccurate, "inaccurate"),
            ("Verified Claims",   verified,   "verified"),
        ]:
            if not sec_results:
                continue
            st.markdown(f"### {sec_title}")
            for r in sec_results:
                badge_label = {"verified":"VERIFIED","inaccurate":"INACCURATE","false":"FALSE"}[card_type]
                sources_html = ""
                for src in r.get("sources", []):
                    if src.get("url"):
                        sources_html += f'<a class="source-link" href="{src["url"]}" target="_blank">{src.get("title", src["url"])[:60]}</a>'

                st.markdown(f"""
                <div class="{card_type}-card">
                    <span class="verdict-badge-{card_type}">{badge_label}</span>
                    <span style="color:#8fa8c0;font-size:0.7rem;margin-left:8px">Confidence: {r.get('confidence','?')}</span>
                    <div class="claim-text">"{r['claim']}"</div>
                    <div class="correct-fact-text">Correct fact: {r.get('correct_fact','Not available')}</div>
                    <div class="explanation-text">{r['explanation']}</div>
                    <div style="margin-top:0.5rem">{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)

        # Download buttons
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        with dl1:
            pdf_bytes = generate_pdf_report(results)
            st.download_button(
                label="⬇️ Download Report (PDF)",
                data=pdf_bytes,
                file_name="factcheck_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        with dl2:
            st.download_button(
                label="⬇️ Download Raw Data (JSON)",
                data=json.dumps(results, indent=2),
                file_name="factcheck_report.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#8fa8c0;">
        <div style="font-size:3rem">📄</div>
        <div style="font-size:1.2rem;margin-top:1rem">Upload a PDF to begin verification</div>
        <div style="font-size:0.9rem;margin-top:0.5rem">
            The system extracts all factual claims and verifies them against live web data.
        </div>
    </div>
    """, unsafe_allow_html=True)
