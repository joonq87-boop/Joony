import streamlit as st
import pymupdf
import os
from google import genai

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Briefer",
    page_icon="📄",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }
    .main { padding-top: 2rem; }
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        width: 100%;
    }
    .stButton > button:hover { background: #333; }
    .brief-card {
        background: #f9f8f6;
        border: 1px solid #e8e5e0;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
    }
    .brief-card h4 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        color: #1a1a1a;
    }
    .tag {
        display: inline-block;
        background: #1a1a1a;
        color: white;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
    }
    .chat-user {
        background: #1a1a1a;
        color: white;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-ai {
        background: #f9f8f6;
        border: 1px solid #e8e5e0;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        font-size: 0.95rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e8e5e0;
        margin: 1.5rem 0;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Gemini client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Make sure it's set in your environment.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()
MODEL = "gemini-2.5-flash"

# ── PDF text extraction ───────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

# ── Summarise document ────────────────────────────────────────────────────────
def summarise(text: str) -> dict:
    prompt = f"""You are a professional document analyst. Analyse the document below and return a structured brief.

Return your response in EXACTLY this format with these exact headers:

SUMMARY:
- [bullet 1]
- [bullet 2]
- [bullet 3]
- [bullet 4]
- [bullet 5]

KEY DATES:
- [date and what it refers to, or "None found" if no dates]

ACTION ITEMS:
- [action item, or "None found" if no actions]

DOCUMENT:
{text[:12000]}"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = response.text

    def extract_section(label, content):
        try:
            start = content.index(label + ":") + len(label) + 1
            next_headers = [h + ":" for h in ["SUMMARY", "KEY DATES", "ACTION ITEMS"] if h != label]
            end = len(content)
            for h in next_headers:
                if h in content[start:]:
                    end = min(end, content.index(h, start))
            return content[start:end].strip()
        except ValueError:
            return "Not found."

    return {
        "summary": extract_section("SUMMARY", raw),
        "dates": extract_section("KEY DATES", raw),
        "actions": extract_section("ACTION ITEMS", raw),
    }

# ── Ask a question about the document ────────────────────────────────────────
def ask_question(question: str, doc_text: str, history: list) -> str:
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant answering questions about a document.
Use only information from the document to answer. Be concise and clear.

DOCUMENT:
{doc_text[:12000]}

CONVERSATION HISTORY:
{history_text}

User question: {question}

Answer:"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()

# ── Session state ─────────────────────────────────────────────────────────────
if "brief" not in st.session_state:
    st.session_state.brief = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# Document Briefer")
st.markdown("Upload a PDF and get an instant summary, key dates, and action items.")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded and uploaded.name != st.session_state.doc_name:
    st.session_state.doc_name = uploaded.name
    st.session_state.brief = None
    st.session_state.chat_history = []
    st.session_state.doc_text = ""

if uploaded:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{uploaded.name}**")
    with col2:
        analyse = st.button("Analyse →")

    if analyse or st.session_state.brief:
        if analyse:
            with st.spinner("Reading document..."):
                st.session_state.doc_text = extract_text(uploaded)
            with st.spinner("Generating brief..."):
                st.session_state.brief = summarise(st.session_state.doc_text)
                st.session_state.chat_history = []

        brief = st.session_state.brief
        if brief:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # Summary
            st.markdown('<div class="brief-card"><div class="tag">Summary</div><h4>Key Points</h4>' +
                brief["summary"].replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="brief-card"><div class="tag">Dates</div><h4>Key Dates</h4>' +
                    brief["dates"].replace("\n", "<br>") + '</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="brief-card"><div class="tag">Actions</div><h4>Action Items</h4>' +
                    brief["actions"].replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

            # Q&A section
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### Ask anything about this document")

            # Chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)

            # Question input
            with st.form("qa_form", clear_on_submit=True):
                question = st.text_input("Ask a question...", label_visibility="collapsed",
                    placeholder="e.g. What are the main risks mentioned?")
                submitted = st.form_submit_button("Ask →")

            if submitted and question.strip():
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    answer = ask_question(question, st.session_state.doc_text, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #999;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
        <div style="font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #666;">
            Drop a PDF above to get started
        </div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
            Contracts · Reports · Research papers · Meeting notes
        </div>
    </div>
    """, unsafe_allow_html=True)
