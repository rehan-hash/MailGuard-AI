import streamlit as st
import pickle
import re
from pypdf import PdfReader
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="MailGuard AI | Spam Defense",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------
# Custom "Expensive" CSS
# -------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Card Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.stTextArea), 
    div[data-testid="stVerticalBlock"] > div:has(div.stFileUploader) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Custom Header */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 0;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        border: none;
        height: 3.5em;
        background: linear-gradient(45deg, #00dbde 0%, #fc00ff 100%);
        color: white;
        font-weight: bold;
        transition: 0.4s;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(252, 0, 255, 0.4);
    }

    /* Radio button labels */
    .stRadio label { color: #ddd !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Logic Functions
# -------------------------
EMAIL_HEADERS = ['subject:', 'from:', 'to:', 'sent:', 'cc:']
MESSAGE_KEYWORDS = [
    'hi', 'hello', 'dear', 'thanks', 'regards', 'sincerely',
    'call', 'text', 'txt', 'reply', 'stop', 'urgent', 'free',
    'winner', 'click', 'offer', 'subscription', 'customer'
]

@st.cache_resource
def load_models():
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, cv = load_models()

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_communication(text):
    text_lower = text.lower()
    header_found = any(header in text_lower for header in EMAIL_HEADERS)
    matched_words = [word for word in MESSAGE_KEYWORDS if word in text_lower]
    return header_found or len(matched_words) >= 4

def extract_pdf_text(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_docx_text(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

# -------------------------
# UI Layout
# -------------------------
st.markdown('<p class="main-title">MAILGUARD AI</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaa; font-size:1.1rem; margin-bottom:40px;">Neural Spam Filtering System</p>', unsafe_allow_html=True)

# Two Column Dashboard
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 📥 Data Ingestion")
    option = st.radio("Detection Mode", ("Manual Text", "Digital Document"), horizontal=True)
    
    input_text = ""
    if option == "Manual Text":
        input_text = st.text_area("Secure Input Field", placeholder="Paste your message here...", height=300)
    else:
        uploaded_file = st.file_uploader("Upload Encrypted File", type=["pdf", "docx", "txt"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                raw_text = extract_pdf_text(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = extract_docx_text(uploaded_file)
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

            if is_valid_communication(raw_text):
                input_text = raw_text
                st.success("✅ Communication Format Verified")
            else:
                st.error("🚨 Validation Failed: Irrelevant Content Detected")
                input_text = ""

with right_col:
    st.markdown("### 🧠 Analysis Engine")
    if st.button("Initialize Neural Scan"):
        if not input_text.strip():
            st.warning("Waiting for valid input data...")
        else:
            with st.spinner("Processing through Neural Layers..."):
                time.sleep(1.2) # Adds to the "expensive" AI feel
                
                cleaned = transform_text(input_text)
                vector_input = cv.transform([cleaned])
                result = model.predict(vector_input)[0]
                proba = model.predict_proba(vector_input)[0] # Confidence

                # Result Display
                st.markdown("---")
                confidence = max(proba) * 100
                
                if str(result).lower() == "spam" or result == 1:
                    st.error(f"## THREAT DETECTED: SPAM")
                    st.metric("Risk Level", f"{confidence:.2f}%", delta="High Alert")
                else:
                    st.success(f"## STATUS: SECURE (HAM)")
                    st.metric("Safety Score", f"{confidence:.2f}%", delta_color="normal")

                # Word Cloud inside the analysis column
                st.subheader("Word Frequency Visualization")
                wc = WordCloud(width=600, height=350, background_color=None, mode="RGBA", colormap="magma").generate(cleaned)
                fig, ax = plt.subplots(facecolor='none')
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)
