import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from docx import Document
from io import BytesIO
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with st.sidebar:
    st.header("🔧 Configuration")
    
    # Azure OpenAI Configuration
    azure_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        placeholder="https://your-resource.openai.azure.com/",
        help="Your Azure OpenAI service endpoint"
    )
    
    api_key = st.text_input(
        "Azure OpenAI API Key",
        type="password",
        placeholder="Enter your API key",
        help="Your Azure OpenAI API key"
    )
    
    deployment_name = st.text_input(
        "Deployment Name",
        placeholder="gpt-35-turbo",
        help="Name of your deployed model"
    )
    
    api_version = st.selectbox(
        "API Version",
        ["2025-01-01-preview"],
        index=0
    )

def initialize_azure_openai(endpoint, api_key, deployment_name, api_version):
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            api_version=api_version,
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {str(e)}")
        return None

llm = initialize_azure_openai(azure_endpoint, api_key, deployment_name, api_version)

# Streamlit UI
st.set_page_config(layout="wide")
 
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

def is_footer_or_header(text):
    """Check if text is likely a footer, header, or unwanted metadata"""
    text_lower = text.lower().strip()
    
    # Common footer/header patterns
    footer_patterns = [
        r'^\d+\s+the\s+gazette\s+of\s+india',  # "20 THE GAZETTE OF INDIA"
        r'part\s+iii[—\-]sec',  # "PART III—SEC"
        r'^\d+\s*$',  # Just page numbers
        r'^\[\s*part\s+[iv]+',  # "[PART III"
        r'extraordinary\s*\]?$',  # "EXTRAORDINARY]"
        r'^\d{1,3}\s+[a-z\s]+gazette',  # Page number + gazette
        r'^page\s+\d+',  # "Page 1"
        r'^\d+\s+[a-z\s]*gazette[a-z\s]*extraordinary',  # Various gazette patterns
    ]
    
    for pattern in footer_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check if it's too short to be meaningful content
    if len(text.strip()) < 10:
        return True
        
    # Check if it's mostly numbers and special characters
    if re.match(r'^[\d\s\[\]\-—.:]+$', text.strip()):
        return True
        
    return False

def is_meaningful_english(text):
    """Check if text contains meaningful English content"""
    if not text or len(text.strip()) < 3:
        return False
        
    # Skip if it's a footer/header
    if is_footer_or_header(text):
        return False
    
    try:
        # Use language detection but be more lenient
        detected_lang = detect(text.strip())
        if detected_lang != "en":
            return False
    except:
        # If detection fails, use other heuristics
        pass
    
    # Check if it contains meaningful English words
    english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(english_words) < 2:  # At least 2 meaningful words
        return False
    
    # Check ratio of alphabetic characters
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0 and alpha_chars / total_chars < 0.3:
        return False
    
    return True

def clean_and_extract_text(text):
    """Clean and extract meaningful text from a page"""
    if not text:
        return ""
    
    # Split into lines first
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip obvious footers/headers
        if is_footer_or_header(line):
            continue
            
        # Clean up the line
        # Remove excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        # Only keep lines with meaningful content
        if is_meaningful_english(line):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def get_summary_prompt(text):
    return f"""
You are a domain expert in insurance compliance and regulation.
 
Your task is to generate a **clean, concise, section-wise summary** of the input document while preserving the **original structure and flow** of the document.
 
---
 
### Mandatory Summarization Rules:
 
1. **Follow the original structure strictly** — maintain the same order of:
   - Section headings
   - Subheadings
   - Bullet points
   - Tables
   - Date-wise event history
   - UIDAI / IRDAI / eGazette circulars
 
2. **Do NOT rename or reformat section titles** — retain the exact headings from the original file.
 
3. **Each section should be summarized in 1–5 lines**, proportional to its original length:
   - Keep it brief, but **do not omit the core message**.
   - Avoid generalizations or overly descriptive rewriting.
 
4. If a section contains **definitions**, summarize them line by line (e.g., Definition A: …).
 
5. If the section contains **tabular data**, preserve **column-wise details**:
   - Include every row and column in a concise bullet or structured format.
   - Do not merge or generalize rows — maintain data fidelity.
 
6. If a section contains **violations, fines, or penalties**, mention each item clearly:
   - List out exact violation titles and actions taken or proposed.
 
7. For **date-wise circulars or history**, ensure that:
   - **No dates are skipped or merged.**
   - Maintain **chronological order**.
   - Mention full references such as "IRDAI Circular dated 12-May-2022".
 
---
 
### Output Format:
 
- Follow the exact **order and structure** of the input file.
- Do **not invent new headings** or sections.
- Avoid decorative formatting, markdown, or unnecessary bolding — use **clean plain text**.
 
---
 
Now, generate a section-wise structured summary of the document below:
--------------------
{text}
"""
 
def summarize_text_with_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    summaries = []
 
    for i, chunk in enumerate(chunks, 1):
        prompt = get_summary_prompt(chunk)
        response = llm([HumanMessage(content=prompt)])
        summaries.append(response.content.strip())
 
    return "\n\n".join(summaries)
 
def generate_docx(summary_text):
    doc = Document()
    doc.add_heading("Summary", level=1)
    for para in summary_text.split("\n\n"):
        doc.add_paragraph(para.strip())
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
 
if uploaded_file:
    st.success("File uploaded successfully!")
    english_text = ""
 
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            raw_text = page.extract_text()
            if raw_text:
                # Clean and extract meaningful text
                cleaned_text = clean_and_extract_text(raw_text)
                
                if cleaned_text.strip():
                    english_text += f"\n\n--- Page {i} ---\n{cleaned_text}"
                else:
                    st.info(f"Page {i}: No meaningful content after filtering")
 
    if english_text.strip():
        # Show preview of extracted text
        with st.expander("Preview Extracted Text"):
            st.text_area("Extracted Content", english_text[:2000] + "..." if len(english_text) > 2000 else english_text, height=300)
        
        with st.spinner("Summarizing content..."):
            full_summary = summarize_text_with_langchain(english_text)
 
        st.subheader("Summary")
        st.text_area("Preview", full_summary, height=500)
 
        docx_file = generate_docx(full_summary)
        st.download_button("Download Summary (DOCX)", data=docx_file, file_name="Summary.docx")
    else:
        st.error("No meaningful content found in the uploaded PDF after filtering.")
