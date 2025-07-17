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
from langdetect.lang_detect_exception import LangDetectException
import re

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
 
def extract_english_text(text):
    try:
        sentences = re.split(r'[.!?]+', text)
        english_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                try:
                    lang = langdetect.detect(sentence)
                    if lang == 'en':
                        english_sentences.append(sentence)
                except LangDetectException:
                    if re.search(r'\b(the|and|or|of|to|in|for|with|by|from|at|is|are|was|were)\b', sentence.lower()):
                        english_sentences.append(sentence)
        
        return'. '.join(english_sentences) + '.'
    
    except Exception as e:
        st.warning(f"Language detection error: {e}. Using original text.")
        return text

def get_summary_prompt(text):
    return f"""
You are a domain expert in insurance compliance and regulation.
 
Your task is to generate a **clean, concise, section-wise summary** of the input  document while preserving the **original structure and flow** of the document.
 
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
            text = page.extract_text()
            if text and extract_english_text(text):
                english_text += f"\n\n--- Page {i} ---\n{text}"
            else:
                st.warning(f"Skipping non-English Page {i}")
 
    if english_text.strip():
        with st.spinner("Summarizing English content..."):
            full_summary = summarize_text_with_langchain(english_text)
 
        st.subheader("Summary")
        st.text_area("Preview", full_summary, height=500)
 
        docx_file = generate_docx(full_summary)
        st.download_button("Download Summary (DOCX)", data=docx_file, file_name="Summary.docx")
    else:
        st.error("No English content found in the uploaded PDF.")
