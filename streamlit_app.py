import streamlit as st
import pdfplumber
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from io import BytesIO
import re
import logging
from datetime import datetime

# NEW IMPORTS FOR DOCX FUNCTIONALITY
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.shared import OxmlElement, qn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with st.sidebar:
    st.header("🔧 Configuration")
    
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

st.set_page_config(layout="wide")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

def is_footer_or_header(text):
    text = text.strip().upper()
    
    if any(pattern in text for pattern in [
        'INSURANCE REGULATORY AND DEVELOPMENT AUTHORITY OF INDIA',
        'NOTIFICATION',
        'REGULATIONS, 2024',
        'F. NO. IRDAI',
        'IRDAI/REG/'
    ]):
        return False
    
    footer_patterns = [
        r'THE GAZETTE OF INDIA.*EXTRAORDINARY.*PART.*SEC.*$',
        r'^\d+\s+THE GAZETTE OF INDIA\s*$',
        r'PART\s+III.*SEC\.\s*$',
        r'PAGE\s+\d+\s*$',
        r'^\d+\s*$',
        r'^[IVX]+\s*$',
    ]
    
    for pattern in footer_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def is_english(text):
    try:
        if is_footer_or_header(text):
            return False
        
        text_stripped = text.strip()
        
        if any(pattern in text_stripped.upper() for pattern in [
            'INSURANCE REGULATORY AND DEVELOPMENT AUTHORITY',
            'NOTIFICATION',
            'F. NO.', 'F.NO.', 'FILE NO.',
            'IRDAI/REG/',
            'REGULATIONS, 2024',
            'HYDERABAD',
            'IN EXERCISE OF'
        ]):
            return True
        
        return detect(text_stripped) == "en"
    except:
        return False

def get_summary_prompt(text):
    return f"""
You are acting as a **Senior Legal Analyst** and Regulatory Compliance Officer specializing in IRDAI, UIDAI, and eGazette circulars.
 
Your task is to generate a **legally precise, clause-preserving, structure-aligned summary** of the in-put regulatory document. Your summary will be reviewed for legal compliance, so accuracy is critical.
 
---
 
### LEGAL SUMMARIZATION RULES
 
**1. STRUCTURE PRESERVATION (Strict Order):**
- Retain **original structure**, including:
  - Section headers, subheaders, and sub-subheaders
  - Clause numbers (e.g., 3.2.1, a), b), c))
  - Bullet f-ormats, indentation levels
- Do not reorder, combine, or rename any sections or sub-sections.
 
**2. CLAUSE-BY-CLAUSE SUMMARIZATION (NO MERGING):**
- **Summarize one clause per bullet/sentence only.**
- If a clause is broken across lines or pages, **treat it as a single clause**.
- Do not combine adjacent points even if they seem similar.
 
**3. PRESERVE LEGAL PHRASES & CAUSALITY TRIGGERS:**
- Never skip or simplify phrases like:
  - **"unless"**, **"until"**, **"after"**, **"shall"**, **"subject to"**, **"provided that"**
- These are **legally binding conditions** and must be **retained with their meaning intact**.
 
**4. DEFINITIONS & EXPLANATORY SECTIONS:**
- If the section contains **definitions** or cl-assifications:
  - List each term separately using this structure:  
    - *Definition: Revival Period* – A policy may be revived within…
  - **Do not merge multiple definitions** into one block.
 
**5. COMMITTEES, PANELS, AUTHORITIES (EXACT NAMES):**
- Retain **every mention of committees and positions verbatim**.
- Never shorten or generalize:
  - "Product Management Committee (PMC)" not "product committee"
  - "Chief Compliance Officer" not "Compliance Head"
  - "Member – Life", "Key Management Persons (KMPs)", "Appointed Actuary", etc.
- Repeat full names every time they appear, even if already mentioned before.
 
**6. TABLES – PRESERVE IN FULL:**
- Summarize **column-by-column**, row-by-row.
- Do not omit any row (e.g., Discontinuance Charges for all policy years).
- If summarizing:  
  - *Table: Discontinuance Charges*  
    - Year 1: Lower of 2% or ₹3,000  
    - Year 2: Lower of 1.5% or ₹2,000  
    …
 
**7. NUMERIC LIMITS & ABBREVIATIONS:**
- Maintain correct expressions like:
  - Rs. 1,000/- (not "Rs 1000")
  - "AP or FV, whichever is lower" (do not paraphrase this)
 
**8. HISTORICAL & AUTHORITY CLAUSES:**
- Include all clauses like:
  - "Repeal and Savings"
  - "Authority's power to issue clarifications"
- Do **not skip final sections** even if repetitive.
 
**9. SIGNATURE, SEAL, PUBLICATION TEXT – OMIT:**
- Strictly exclude:
  - Signature blocks (e.g., "Debasish Panda, Chairperson")
  - Digital signing metadata ("Digitally signed by Manoj Kumar Verma")
  - Footer/publication notices ("Uploaded by Dte. of Printing…")
 
**10. LINE BREAKS & ORPHAN HANDLING:**
- Do not treat broken lines (from PDF f-ormatting) as new clauses.
- Ensure a single sentence broken across lines is still summarized as one thought.
 
---
 
### OUTPUT FORMAT:
- Use clean plain text.
- Preserve order and hierarchy (e.g., 1 → a → i).
- Do not use Markdown f-ormatting (no **bold**, `code`, or extra spacing).
- Do not invent or rename headings.
 
---
 
### SUMMARY LENGTH RULE:
- Ensure total summary length is approx. **50% of English content pages).
 
---
 
Now begin the **section-wise clause-preserving summary** of the following legal document:
--------------------
{text}
"""

def summarize_text_with_langchain(text):
    st.subheader("📋 Text Preview - Before LLM Processing")
    st.info(f"**Total characters:** {len(text)}")
    
    preview_text = text[:2000]
    if len(text) > 2000:
        preview_text += "\n\n... [Text truncated for preview. Full text will be processed by LLM]"
    
    st.text_area("Extracted Text Preview", preview_text, height=300, disabled=True)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    st.info(f"**Text will be split into {len(chunks)} chunks** for processing")
    
    with st.expander("📄 View Chunk Previews"):
        for i, chunk in enumerate(chunks, 1):
            st.write(f"**Chunk {i}** (Length: {len(chunk)} characters)")
            chunk_preview = chunk[:500] + "..." if len(chunk) > 500 else chunk
            st.text_area(f"Chunk {i} Preview", chunk_preview, height=150, disabled=True, key=f"chunk_{i}")
    
    summaries = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks, 1):
        st.write(f"Processing chunk {i}/{len(chunks)}...")
        
        with st.expander(f"🔍 LLM Input for Chunk {i}"):
            prompt = get_summary_prompt(chunk)
            st.text_area(f"Full Prompt for Chunk {i}", prompt, height=200, disabled=True, key=f"prompt_{i}")
        
        response = llm([HumanMessage(content=prompt)])
        summaries.append(response.content.strip())
        
        progress_bar.progress(i / len(chunks))
    
    progress_bar.empty()
    return "\n\n".join(summaries)

def create_pdf_with_fpdf(summary_text, original_filename=None):
    """Generate PDF using FPDF - Simple and fast"""
    
    class PDFDoc(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Regulatory Document Summary', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDFDoc()
    pdf.add_page()
    
    # Add metadata
    pdf.set_font('Arial', '', 10)
    if original_filename:
        pdf.cell(0, 8, f'Original Document: {original_filename}', 0, 1)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 0, 1)
    pdf.ln(10)
    
    # Process content
    lines = summary_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
            
        if line.upper().startswith(('CHAPTER', 'SECTION', 'PART')):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, line, 0, 1)
        elif line.startswith('*Definition:'):
            pdf.set_font('Arial', 'B', 10)
            wrapped = textwrap.wrap(line, width=80)
            for wrapped_line in wrapped:
                pdf.cell(0, 6, wrapped_line, 0, 1)
        else:
            pdf.set_font('Arial', '', 10)
            wrapped = textwrap.wrap(line, width=80)
            for wrapped_line in wrapped:
                pdf.cell(0, 6, wrapped_line, 0, 1)
    
    return pdf.output(dest='S').encode('latin1')

if uploaded_file is not None and llm is not None:
    st.header("📄 PDF Document Summarizer")
    
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        text_content = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    for line in lines:
                        if line.strip() and is_english(line):
                            text_content += line + '\n'
    
    if text_content.strip():
        st.success(f"✅ Successfully extracted {len(text_content)} characters of English text")
        
        # Generate summary
        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Generating AI summary... This may take a few minutes."):
                try:
                    summary = summarize_text_with_langchain(text_content)
                    
                    # Display the summary
                    st.subheader("📝 Generated Summary")
                    st.text_area("Summary", summary, height=400, disabled=True)
                    
                    # Create download options
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col2:
                        st.write("**PDF Format**")
                        
                        # PDF method selection
                        pdf_method = st.selectbox(
                            "PDF Method:",
                            ["WeasyPrint (Professional)", "FPDF (Simple)"],
                            key="pdf_method"
                        )
                        
                        if st.button("Generate PDF", key="gen_pdf"):
                            with st.spinner("Creating PDF..."):
                                try:
                                    if pdf_method == "WeasyPrint (Professional)":
                                        pdf_bytes = create_pdf_with_weasyprint(summary, uploaded_file.name)
                                    else:
                                        pdf_bytes = create_pdf_with_fpdf(summary, uploaded_file.name)
                                    
                                    pdf_filename = f"summary_{uploaded_file.name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                    
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=pdf_bytes,
                                        file_name=pdf_filename,
                                        mime="application/pdf",
                                        key="download_pdf"
                                    )
                                    
                                    st.success("✅ PDF generated successfully!")
                                    
                                except Exception as e:
                                    st.error(f"PDF generation error: {str(e)}")
                                    st.info("Try installing: pip install weasyprint")
                    
                    st.success("✅ Summary generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    logger.error(f"Summary generation error: {str(e)}")
    else:
        st.error("❌ No English text could be extracted from the PDF")
