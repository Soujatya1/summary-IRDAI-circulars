import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
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

# Streamlit UI
st.set_page_config(layout="wide")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

def is_footer_or_header(text):
    text = text.strip().upper()
    
    footer_patterns = [
        r'THE GAZETTE OF INDIA.*EXTRAORDINARY.*PART.*SEC',
        r'^\d+\s+THE GAZETTE OF INDIA',
        r'PART\s+III.*SEC\.',
        r'EXTRAORDINARY.*PART.*III',
        r'^\d+\s+.*GAZETTE.*INDIA',
        r'PAGE\s+\d+',
        r'^\d+\s*$',
        r'^[IVX]+\s*$',
        r'CONTINUED\s+ON\s+NEXT\s+PAGE',
        r'^\d+\s+OF\s+\d+$',
    ]
    
    for pattern in footer_patterns:
        if re.search(pattern, text):
            return True
    
    if len(text.split()) <= 2 and any(word in text for word in ['GAZETTE', 'INDIA', 'PART', 'SEC']):
        return True
    
    return False

def is_english(text):
    try:
        if is_footer_or_header(text):
            return False
        return detect(text.strip()) == "en"
    except:
        return False

def get_summary_prompt(text):
    return f"""
You are a domain expert in insurance compliance and regulation. Your task is to generate a **clean, concise, section-wise summary** of the input document while preserving the **original structure and flow** of the document.

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

### Formatting Guidelines:
- Use **bold** only for main headings and subheadings
- Use bullet points (•) for lists
- Do NOT bold entire paragraphs or sentences that are not headings
- Maintain proper paragraph structure
- Use numbered headings like "1. Definitions:" when present in original

---

### Output Format:
- Follow the exact **order and structure** of the input file.
- Do **not invent new headings** or sections.
- Please return headers and sub-headers in bold format using **text**
- Regular content should be in normal text without bold formatting

---

Now, generate a section-wise structured summary of the document below:

--------------------
{text}
"""

def summarize_text_with_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_text(text)
    summaries = []
    
    for i, chunk in enumerate(chunks, 1):
        prompt = get_summary_prompt(chunk)
        response = llm([HumanMessage(content=prompt)])
        summary = response.content.strip()
        
        # Clean up any formatting issues in the summary
        summary = re.sub(r'\*\*\s*\*\*', '', summary)  # Remove empty bold tags
        summary = re.sub(r'\n{3,}', '\n\n', summary)  # Normalize line breaks
        
        summaries.append(summary)
    
    # Join summaries and do final cleanup
    full_summary = "\n\n".join(summaries)
    
    # Post-process to fix any remaining formatting issues
    full_summary = re.sub(r'\*\*([^*]+)\*\*\s*\*\*([^*]+)\*\*', r'**\1 \2**', full_summary)  # Merge adjacent bold
    full_summary = re.sub(r'\n{3,}', '\n\n', full_summary)  # Normalize line breaks
    
    return full_summary

def create_pdf_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor='black'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor='black'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4,
        alignment=TA_LEFT,
        textColor='black'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
        textColor='black'
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=20,
        alignment=TA_JUSTIFY,
        textColor='black'
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'normal': normal_style,
        'bullet': bullet_style
    }

def parse_markdown_to_pdf_elements(text, styles):
    elements = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 6))
            continue
        
        # Check for numbered headings (e.g., "1. Definitions:")
        if re.match(r'^\d+\.\s+.*:', line):
            elements.append(Paragraph(f"<b>{line}</b>", styles['heading']))
        # Check for lines that start and end with ** (full line bold)
        elif line.startswith('**') and line.endswith('**') and line.count('**') == 2:
            heading_text = line[2:-2].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        # Check for markdown headers
        elif line.startswith('###'):
            heading_text = line[3:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['subheading']))
        elif line.startswith('##'):
            heading_text = line[2:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        elif line.startswith('#'):
            heading_text = line[1:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        # Check for bullet points
        elif line.startswith('- ') or line.startswith('• '):
            bullet_text = line[2:].strip()
            # Handle inline bold formatting within bullet points
            formatted_bullet = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(f"• {formatted_bullet}", styles['bullet']))
        elif line.startswith('* '):
            bullet_text = line[2:].strip()
            formatted_bullet = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(f"• {formatted_bullet}", styles['bullet']))
        # Regular text with inline formatting
        else:
            # Handle inline bold formatting more carefully
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            elements.append(Paragraph(formatted_line, styles['normal']))
    
    return elements

def generate_pdf(summary_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = create_pdf_styles()
    
    story = []
    
    story.append(Paragraph("<b>Document Summary</b>", styles['title']))
    story.append(Spacer(1, 20))
    
    elements = parse_markdown_to_pdf_elements(summary_text, styles)
    story.extend(elements)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def clean_extracted_text(text):
    text = re.sub(r'\n\n--- Page \d+ ---\n', '\n\n', text)
    text = re.sub(r'--- Page \d+ ---', '', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def generate_download_filename(original_filename):
    """Generate a download filename based on the original filename"""
    # Remove the .pdf extension if present
    name_without_ext = os.path.splitext(original_filename)[0]
    # Add _summary suffix and .pdf extension
    return f"{name_without_ext}_summary.pdf"

if uploaded_file:
    st.success("File uploaded successfully!")
    english_text = ""
    
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
                english_sentences = [s for s in sentences if is_english(s)]
                
                if english_sentences:
                    english_text += f"\n\n--- Page {i} ---\n" + ".".join(english_sentences) + "."
                else:
                    st.warning(f"Skipping non-English Page {i}")
    
    if english_text.strip():
        english_text = clean_extracted_text(english_text)
        with st.spinner("Summarizing English content..."):
            full_summary = summarize_text_with_langchain(english_text)
        
        st.subheader("Summary")
        st.text_area("Preview", full_summary, height=500)
        
        pdf_file = generate_pdf(full_summary)
        
        # Generate download filename based on source document name
        download_filename = generate_download_filename(uploaded_file.name)
        
        st.download_button(
            "Download Summary (PDF)", 
            data=pdf_file, 
            file_name=download_filename,
            mime="application/pdf"
        )
    else:
        st.error("No English content found in the uploaded PDF.")
