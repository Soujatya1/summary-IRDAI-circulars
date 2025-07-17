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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sidebar configuration
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
    """Initialize Azure OpenAI client"""
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

def is_footer_or_header(text):
    """Check if text is a footer or header that should be filtered out"""
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
    """Check if text is in English"""
    try:
        if is_footer_or_header(text):
            return False
        return detect(text.strip()) == "en"
    except:
        return False

def get_summary_prompt(text):
    """Generate prompt for summarization"""
    return f"""
You are a domain expert in insurance compliance and regulation. Generate a clean, concise summary of the document while maintaining the original structure.

### FORMATTING RULES:

1. **Use bold formatting (**text**) ONLY for:**
   - Main section headings (e.g., "CHAPTER I", "PRELIMINARY")
   - Numbered section titles (e.g., "1. Definitions:", "2. Objectives:")
   - Clear subheadings that are titles in the original document

2. **NEVER use bold formatting for:**
   - Definition items or explanatory text
   - Regular paragraphs or sentences
   - Bullet point content
   - Policy details or conditions

3. **Structure:**
   - Maintain the exact order of the original document
   - Use bullet points (•) for lists
   - Keep section numbering as in original
   - Use normal text for all content except clear headings

4. **For definitions:**
   - Format as: **Definitions:**
   - List each as: • Term: Explanation in normal text

5. **Summarize each section in 1-5 lines** proportional to original length

### EXAMPLES:

**1. Definitions:**
• Act: Insurance Act, 1938 (4 of 1938)
• Authority: Insurance Regulatory and Development Authority of India
• File and use: Procedure for insurers to market products after filing

**Product structure:**
All insurance products are categorized as linked or non-linked. Linked products include unit linked and index linked products.

Generate summary following these rules exactly:

{text}
"""

def summarize_text_with_langchain(text, llm):
    """Summarize text using LangChain with Azure OpenAI"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_text(text)
    summaries = []
    
    for i, chunk in enumerate(chunks, 1):
        prompt = get_summary_prompt(chunk)
        response = llm([HumanMessage(content=prompt)])
        summary = response.content.strip()
        
        # Clean up formatting issues
        summary = clean_summary_formatting(summary)
        summaries.append(summary)
    
    # Join and final cleanup
    full_summary = "\n\n".join(summaries)
    return final_cleanup(full_summary)

def clean_summary_formatting(summary):
    """Clean up formatting issues in individual summary chunks"""
    # Remove empty bold tags
    summary = re.sub(r'\*\*\s*\*\*', '', summary)
    
    # Remove bold from common non-heading phrases
    inappropriate_bold_patterns = [
        r'\*\*(the continued insurability.*?)\*\*',
        r'\*\*(based on the information.*?)\*\*',
        r'\*\*(in accordance with.*?)\*\*',
        r'\*\*(as per the.*?)\*\*',
        r'\*\*(during the.*?)\*\*'
    ]
    
    for pattern in inappropriate_bold_patterns:
        summary = re.sub(pattern, r'\1', summary, flags=re.IGNORECASE)
    
    # Remove bold from mid-sentence text
    summary = re.sub(r'(\w+)\s+\*\*(.*?)\*\*\s+(\w+)', r'\1 \2 \3', summary)
    
    # Normalize line breaks
    summary = re.sub(r'\n{3,}', '\n\n', summary)
    
    return summary

def final_cleanup(full_summary):
    """Final cleanup of the complete summary"""
    # Remove bold from text containing common connecting words
    full_summary = re.sub(
        r'\*\*([^*\n]*(?:the|and|or|in|of|to|with|by|for|on|at|from)[^*\n]*)\*\*', 
        r'\1', 
        full_summary, 
        flags=re.IGNORECASE
    )
    
    # Clean up spacing
    full_summary = re.sub(r'\n{3,}', '\n\n', full_summary)
    full_summary = re.sub(r'  +', ' ', full_summary)
    
    return full_summary

def create_pdf_styles():
    """Create PDF styles for different elements"""
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

def is_heading_line(line):
    """Determine if a line should be treated as a heading"""
    # Numbered section headings (e.g., "**1. Definitions:**")
    if re.match(r'^\*\*\d+\.\s+.*:\*\*$', line):
        return True
    
    # Chapter headings (e.g., "**CHAPTER I**", "**PRELIMINARY**")
    if re.match(r'^\*\*[A-Z\s]+\*\*$', line) and any(word in line.upper() for word in ['CHAPTER', 'PRELIMINARY', 'MISCELLANEOUS', 'SCHEDULE']):
        return True
    
    # Section headings ending with colon (e.g., "**Definitions:**")
    if re.match(r'^\*\*[^*]+:\*\*$', line):
        return True
    
    return False

def parse_markdown_to_pdf_elements(text, styles):
    """Parse markdown text and convert to PDF elements"""
    elements = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            elements.append(Spacer(1, 6))
            continue
        
        # Check if line is a heading
        if is_heading_line(line):
            heading_text = line[2:-2].strip()  # Remove ** from both ends
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        
        # Markdown headers
        elif line.startswith('###'):
            heading_text = line[3:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['subheading']))
        elif line.startswith('##'):
            heading_text = line[2:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        elif line.startswith('#'):
            heading_text = line[1:].strip()
            elements.append(Paragraph(f"<b>{heading_text}</b>", styles['heading']))
        
        # Bullet points
        elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            # Very minimal inline formatting for bullet points
            formatted_bullet = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(f"• {formatted_bullet}", styles['bullet']))
        
        # Regular text
        else:
            # Be very conservative with inline bold formatting
            if '**' in line:
                # Check if this might be misformatted content
                if any(word in line.lower() for word in ['the', 'and', 'or', 'in', 'of', 'to', 'with', 'by', 'for', 'on', 'at', 'from', 'during', 'based', 'accordance']):
                    # Remove bold formatting for likely regular text
                    clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                    elements.append(Paragraph(clean_line, styles['normal']))
                else:
                    # Allow minimal inline bold for truly emphasized terms
                    formatted_line = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line)
                    elements.append(Paragraph(formatted_line, styles['normal']))
            else:
                elements.append(Paragraph(line, styles['normal']))
    
    return elements

def generate_pdf(summary_text):
    """Generate PDF from summary text"""
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
    
    # Title
    story.append(Paragraph("<b>Document Summary</b>", styles['title']))
    story.append(Spacer(1, 20))
    
    # Content
    elements = parse_markdown_to_pdf_elements(summary_text, styles)
    story.extend(elements)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def clean_extracted_text(text):
    """Clean extracted text from PDF"""
    # Remove page markers
    text = re.sub(r'\n\n--- Page \d+ ---\n', '\n\n', text)
    text = re.sub(r'--- Page \d+ ---', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def generate_download_filename(original_filename):
    """Generate download filename based on original filename"""
    name_without_ext = os.path.splitext(original_filename)[0]
    return f"{name_without_ext}_summary.pdf"

# Initialize LLM
llm = initialize_azure_openai(azure_endpoint, api_key, deployment_name, api_version)

# Main UI
st.set_page_config(layout="wide")
st.title("📄 PDF Document Summarizer")
st.markdown("Upload a PDF document to generate a structured summary")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and llm:
    st.success("File uploaded successfully!")
    
    with st.spinner("Extracting text from PDF..."):
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
        
        with st.spinner("Generating summary..."):
            try:
                full_summary = summarize_text_with_langchain(english_text, llm)
                
                st.subheader("📋 Summary")
                st.text_area("Preview", full_summary, height=500)
                
                # Generate PDF
                pdf_file = generate_pdf(full_summary)
                download_filename = generate_download_filename(uploaded_file.name)
                
                st.download_button(
                    label="📥 Download Summary (PDF)", 
                    data=pdf_file, 
                    file_name=download_filename,
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                logger.error(f"Summarization error: {str(e)}")
    else:
        st.error("No English content found in the uploaded PDF.")

elif uploaded_file and not llm:
    st.error("Please configure Azure OpenAI settings in the sidebar.")
elif not uploaded_file:
    st.info("Please upload a PDF file to get started.")
