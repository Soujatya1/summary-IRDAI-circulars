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
  - **“unless”**, **“until”**, **“after”**, **“shall”**, **“subject to”**, **“provided that”**
- These are **legally binding conditions** and must be **retained with their meaning intact**.
 
**4. DEFINITIONS & EXPLANATORY SECTIONS:**
- If the section contains **definitions** or cl-assifications:
  - List each term separately using this structure:  
    - *Definition: Revival Period* – A policy may be revived within…
  - **Do not merge multiple definitions** into one block.
 
**5. COMMITTEES, PANELS, AUTHORITIES (EXACT NAMES):**
- Retain **every mention of committees and positions verbatim**.
- Never shorten or generalize:
  - “Product Management Committee (PMC)” not “product committee”
  - “Chief Compliance Officer” not “Compliance Head”
  - “Member – Life”, “Key Management Persons (KMPs)”, “Appointed Actuary”, etc.
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
  - Rs. 1,000/- (not “Rs 1000”)
  - “AP or FV, whichever is lower” (do not paraphrase this)
 
**8. HISTORICAL & AUTHORITY CLAUSES:**
- Include all clauses like:
  - “Repeal and Savings”
  - “Authority’s power to issue clarifications”
- Do **not skip final sections** even if repetitive.
 
**9. SIGNATURE, SEAL, PUBLICATION TEXT – OMIT:**
- Strictly exclude:
  - Signature blocks (e.g., “Debasish Panda, Chairperson”)
  - Digital signing metadata (“Digitally signed by Manoj Kumar Verma”)
  - Footer/publication notices (“Uploaded by Dte. of Printing…”)
 
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

def create_pdf_styles():
    """Enhanced PDF styles with better spacing and hierarchy"""
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_CENTER,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    minor_heading_style = ParagraphStyle(
        'CustomMinorHeading',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=5,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=4,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=0,
        rightIndent=0
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        leftIndent=20,
        bulletIndent=10,
        alignment=TA_LEFT,
        textColor='black'
    )
    
    nested_bullet_style = ParagraphStyle(
        'CustomNestedBullet',
        parent=styles['Normal'],
        fontSize=9,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=40,
        bulletIndent=30,
        alignment=TA_LEFT,
        textColor='black'
    )
    
    definition_style = ParagraphStyle(
        'CustomDefinition',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=15,
        alignment=TA_JUSTIFY,
        textColor='black'
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'minor_heading': minor_heading_style,
        'normal': normal_style,
        'bullet': bullet_style,
        'nested_bullet': nested_bullet_style,
        'definition': definition_style
    }

def detect_text_type(line):
    """Improved text type detection with priority order"""
    line_stripped = line.strip()
    
    # Skip empty lines
    if not line_stripped:
        return 'empty'
    
    # Check for numbered sections/clauses (highest priority)
    if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', line_stripped):
        return 'numbered_heading'
    
    # Check for lettered sections (a), b), etc.)
    if re.match(r'^[a-z]\)\s+', line_stripped) or re.match(r'^\([a-z]\)\s+', line_stripped):
        return 'lettered_bullet'
        
    # Check for roman numerals
    if re.match(r'^[ivx]+\)\s+', line_stripped, re.IGNORECASE) or re.match(r'^\([ivx]+\)\s+', line_stripped, re.IGNORECASE):
        return 'roman_bullet'
    
    # Check for explicit markdown headers
    if line_stripped.startswith('###'):
        return 'h3'
    elif line_stripped.startswith('##'):
        return 'h2'
    elif line_stripped.startswith('#'):
        return 'h1'
    
    # Check for bold headings **Text**
    if re.match(r'^\*\*[^*]+\*\*:?\s*$', line_stripped):
        return 'bold_heading'
    
    # Check for ALL CAPS headings (likely section headers)
    if line_stripped.isupper() and len(line_stripped.split()) <= 10 and not line_stripped.endswith('.'):
        return 'caps_heading'
    
    # Check for Title Case headings ending with colon
    if line_stripped.endswith(':') and line_stripped[:-1].istitle() and len(line_stripped.split()) <= 8:
        return 'colon_heading'
    
    # Check for regular bullets
    if line_stripped.startswith('- ') or line_stripped.startswith('• ') or line_stripped.startswith('* '):
        return 'bullet'
    
    # Check for numbered lists
    if re.match(r'^\d+\.\s+', line_stripped):
        return 'numbered_bullet'
    
    # Check for definitions (Definition: Term - description)
    if re.match(r'^[*_]?Definition[*_]?:?\s+[A-Z]', line_stripped, re.IGNORECASE):
        return 'definition'
    
    # Check for table-like content
    if '|' in line_stripped or re.search(r'\s{3,}', line_stripped):
        return 'table_content'
    
    # Default to normal paragraph
    return 'normal'

def parse_markdown_to_pdf_elements(text, styles):
    """Improved parsing with better text type detection and formatting"""
    elements = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.rstrip()  # Keep leading spaces for indentation detection
        text_type = detect_text_type(line)
        
        if text_type == 'empty':
            elements.append(Spacer(1, 6))
            continue
        
        # Clean the line for processing
        clean_line = line.strip()
        
        # Handle different text types
        if text_type == 'h1':
            heading_text = clean_line[1:].strip()
            if heading_text:
                elements.append(Paragraph(heading_text, styles['heading']))
                
        elif text_type == 'h2':
            heading_text = clean_line[2:].strip()
            if heading_text:
                elements.append(Paragraph(heading_text, styles['subheading']))
                
        elif text_type == 'h3':
            heading_text = clean_line[3:].strip()
            if heading_text:
                elements.append(Paragraph(heading_text, styles['minor_heading']))
                
        elif text_type == 'numbered_heading':
            # For numbered sections like "3.2.1 Policy Revival"
            elements.append(Paragraph(clean_line, styles['subheading']))
            
        elif text_type == 'bold_heading':
            # Remove ** formatting and use as heading
            heading_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_line).strip(':')
            elements.append(Paragraph(heading_text, styles['subheading']))
            
        elif text_type == 'caps_heading':
            elements.append(Paragraph(clean_line, styles['heading']))
            
        elif text_type == 'colon_heading':
            heading_text = clean_line.rstrip(':')
            elements.append(Paragraph(heading_text, styles['minor_heading']))
            
        elif text_type in ['bullet', 'lettered_bullet', 'roman_bullet', 'numbered_bullet']:
            # Handle different bullet types
            if text_type == 'bullet':
                bullet_text = clean_line[2:].strip()  # Remove '- ' or '• '
                bullet_symbol = "•"
            elif text_type == 'lettered_bullet':
                bullet_text = re.sub(r'^[a-z]\)\s+', '', clean_line) or re.sub(r'^\([a-z]\)\s+', '', clean_line)
                bullet_symbol = "◦"
            elif text_type == 'roman_bullet':
                bullet_text = re.sub(r'^[ivx]+\)\s+', '', clean_line, flags=re.IGNORECASE) or re.sub(r'^\([ivx]+\)\s+', '', clean_line, flags=re.IGNORECASE)
                bullet_symbol = "▪"
            else:  # numbered_bullet
                bullet_text = re.sub(r'^\d+\.\s+', '', clean_line)
                bullet_symbol = "•"
            
            # Handle bold text within bullets
            if '**' in bullet_text:
                bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            
            # Determine indentation level
            indent_level = len(line) - len(line.lstrip())
            if indent_level > 10 or text_type in ['lettered_bullet', 'roman_bullet']:
                style_to_use = styles['nested_bullet']
            else:
                style_to_use = styles['bullet']
                
            elements.append(Paragraph(f"{bullet_symbol} {bullet_text}", style_to_use))
            
        elif text_type == 'definition':
            # Special formatting for definitions
            if '**' in clean_line:
                definition_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            else:
                definition_text = clean_line
            elements.append(Paragraph(definition_text, styles['definition']))
            
        elif text_type == 'table_content':
            # Simple table handling - could be enhanced further
            elements.append(Paragraph(clean_line, styles['normal']))
            
        else:  # normal text
            # Handle bold text formatting
            if '**' in clean_line:
                formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            else:
                formatted_text = clean_line
                
            elements.append(Paragraph(formatted_text, styles['normal']))
    
    return elements

def generate_pdf(summary_text):
    """Enhanced PDF generation with better page handling"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=60,
        leftMargin=60,
        topMargin=60,
        bottomMargin=60
    )
    
    styles = create_pdf_styles()
    
    story = []
    
    # Add title if the summary starts with a clear title
    lines = summary_text.split('\n')
    first_significant_line = next((line.strip() for line in lines if line.strip()), "Document Summary")
    
    # If first line looks like a title, use it
    if (first_significant_line.isupper() or 
        'NOTIFICATION' in first_significant_line.upper() or 
        'REGULATION' in first_significant_line.upper()):
        story.append(Paragraph(first_significant_line, styles['title']))
        story.append(Spacer(1, 20))
        # Remove the title from the main text
        summary_text = '\n'.join(lines[1:])
    
    # Parse and add main content
    elements = parse_markdown_to_pdf_elements(summary_text, styles)
    story.extend(elements)
    
    # Build the PDF
    try:
        doc.build(story)
    except Exception as e:
        # Fallback: if there's an issue, create a simpler version
        story = [Paragraph("Document Summary", styles['title']), Spacer(1, 20)]
        # Split into simple paragraphs
        for line in summary_text.split('\n'):
            if line.strip():
                story.append(Paragraph(line.strip(), styles['normal']))
            else:
                story.append(Spacer(1, 6))
        doc.build(story)
    
    buffer.seek(0)
    return buffer

def clean_extracted_text(text):
    text = re.sub(r'\n\n--- Page \d+ ---\n', '\n\n', text)
    text = re.sub(r'--- Page \d+ ---', '', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def generate_download_filename(original_filename):
    name_without_ext = os.path.splitext(original_filename)[0]
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
        
        download_filename = generate_download_filename(uploaded_file.name)
        
        st.download_button(
            "Download Summary (PDF)", 
            data=pdf_file, 
            file_name=download_filename,
            mime="application/pdf"
        )
    else:
        st.error("No English content found in the uploaded PDF.")
