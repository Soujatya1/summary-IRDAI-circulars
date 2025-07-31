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

def create_pdf_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=24,
        alignment=TA_CENTER,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    # Main heading (Level 1)
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=18,
        spaceAfter=8,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=0
    )
    
    # Sub heading (Level 2)
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=12
    )
    
    # Minor heading (Level 3)
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=4,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=24
    )
    
    # Sub-minor heading (Level 4)
    heading4_style = ParagraphStyle(
        'CustomHeading4',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=3,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=36
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=0
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=24,
        alignment=TA_JUSTIFY,
        textColor='black'
    )
    
    # Indented content styles for different levels
    content1_style = ParagraphStyle(
        'CustomContent1',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=12
    )
    
    content2_style = ParagraphStyle(
        'CustomContent2',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=24
    )
    
    content3_style = ParagraphStyle(
        'CustomContent3',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=36
    )
    
    return {
        'title': title_style,
        'heading1': heading1_style,
        'heading2': heading2_style,
        'heading3': heading3_style,
        'heading4': heading4_style,
        'normal': normal_style,
        'bullet': bullet_style,
        'content1': content1_style,
        'content2': content2_style,
        'content3': content3_style
    }

def detect_header_level_and_type(line):
    """
    Detect if a line is a header and return its level and type
    Returns: (is_header, level, header_type, clean_text)
    """
    line = line.strip()
    if not line:
        return False, 0, None, line
    
    # Pattern 1: Numbered headers (1., 2., 1.1, 1.2, etc.)
    numbered_pattern = r'^(\d+(?:\.\d+)*)\.\s+(.+)$'
    match = re.match(numbered_pattern, line)
    if match:
        number_part = match.group(1)
        text_part = match.group(2).strip()
        level = number_part.count('.') + 1
        return True, level, 'numbered', f"<b>{number_part}. {text_part}</b>"
    
    # Pattern 2: Roman numerals (I., II., III., etc.)
    roman_pattern = r'^([IVX]+)\.\s+(.+)$'
    match = re.match(roman_pattern, line)
    if match:
        roman_part = match.group(1)
        text_part = match.group(2).strip()
        return True, 1, 'roman', f"<b>{roman_part}. {text_part}</b>"
    
    # Pattern 3: Letter headers (a), b), c), etc.)
    letter_pattern = r'^([a-z])\)\s+(.+)$'
    match = re.match(letter_pattern, line)
    if match:
        letter_part = match.group(1)
        text_part = match.group(2).strip()
        return True, 3, 'letter', f"<b>{letter_part}) {text_part}</b>"
    
    # Pattern 4: Parenthetical numbers ((1), (2), etc.)
    paren_pattern = r'^\((\d+)\)\s+(.+)$'
    match = re.match(paren_pattern, line)
    if match:
        num_part = match.group(1)
        text_part = match.group(2).strip()
        return True, 4, 'parenthetical', f"<b>({num_part}) {text_part}</b>"
    
    # Pattern 5: All caps lines (likely headers)
    if line.isupper() and len(line.split()) <= 10 and not line.endswith('.'):
        return True, 1, 'caps', f"<b>{line}</b>"
    
    # Pattern 6: Lines ending with colon (section headers)
    if line.endswith(':') and len(line.split()) <= 8:
        clean_text = line[:-1].strip()
        return True, 2, 'colon', f"<b>{clean_text}:</b>"
    
    # Pattern 7: Bold text patterns (**text**)
    bold_pattern = r'^\*\*(.+?)\*\*:?\s*$'
    match = re.match(bold_pattern, line)
    if match:
        text_part = match.group(1).strip()
        return True, 2, 'bold', f"<b>{text_part}</b>"
    
    # Pattern 8: Title case lines with specific characteristics
    if (line.istitle() and 
        len(line.split()) <= 8 and 
        not line.endswith('.') and 
        not any(char in line for char in ['(', ')', '"', "'"])):
        return True, 2, 'title', f"<b>{line}</b>"
    
    return False, 0, None, line

def parse_markdown_to_pdf_elements(text, styles):
    elements = []
    lines = text.split('\n')
    current_section_level = 0
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            elements.append(Spacer(1, 6))
            continue
        
        # Detect headers
        is_header, level, header_type, formatted_text = detect_header_level_and_type(line)
        
        if is_header:
            current_section_level = level
            
            # Choose appropriate style based on level
            if level == 1:
                style = styles['heading1']
            elif level == 2:
                style = styles['heading2']
            elif level == 3:
                style = styles['heading3']
            else:
                style = styles['heading4']
            
            elements.append(Paragraph(formatted_text, style))
            continue
        
        # Handle bullet points and lists
        if line.startswith('- ') or line.startswith('• '):
            bullet_text = line[2:].strip()
            # Convert any remaining bold markers
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(f"• {bullet_text}", styles['bullet']))
            continue
        
        if line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(f"• {bullet_text}", styles['bullet']))
            continue
        
        # Handle numbered lists that aren't headers
        if re.match(r'^\d+\.\s', line) and not detect_header_level_and_type(line)[0]:
            list_text = re.sub(r'^\d+\.\s', '', line)
            list_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', list_text)
            elements.append(Paragraph(f"• {list_text}", styles['bullet']))
            continue
        
        # Regular content - choose indentation based on current section level
        content_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        
        if current_section_level == 1:
            style = styles['content1']
        elif current_section_level == 2:
            style = styles['content2']
        elif current_section_level >= 3:
            style = styles['content3']
        else:
            style = styles['normal']
        
        elements.append(Paragraph(content_text, style))
    
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
    
    # Add title if the document starts with a clear title
    lines = summary_text.split('\n')
    if lines and lines[0].strip() and not detect_header_level_and_type(lines[0].strip())[0]:
        first_line = lines[0].strip()
        if len(first_line.split()) <= 12 and first_line.isupper():
            story.append(Paragraph(f"<b>{first_line}</b>", styles['title']))
            story.append(Spacer(1, 12))
            summary_text = '\n'.join(lines[1:])
    
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
