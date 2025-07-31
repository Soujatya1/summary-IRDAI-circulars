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
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor='black',
        fontName='Helvetica-Bold'
    )
    
    # Main heading (1., 2., etc.)
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=0
    )
    
    # Sub heading (1.1, 1.2, etc.)
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=15
    )
    
    # Sub-sub heading (1.1.1, 1.1.2, etc.)
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=5,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=30
    )
    
    # Minor heading (a), b), etc.)
    heading4_style = ParagraphStyle(
        'CustomHeading4',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=4,
        alignment=TA_LEFT,
        textColor='black',
        fontName='Helvetica-Bold',
        leftIndent=45
    )
    
    # Normal paragraph
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=0,
        firstLineIndent=0
    )
    
    # Indented normal text (for content under headings)
    indented_style = ParagraphStyle(
        'CustomIndented',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=15,
        firstLineIndent=0
    )
    
    # Deep indented text
    deep_indented_style = ParagraphStyle(
        'CustomDeepIndented',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        textColor='black',
        leftIndent=30,
        firstLineIndent=0
    )
    
    # Bullet points
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=2,
        spaceAfter=4,
        leftIndent=20,
        alignment=TA_JUSTIFY,
        textColor='black',
        bulletIndent=5
    )
    
    # Definition style
    definition_style = ParagraphStyle(
        'CustomDefinition',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=25,
        alignment=TA_JUSTIFY,
        textColor='black',
        firstLineIndent=-10
    )
    
    return {
        'title': title_style,
        'heading1': heading1_style,
        'heading2': heading2_style,
        'heading3': heading3_style,
        'heading4': heading4_style,
        'normal': normal_style,
        'indented': indented_style,
        'deep_indented': deep_indented_style,
        'bullet': bullet_style,
        'definition': definition_style
    }

def detect_heading_level(line):
    """
    Detect the heading level based on numbering patterns
    Returns (level, cleaned_text) where level is 1-4, or 0 for normal text
    """
    line = line.strip()
    
    # Level 1: Numbers like "1.", "2.", "10." at start
    if re.match(r'^\d+\.\s+', line):
        return 1, line
    
    # Level 2: Decimal numbering like "1.1", "2.3", "10.15"
    if re.match(r'^\d+\.\d+\.?\s+', line):
        return 2, line
    
    # Level 3: Three-level numbering like "1.1.1", "2.3.4"
    if re.match(r'^\d+\.\d+\.\d+\.?\s+', line):
        return 3, line
    
    # Level 4: Letter numbering like "a)", "b)", "(i)", "(ii)"
    if re.match(r'^[a-z]\)\s+|^\([ivx]+\)\s+|^\([a-z]\)\s+', line.lower()):
        return 4, line
    
    # Check for bold headings (common in regulatory docs)
    if line.startswith('**') and line.endswith('**'):
        clean_line = line[2:-2].strip()
        # If it's short and looks like a heading
        if len(clean_line.split()) <= 8 and (clean_line.isupper() or clean_line.istitle()):
            return 1, clean_line
    
    # Check for ALL CAPS headings (but not too long)
    if line.isupper() and len(line.split()) <= 10 and not line.endswith('.'):
        return 1, line
    
    # Check for Title Case headings ending with colon
    if line.endswith(':') and line.istitle() and len(line.split()) <= 8:
        return 2, line
    
    return 0, line

def parse_improved_markdown_to_pdf_elements(text, styles):
    """
    Improved parser that better handles regulatory document structure
    """
    elements = []
    lines = text.split('\n')
    current_indent_level = 0
    
    for i, line in enumerate(lines):
        original_line = line
        line = line.strip()
        
        # Skip empty lines but add appropriate spacing
        if not line:
            elements.append(Spacer(1, 6))
            continue
        
        # Detect heading level
        heading_level, cleaned_line = detect_heading_level(line)
        
        if heading_level > 0:
            # This is a heading
            current_indent_level = heading_level
            
            # Clean up the text for better formatting
            if cleaned_line.startswith('**') and cleaned_line.endswith('**'):
                cleaned_line = cleaned_line[2:-2].strip()
            
            # Choose appropriate style based on level
            if heading_level == 1:
                style = styles['heading1']
            elif heading_level == 2:
                style = styles['heading2']
            elif heading_level == 3:
                style = styles['heading3']
            else:  # level 4
                style = styles['heading4']
            
            elements.append(Paragraph(cleaned_line, style))
            
        else:
            # This is regular content
            
            # Handle bullet points
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                bullet_text = line[2:].strip()
                # Handle bold text in bullets
                if '**' in bullet_text:
                    bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
                elements.append(Paragraph(f"• {bullet_text}", styles['bullet']))
                
            # Handle definition-style content (Term: Definition)
            elif ':' in line and line.split(':')[0].strip().replace('*', '').istitle():
                # This looks like a definition
                parts = line.split(':', 1)
                term = parts[0].strip().replace('*', '')
                definition = parts[1].strip() if len(parts) > 1 else ''
                
                if definition:
                    definition_text = f"<b>{term}:</b> {definition}"
                else:
                    definition_text = f"<b>{term}</b>"
                
                elements.append(Paragraph(definition_text, styles['definition']))
                
            # Handle numbered lists that aren't headings
            elif re.match(r'^\d+\)\s+', line) or re.match(r'^\(\d+\)\s+', line):
                list_text = re.sub(r'^\d+\)\s+|^\(\d+\)\s+', '', line)
                if '**' in list_text:
                    list_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', list_text)
                elements.append(Paragraph(f"• {list_text}", styles['bullet']))
                
            else:
                # Regular paragraph text
                
                # Handle bold text
                if '**' in line:
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                
                # Choose style based on current context
                if current_indent_level == 0:
                    style = styles['normal']
                elif current_indent_level == 1:
                    style = styles['indented']
                else:
                    style = styles['deep_indented']
                
                elements.append(Paragraph(line, style))
    
    return elements

def generate_pdf(summary_text, title="Document Summary"):
    """
    Generate PDF with improved formatting
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=60,
        leftMargin=60,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = create_pdf_styles()
    story = []
    
    # Add title if provided
    if title and title.strip():
        story.append(Paragraph(title, styles['title']))
        story.append(Spacer(1, 20))
    
    # Parse and add content
    elements = parse_improved_markdown_to_pdf_elements(summary_text, styles)
    story.extend(elements)
    
    # Build the PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        # If there's an error, create a simple fallback PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        fallback_story = [
            Paragraph("Document Summary", styles['title']),
            Spacer(1, 20),
            Paragraph("Error in formatting. Raw content:", styles['heading1']),
            Spacer(1, 10),
            Paragraph(summary_text.replace('\n', '<br/>'), styles['normal'])
        ]
        doc.build(fallback_story)
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
