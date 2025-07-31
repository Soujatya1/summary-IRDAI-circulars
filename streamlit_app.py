import streamlit as st
import pdfplumber
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from fpdf import FPDF
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
- Ensure total summary length is approx. **50% of English content pages**.
 
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

class CustomFPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # Optional: Add header if needed
        pass
        
    def footer(self):
        # Add page number at the bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text_for_pdf(text):
    """Clean text to handle encoding issues with FPDF"""
    # Replace problematic characters
    text = text.replace('₹', 'Rs.')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('…', '...')
    text = text.replace('•', '*')  # Replace bullet with asterisk
    text = text.replace('◦', '-')  # Replace hollow bullet
    text = text.replace('▪', '*')  # Replace square bullet
    text = text.replace('▫', '-')  # Replace hollow square bullet
    text = text.replace('‣', '>')  # Replace triangular bullet
    text = text.replace('⁃', '-')  # Replace hyphen bullet
    text = text.replace('§', 'Section')  # Section symbol
    text = text.replace('©', '(c)')  # Copyright symbol
    text = text.replace('®', '(r)')  # Registered trademark
    text = text.replace('™', '(tm)')  # Trademark
    text = text.replace('°', ' degrees')  # Degree symbol
    text = text.replace('×', 'x')  # Multiplication sign
    text = text.replace('÷', '/')  # Division sign
    text = text.replace('≤', '<=')  # Less than or equal
    text = text.replace('≥', '>=')  # Greater than or equal
    text = text.replace('≠', '!=')  # Not equal
    text = text.replace('α', 'alpha')  # Greek letters
    text = text.replace('β', 'beta')
    text = text.replace('γ', 'gamma')
    text = text.replace('δ', 'delta')
    text = text.replace('λ', 'lambda')
    text = text.replace('μ', 'mu')
    text = text.replace('π', 'pi')
    text = text.replace('σ', 'sigma')
    text = text.replace('Ω', 'Omega')
    
    # Handle other currency symbols
    text = text.replace('€', 'EUR')
    text = text.replace('£', 'GBP')
    text = text.replace('¥', 'JPY')
    text = text.replace('¢', 'cents')
    
    # Handle fractions
    text = text.replace('½', '1/2')
    text = text.replace('⅓', '1/3')
    text = text.replace('¼', '1/4')
    text = text.replace('¾', '3/4')
    text = text.replace('⅛', '1/8')
    text = text.replace('⅜', '3/8')
    text = text.replace('⅝', '5/8')
    text = text.replace('⅞', '7/8')
    
    # Handle arrows and symbols
    text = text.replace('→', '->')
    text = text.replace('←', '<-')
    text = text.replace('↑', '^')
    text = text.replace('↓', 'v')
    text = text.replace('↔', '<->')
    text = text.replace('⇒', '=>')
    text = text.replace('⇐', '<=')
    text = text.replace('⇔', '<=>')
    
    # Handle quotation marks and apostrophes
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('‚', ',')
    text = text.replace('„', '"')
    text = text.replace('‹', '<')
    text = text.replace('›', '>')
    
    # Handle various dashes and spaces
    text = text.replace('‐', '-')  # Hyphen
    text = text.replace('‑', '-')  # Non-breaking hyphen
    text = text.replace('‒', '-')  # Figure dash
    text = text.replace('–', '-')  # En dash
    text = text.replace('—', '--') # Em dash
    text = text.replace('―', '--') # Horizontal bar
    text = text.replace(' ', ' ')  # Non-breaking space
    text = text.replace(' ', ' ')  # En space
    text = text.replace(' ', ' ')  # Em space
    text = text.replace(' ', ' ')  # Thin space
    
    # Remove or replace any remaining problematic unicode characters
    # This will convert any remaining non-Latin-1 characters to closest equivalent or remove them
    try:
        # First try to encode/decode to catch any remaining issues
        text = text.encode('latin-1', 'ignore').decode('latin-1')
    except:
        # If still having issues, do a more aggressive cleaning
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if ord(c) < 256)
    
    return text

def parse_and_format_text(pdf, text):
    """Parse text and add formatted content to PDF"""
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)  # Small line break for empty lines
            continue
        
        line = clean_text_for_pdf(line)
        
        # Check for different heading levels and formats
        if line.startswith('###'):
            # Sub-subheading
            heading_text = line[3:].strip()
            if heading_text:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 11)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 8, heading_text, 0, 1, 'L')
                pdf.ln(2)
                continue
        
        elif line.startswith('##'):
            # Subheading
            heading_text = line[2:].strip()
            if heading_text:
                pdf.ln(6)
                pdf.set_font('Arial', 'B', 12)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 8, heading_text, 0, 1, 'L')
                pdf.ln(3)
                continue
        
        elif line.startswith('#'):
            # Main heading
            heading_text = line[1:].strip()
            if heading_text:
                pdf.ln(8)
                pdf.set_font('Arial', 'B', 14)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 10, heading_text, 0, 1, 'L')
                pdf.ln(4)
                continue
        
        # Check for bold text patterns (markdown style)
        bold_match = re.match(r'^\*\*(.*?)\*\*:?\s*$', line)
        if bold_match:
            heading_text = bold_match.group(1).strip()
            if heading_text:
                pdf.ln(4)
                pdf.set_font('Arial', 'B', 11)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 7, heading_text, 0, 1, 'L')
                pdf.ln(2)
                continue
        
        # Check for lines that end with colon (likely headings)
        if line.endswith(':') and len(line.split()) <= 8:
            pdf.ln(4)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, line, 0, 1, 'L')
            pdf.ln(2)
            continue
        
        # Check for bullet points
        if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(0, 0, 0)
            
            # Add bullet point with indentation
            pdf.cell(10, 5, '*', 0, 0, 'L')
            
            # Handle long bullet text with multi-line
            remaining_width = pdf.w - pdf.l_margin - pdf.r_margin - 10
            if pdf.get_string_width(bullet_text) > remaining_width:
                # Multi-line bullet point
                words = bullet_text.split()
                current_line = ""
                first_line = True
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if pdf.get_string_width(test_line) <= remaining_width:
                        current_line = test_line
                    else:
                        if current_line:
                            if first_line:
                                pdf.cell(0, 5, current_line, 0, 1, 'L')
                                first_line = False
                            else:
                                pdf.cell(10, 5, '', 0, 0, 'L')  # Indent continuation
                                pdf.cell(0, 5, current_line, 0, 1, 'L')
                            current_line = word
                        else:
                            # Word is too long, just add it
                            if first_line:
                                pdf.cell(0, 5, word, 0, 1, 'L')
                                first_line = False
                            else:
                                pdf.cell(10, 5, '', 0, 0, 'L')
                                pdf.cell(0, 5, word, 0, 1, 'L')
                
                if current_line:
                    if first_line:
                        pdf.cell(0, 5, current_line, 0, 1, 'L')
                    else:
                        pdf.cell(10, 5, '', 0, 0, 'L')
                        pdf.cell(0, 5, current_line, 0, 1, 'L')
            else:
                pdf.cell(0, 5, bullet_text, 0, 1, 'L')
            
            pdf.ln(1)
            continue
        
        # Check for numbered lists
        if re.match(r'^\d+\.\s', line):
            list_text = re.sub(r'^\d+\.\s', '', line)
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(0, 0, 0)
            
            # Get the number part
            number_match = re.match(r'^(\d+)\.\s', line)
            if number_match:
                number = number_match.group(1) + ". "
                pdf.cell(15, 5, number, 0, 0, 'L')
                
                # Handle long list text with multi-line
                remaining_width = pdf.w - pdf.l_margin - pdf.r_margin - 15
                if pdf.get_string_width(list_text) > remaining_width:
                    # Multi-line list item
                    words = list_text.split()
                    current_line = ""
                    first_line = True
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        if pdf.get_string_width(test_line) <= remaining_width:
                            current_line = test_line
                        else:
                            if current_line:
                                if first_line:
                                    pdf.cell(0, 5, current_line, 0, 1, 'L')
                                    first_line = False
                                else:
                                    pdf.cell(15, 5, '', 0, 0, 'L')  # Indent continuation
                                    pdf.cell(0, 5, current_line, 0, 1, 'L')
                                current_line = word
                            else:
                                if first_line:
                                    pdf.cell(0, 5, word, 0, 1, 'L')
                                    first_line = False
                                else:
                                    pdf.cell(15, 5, '', 0, 0, 'L')
                                    pdf.cell(0, 5, word, 0, 1, 'L')
                    
                    if current_line:
                        if first_line:
                            pdf.cell(0, 5, current_line, 0, 1, 'L')
                        else:
                            pdf.cell(15, 5, '', 0, 0, 'L')
                            pdf.cell(0, 5, current_line, 0, 1, 'L')
                else:
                    pdf.cell(0, 5, list_text, 0, 1, 'L')
            
            pdf.ln(1)
            continue
        
        # Regular paragraph text
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0, 0, 0)
        
        # Handle long paragraphs with multi-line
        available_width = pdf.w - pdf.l_margin - pdf.r_margin
        if pdf.get_string_width(line) > available_width:
            # Multi-line paragraph
            words = line.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if pdf.get_string_width(test_line) <= available_width:
                    current_line = test_line
                else:
                    if current_line:
                        pdf.cell(0, 5, current_line, 0, 1, 'L')
                        current_line = word
                    else:
                        # Word is too long, just add it
                        pdf.cell(0, 5, word, 0, 1, 'L')
            
            if current_line:
                pdf.cell(0, 5, current_line, 0, 1, 'L')
        else:
            pdf.cell(0, 5, line, 0, 1, 'L')
        
        pdf.ln(2)

def generate_pdf(summary_text):
    """Generate PDF using FPDF"""
    pdf = CustomFPDF()
    pdf.add_page()
    
    # Set title
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, 'Document Summary', 0, 1, 'C')
    pdf.ln(10)
    
    # Parse and add formatted content
    parse_and_format_text(pdf, summary_text)
    
    # Create BytesIO buffer and save PDF
    buffer = BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin-1')
    buffer.write(pdf_string)
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
        
        try:
            pdf_file = generate_pdf(full_summary)
            
            download_filename = generate_download_filename(uploaded_file.name)
            
            st.download_button(
                "Download Summary (PDF)", 
                data=pdf_file, 
                file_name=download_filename,
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.info("You can still copy the summary text above.")
    else:
        st.error("No English content found in the uploaded PDF.")
