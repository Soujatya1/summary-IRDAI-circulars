import streamlit as st
import fitz
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from dotenv import load_dotenv
from fpdf import FPDF
from io import BytesIO
import os
import re

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

def detect_english_sentences(text):
    """
    Extract English sentences and numeric content from the given text
    """
    # Normalize whitespace and preserve paragraph breaks
    text = re.sub(r'\n\s*\n', ' |PARA_BREAK| ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split on sentence boundaries, but preserve numbered sections
    # Pattern: sentence endings (. ! ?) followed by space, but not after single digits
    sentence_pattern = r'(?<!\b\d)[.!?]+\s+(?=[A-Z]|\d+\.|\|PARA_BREAK\||$)'
    
    # Split text into potential sentences
    segments = re.split(sentence_pattern, text)
    
    # Further process paragraph breaks
    processed_segments = []
    for segment in segments:
        if '|PARA_BREAK|' in segment:
            parts = [p.strip() for p in segment.split('|PARA_BREAK|') if p.strip()]
            processed_segments.extend(parts)
        else:
            processed_segments.append(segment.strip())
    
    english_sentences = []
    
    for segment in processed_segments:
        if not segment or len(segment) < 3:
            continue
        
        # Count different types of content
        english_chars = len(re.findall(r'[a-zA-Z]', segment))
        digits = len(re.findall(r'[0-9]', segment))
        total_chars = len(re.sub(r'\s+', '', segment))
        
        # Check for section headers (like "2.1 Background" or "Chapter 3")
        is_section_header = bool(re.match(r'^\d+\.?\d*\.?\s*[A-Za-z]|^[A-Za-z][^.]*\s+\d+\.?\d*', segment))
        
        # Check for reference patterns to exclude (like IRDAI/E&C/ORD/MISC/115/09/2024)
        is_reference_pattern = [
            r'(?i)ref\.?\s*[:\-]?\s*[A-Z]+[/\\][A-Z&]+[/\\][A-Z]+[/\\][A-Z]+[/\\]\d+[/\\]\d+[/\\]\d{4}',  # IRDAI/E&C/ORD/MISC/115/09/2024
            r'(?i)[A-Z]{2,}[/\\][A-Z&]{2,}[/\\][A-Z]{2,}[/\\][A-Z]{2,}[/\\]\d+[/\\]\d+[/\\]\d{4}',       # General pattern
            r'(?i)[A-Z]{3,}[/\\-][A-Z&/\\-]+[/\\-]\d+[/\\-]\d+[/\\-]\d{4}',                              # Variations with dashes
            r'(?i)^[A-Z]+[/\\][A-Z&]+.*\d+[/\\]\d+[/\\]\d{4}'
        ]
        
        # Decision logic - Skip reference patterns
        should_include = False
        
        # Skip if it's a reference pattern
        if is_reference_pattern:
            should_include = False
        
        # Always include section headers (unless they're reference patterns)
        elif is_section_header:
            should_include = True
        
        # Include if has sufficient English content
        elif english_chars >= 5:
            try:
                # Try language detection for English text
                detected_lang = detect(segment)
                if detected_lang == "en":
                    should_include = True
            except:
                # Fallback: if mostly English characters, assume English
                if total_chars > 0 and (english_chars / total_chars) > 0.3:
                    should_include = True
        
        # Include if has meaningful numeric content (even without English)
        elif has_meaningful_numbers and digits >= 2:
            should_include = True
        
        # Include mixed content (English + numbers)
        elif english_chars >= 3 and digits >= 2:
            meaningful_content = english_chars + digits
            if total_chars > 0 and (meaningful_content / total_chars) > 0.4:
                should_include = True
        
        if should_include:
            english_sentences.append(segment)
    
    return english_sentences

def extract_english_content_sentence_level(doc):
    """
    Extract English content at sentence level from PDF document
    """
    english_text = ""
    english_sentence_count = 0
    total_page_count = len(doc)
    pages_with_english = 0
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        
        if page_text:
            # Get English sentences from this page
            english_sentences = detect_english_sentences(page_text)
            
            if english_sentences:
                pages_with_english += 1
                english_sentence_count += len(english_sentences)
                
                # Join sentences with proper spacing
                page_english_text = '. '.join(english_sentences)
                
                # Add page separator for better context
                english_text += f"\n\n--- Page {page_num + 1} ---\n"
                english_text += page_english_text + ".\n"
    
    return english_text, english_sentence_count, total_page_count, pages_with_english

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

def get_summary_prompt(text):
    return f"""
You are acting as a **Senior Legal Analyst** and Regulatory Compliance Officer specializing in IRDAI, UIDAI, and eGazette circulars.
 
Your task is to generate a **legally precise, clause-preserving, structure-aligned summary** of the in-put regulatory document. Your summary will be reviewed for legal compliance, so accuracy is critical.

Use clean plain text.
Do not use Markdown formatting (no **bold**, `code`, or extra spacing).

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
- Use ONLY plain text with NO formatting symbols
- DO NOT use ANY of these characters: # * ** __ ` ~ [ ] ( ) for formatting
- DO NOT use markdown headers like ## or ###
- DO NOT use bold markers like **text** or __text__
- DO NOT use italic markers like *text* or _text_
- Use simple dashes (-) or bullet points (•) for lists
- Preserve order and hierarchy using numbers and letters (1, a, i)
- Use CAPITAL LETTERS for emphasis instead of bold
- Use line breaks and indentation for structure
- Preserve order and hierarchy (e.g., 1 → a → i).
- Do not invent or rename headings.
 
---
 
### SUMMARY LENGTH RULE:
- Ensure total summary length is approx. **50% of English content pages**.
 
---
 
Now begin the **section-wise clause-preserving summary** of the following legal document:
--------------------
{text}
"""

class UTF8PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_font('Arial', '', 12)
    
    def header(self):
        # Only show header on first page
        if self.page_no() == 1:
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'IRDAI Circular Summary', 0, 1, 'C')
            self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def clean_text(self, text):
        """Clean text to remove problematic Unicode characters"""
        # Define character replacements
        replacements = {
            '\u2019': "'",      # Right single quotation mark
            '\u2018': "'",      # Left single quotation mark
            '\u201c': '"',      # Left double quotation mark
            '\u201d': '"',      # Right double quotation mark
            '\u2013': '-',      # En dash
            '\u2014': '-',      # Em dash
            '\u2026': '...',    # Horizontal ellipsis
            '\u2022': '•',      # Bullet point
            '\u20b9': 'Rs. ',   # Indian Rupee sign
            '\u00a0': ' ',      # Non-breaking space
            '\u2212': '-',      # Minus sign
            '\u00b0': 'deg',    # Degree symbol
            '\u00a9': '(c)',    # Copyright symbol
            '\u00ae': '(r)',    # Registered trademark
            '\u2122': 'TM',     # Trademark symbol
        }
        
        # Apply replacements
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Remove any remaining non-ASCII characters
        try:
            text.encode('latin-1')
            return text
        except UnicodeEncodeError:
            # If still problematic, keep only ASCII characters
            return ''.join(char if ord(char) < 128 else '?' for char in text)

def generate_pdf(summary_text):
    pdf = UTF8PDF()
    
    # Set margins
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.set_top_margin(30)
    
    pdf.set_font('Arial', '', 11)  # Slightly smaller font for better fitting
    
    # Clean the entire text first
    clean_text = pdf.clean_text(summary_text.strip())
    
    # Split by lines to preserve exact structure
    lines = clean_text.split('\n')
    
    for line in lines:
        # Handle empty lines (preserve spacing)
        if not line.strip():
            pdf.ln(5)  # Add some vertical space for empty lines
            continue
        
        # Check if line fits on current page, if not add new page
        if pdf.get_y() > pdf.h - 30:  # 30mm from bottom
            pdf.add_page()
        
        # Handle long lines that need wrapping
        words = line.split(' ')
        current_line = ''
        
        for word in words:
            test_line = current_line + ' ' + word if current_line else word
            
            # Check if the test line fits within margins
            if pdf.get_string_width(test_line) <= (pdf.w - pdf.l_margin - pdf.r_margin):
                current_line = test_line
            else:
                # Print current line and start new line with current word
                if current_line:
                    pdf.cell(0, 5, current_line, 0, 1)
                current_line = word
        
        # Print the remaining line
        if current_line:
            pdf.cell(0, 5, current_line, 0, 1)
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    try:
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
        buffer.write(pdf_bytes)
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None
    
    buffer.seek(0)
    return buffer

st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if llm is None:
        st.error("Cannot process document: Azure OpenAI client not properly configured.")
        st.stop()
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Use sentence-level detection
    english_text, english_sentence_count, total_page_count, pages_with_english = extract_english_content_sentence_level(doc)
    
    if english_sentence_count == 0:
        st.error("No English sentences detected in the document.")
        st.stop()

    st.success(f"Total pages: {total_page_count} | Pages with English content: {pages_with_english} | English sentences extracted: {english_sentence_count}")
    
    # Optional: Show a preview of extracted content
    with st.expander("Preview Extracted English Content"):
        st.text_area("Extracted English Text (first 2000 characters):", 
                    value=english_text[:2000] + "..." if len(english_text) > 2000 else english_text, 
                    height=200)

    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=50)
    chunks = splitter.split_text(english_text)

    st.info(f"Summarizing {english_sentence_count} English sentences across {len(chunks)} chunks...")

    full_summary = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1} of {len(chunks)}..."):
            messages = [
                SystemMessage(content="You are a professional IRDAI summarizer. Follow all instructions strictly."),
                HumanMessage(content=get_summary_prompt(chunk))
            ]
            response = llm(messages)
            full_summary += "\n\n" + response.content.strip()

    # Rest of the code remains the same (deduplication, display, PDF generation)
    def remove_redundant_blocks(text):
        lines = text.strip().split("\n")
        cleaned = []
        prev = ""
        for line in lines:
            if line.strip() != prev.strip():
                cleaned.append(line)
            prev = line
        return "\n".join(cleaned)

    full_summary = remove_redundant_blocks(full_summary)

    # Show summary
    st.subheader("Section-wise Summary")
    st.text_area("Generated Summary:", value=full_summary, height=600)

    # Download button for PDF
    pdf_file = generate_pdf(full_summary)
    st.download_button(
        label="Download Summary as .pdf",
        data=pdf_file,
        file_name="irdai_summary.pdf",
        mime="application/pdf"
    )
