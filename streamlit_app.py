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

def extract_numbering_patterns(text):
    """
    Extract and identify various numbering patterns from the text
    """
    patterns = {
        'section_numbers': r'\b(\d+\.\d+(?:\.\d+)*)\s*[:\-\.]?\s*([A-Z][^.]*?)(?=\n|\d+\.\d+|$)',
        'clause_numbers': r'\b(\d+\.\d+(?:\.\d+)*)\s*(.+?)(?=\n\d+\.\d+|\n[A-Z]|\n$)',
        'sub_clauses': r'\b([a-z]\)|[ivx]+\)|[IVX]+\)|\([a-z]\)|\([ivx]+\)|\([IVX]+\))\s*(.+?)(?=\n[a-z]\)|\n[ivx]+\)|\n[IVX]+\)|\n\([a-z]\)|\n\([ivx]+\)|\n\([IVX]+\)|\n$)',
        'bullet_points': r'^\s*[•\-\*]\s*(.+?)(?=\n\s*[•\-\*]|\n[A-Z]|\n\d+\.|\n$)',
        'numbered_lists': r'^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\n[A-Z]|\n$)',
        'alphabetic_lists': r'^\s*([A-Za-z])\.\s*(.+?)(?=\n\s*[A-Za-z]\.|\n[A-Z]|\n\d+\.|\n$)'
    }
    
    found_patterns = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            found_patterns[pattern_name] = matches
    
    return found_patterns

def detect_english_sentences(text):
    """
    Extract only English sentences from the given text while preserving numbering
    """
    # Split text into sentences using multiple delimiters
    # This regex splits on periods, exclamation marks, question marks, and newlines
    # while trying to preserve sentence structure and numbering
    sentence_endings = r'[.!?]+\s*'
    potential_sentences = re.split(sentence_endings, text)
    
    english_sentences = []
    
    for sentence in potential_sentences:
        sentence = sentence.strip()
        
        # Skip very short sentences (likely fragments) unless they contain numbering
        if len(sentence) < 10 and not re.search(r'\b\d+\.\d*|\([a-z]\)|\b[ivx]+\)', sentence):
            continue
            
        # Skip sentences that are mostly numbers or special characters unless they're part of structured content
        if len(re.sub(r'[^a-zA-Z]', '', sentence)) < 5 and not re.search(r'\b\d+\.\d*|\([a-z]\)|\b[ivx]+\)', sentence):
            continue
            
        try:
            # Detect language of this sentence
            detected_lang = detect(sentence)
            
            if detected_lang == "en":
                english_sentences.append(sentence)
                
        except Exception as e:
            # If detection fails, try a simple heuristic
            # Check if sentence contains mostly English characters or structured numbering
            english_chars = len(re.findall(r'[a-zA-Z]', sentence))
            total_chars = len(re.sub(r'\s', '', sentence))
            has_numbering = bool(re.search(r'\b\d+\.\d*|\([a-z]\)|\b[ivx]+\)', sentence))
            
            if total_chars > 0 and ((english_chars / total_chars) > 0.7 or has_numbering):
                english_sentences.append(sentence)
    
    return english_sentences

def extract_english_content_sentence_level(doc):
    """
    Extract English content at sentence level from PDF document while preserving structure
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
                
                # Join sentences with proper spacing while preserving structure
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
 
Your task is to generate a **legally precise, clause-preserving, structure-aligned summary** of the input regulatory document. Your summary will be reviewed for legal compliance, so accuracy is critical.

Use clean plain text.
Do not use Markdown formatting (no **bold**, `code`, or extra spacing).

---
 
### CRITICAL NUMBERING PRESERVATION RULES
 
**1. EXACT NUMBERING RETENTION (MANDATORY):**
- **NEVER change, skip, or renumber** any section, clause, or sub-clause numbers
- Preserve EXACT numbering as it appears in source: 
  - If source shows "2.3.1", your summary MUST show "2.3.1"
  - If source shows "a)", your summary MUST show "a)"
  - If source shows "(i)", your summary MUST show "(i)"
  - If source shows "Section 4.2.5", your summary MUST show "Section 4.2.5"
- **Maintain hierarchical relationships**: If 2.1 comes before 2.2, preserve this order
- **Never merge numbered items**: Each numbered clause gets its own summary line

**2. STRUCTURE PRESERVATION (Strict Order):**
- Retain **original structure** EXACTLY, including:
  - Section headers with their original numbers (e.g., "3. POLICY CONDITIONS")
  - Subheaders with numbers (e.g., "3.1 Revival Period", "3.2 Grace Period")
  - Sub-subheaders (e.g., "3.2.1 Application Process", "3.2.2 Documentation")
  - Clause numbers (e.g., 3.2.1, a), b), c), (i), (ii), (iii))
  - Bullet formats, indentation levels
- Do not reorder, combine, or rename any sections or sub-sections
- If source document has gaps in numbering (e.g., goes from 2.1 to 2.3), preserve the gap
 
**3. CLAUSE-BY-CLAUSE SUMMARIZATION (NO MERGING):**
- **Summarize one numbered clause per line only**
- **Each number gets its own summary line** - NEVER combine multiple numbered items
- If a clause is broken across lines or pages, **treat it as a single numbered clause**
- Example format:
  - 2.1 [Summary of clause 2.1]
  - 2.2 [Summary of clause 2.2]
  - 2.2.1 [Summary of sub-clause 2.2.1]
  - a) [Summary of point a]
  - b) [Summary of point b]
 
**4. PRESERVE LEGAL PHRASES & CAUSALITY TRIGGERS:**
- Never skip or simplify phrases like:
  - **"unless"**, **"until"**, **"after"**, **"shall"**, **"subject to"**, **"provided that"**
  - **"notwithstanding"**, **"in accordance with"**, **"as per"**, **"pursuant to"**
- These are **legally binding conditions** and must be **retained with their meaning intact**
 
**5. DEFINITIONS & EXPLANATORY SECTIONS:**
- If the section contains **definitions** or classifications:
  - Preserve the section number and list each term with its number:
    - 1.1 Definition: Revival Period – A policy may be revived within…
    - 1.2 Definition: Grace Period – The period during which…
  - **Do not merge multiple definitions** into one block
 
**6. COMMITTEES, PANELS, AUTHORITIES (EXACT NAMES):**
- Retain **every mention of committees and positions verbatim**
- Never shorten or generalize:
  - "Product Management Committee (PMC)" not "product committee"
  - "Chief Compliance Officer" not "Compliance Head"
  - "Member – Life", "Key Management Persons (KMPs)", "Appointed Actuary", etc.
- Repeat full names every time they appear, even if already mentioned before
 
**7. TABLES – PRESERVE IN FULL WITH NUMBERING:**
- If tables are numbered (e.g., "Table 2.1", "Annexure A"), retain these numbers
- Summarize **column-by-column**, row-by-row with original structure
- Do not omit any row (e.g., Discontinuance Charges for all policy years)
- Example format:
  - Table 2.1: Discontinuance Charges
    - Year 1: Lower of 2% or ₹3,000  
    - Year 2: Lower of 1.5% or ₹2,000  
    …
 
**8. NUMERIC LIMITS & ABBREVIATIONS:**
- Maintain correct expressions like:
  - Rs. 1,000/- (not "Rs 1000")
  - "AP or FV, whichever is lower" (do not paraphrase this)
- Preserve exact numerical references and their associated numbering
 
**9. HISTORICAL & AUTHORITY CLAUSES:**
- Include all numbered clauses like:
  - "5.1 Repeal and Savings"
  - "6.2 Authority's power to issue clarifications"
- Do **not skip final sections** even if repetitive - preserve their numbers
 
**10. SIGNATURE, SEAL, PUBLICATION TEXT – OMIT:**
- Strictly exclude:
  - Signature blocks (e.g., "Debasish Panda, Chairperson")
  - Digital signing metadata ("Digitally signed by Manoj Kumar Verma")
  - Footer/publication notices ("Uploaded by Dte. of Printing…")
 
**11. LINE BREAKS & ORPHAN HANDLING:**
- Do not treat broken lines (from PDF formatting) as new clauses
- Ensure a single numbered sentence broken across lines is still summarized as one numbered thought
- Maintain the original numbering even if text spans multiple lines
 
---
 
### OUTPUT FORMAT:
- Use ONLY plain text with NO formatting symbols
- DO NOT use ANY of these characters: # * ** __ ` ~ [ ] ( ) for formatting
- DO NOT use markdown headers like ## or ###
- DO NOT use bold markers like **text** or __text__
- DO NOT use italic markers like *text* or _text_
- Use simple dashes (-) or bullet points (•) for unnumbered lists only
- **ALWAYS preserve original numbering** - this is the most critical requirement
- Use CAPITAL LETTERS for emphasis instead of bold
- Use line breaks and indentation for structure
- Preserve order and hierarchy with EXACT original numbers (e.g., 1.2.3 → a → i)
- Do not invent or rename headings - keep original numbering and titles
 
---
 
### SUMMARY LENGTH RULE:
- Ensure total summary length is approx. **50% of English content pages**
 
---

### CRITICAL REMINDER:
The most important aspect of this summary is PRESERVING THE EXACT NUMBERING from the source document. Every section, clause, sub-clause, and point number must appear EXACTLY as it does in the original document. This is a legal requirement for regulatory compliance.
 
Now begin the **section-wise clause-preserving summary** with **EXACT numbering retention** of the following legal document:
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
        """Clean text to remove problematic Unicode characters while preserving numbering"""
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
        
        # Remove any remaining non-ASCII characters but preserve numbers and basic punctuation
        try:
            text.encode('latin-1')
            return text
        except UnicodeEncodeError:
            # If still problematic, keep ASCII characters and important symbols for numbering
            cleaned = ''
            for char in text:
                if ord(char) < 128:
                    cleaned += char
                elif char in '().-':  # Important for numbering
                    cleaned += char
                else:
                    cleaned += '?'
            return cleaned

def generate_pdf(summary_text):
    pdf = UTF8PDF()
    
    # Set margins
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.set_top_margin(30)
    
    pdf.set_font('Arial', '', 11)  # Slightly smaller font for better fitting
    
    # Clean the entire text first while preserving structure
    clean_text = pdf.clean_text(summary_text.strip())
    
    # Split by lines to preserve exact structure and numbering
    lines = clean_text.split('\n')
    
    for line in lines:
        # Handle empty lines (preserve spacing)
        if not line.strip():
            pdf.ln(5)  # Add some vertical space for empty lines
            continue
        
        # Check if line fits on current page, if not add new page
        if pdf.get_y() > pdf.h - 30:  # 30mm from bottom
            pdf.add_page()
        
        # Handle long lines that need wrapping while preserving numbering at start
        words = line.split(' ')
        current_line = ''
        
        # Check if this line starts with numbering and preserve indentation
        is_numbered = bool(re.match(r'^\s*(\d+\.(?:\d+\.)*|\([a-z]\)|\b[ivx]+\))', line.strip()))
        indent = len(line) - len(line.lstrip()) if is_numbered else 0
        
        for word in words:
            test_line = current_line + ' ' + word if current_line else word
            
            # Check if the test line fits within margins
            if pdf.get_string_width(test_line) <= (pdf.w - pdf.l_margin - pdf.r_margin - indent):
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

def remove_redundant_blocks(text):
    """Remove redundant blocks while preserving numbering structure"""
    lines = text.strip().split("\n")
    cleaned = []
    prev = ""
    
    for line in lines:
        line_stripped = line.strip()
        
        # Always keep numbered lines even if they seem similar
        if re.match(r'^\s*(\d+\.(?:\d+\.)*|\([a-z]\)|\b[ivx]+\))', line_stripped):
            cleaned.append(line)
        # For non-numbered lines, check for redundancy
        elif line_stripped != prev.strip():
            cleaned.append(line)
        
        prev = line
    
    return "\n".join(cleaned)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("IRDAI Circular Summarizer with Enhanced Numbering Retention")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if llm is None:
        st.error("Cannot process document: Azure OpenAI client not properly configured.")
        st.stop()
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Use sentence-level detection with enhanced numbering preservation
    english_text, english_sentence_count, total_page_count, pages_with_english = extract_english_content_sentence_level(doc)
    
    if english_sentence_count == 0:
        st.error("No English sentences detected in the document.")
        st.stop()

    st.success(f"Total pages: {total_page_count} | Pages with English content: {pages_with_english} | English sentences extracted: {english_sentence_count}")
    
    # Extract and display numbering patterns found
    numbering_patterns = extract_numbering_patterns(english_text)
    if numbering_patterns:
        with st.expander("Detected Numbering Patterns"):
            for pattern_type, matches in numbering_patterns.items():
                st.write(f"**{pattern_type.replace('_', ' ').title()}:** {len(matches)} found")
                if matches:
                    st.write("Examples:", [match[0] if isinstance(match, tuple) else match for match in matches[:3]])
    
    # Optional: Show a preview of extracted content
    with st.expander("Preview Extracted English Content"):
        st.text_area("Extracted English Text (first 2000 characters):", 
                    value=english_text[:2000] + "..." if len(english_text) > 2000 else english_text, 
                    height=200)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500, 
        chunk_overlap=100,  # Increased overlap to preserve numbering context
        separators=["\n\n", "\n", ". ", " ", ""]  # Modified separators to better preserve structure
    )
    chunks = splitter.split_text(english_text)

    st.info(f"Summarizing {english_sentence_count} English sentences across {len(chunks)} chunks...")

    full_summary = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1} of {len(chunks)}..."):
            messages = [
                SystemMessage(content="You are a professional IRDAI summarizer. Follow all instructions strictly, especially numbering preservation."),
                HumanMessage(content=get_summary_prompt(chunk))
            ]
            response = llm(messages)
            chunk_summary = response.content.strip()
            
            # Add chunk separator only if there's meaningful content
            if chunk_summary and not chunk_summary.isspace():
                if full_summary:  # Add separator only if there's already content
                    full_summary += "\n\n" + chunk_summary
                else:
                    full_summary = chunk_summary

    # Enhanced deduplication that preserves numbering
    full_summary = remove_redundant_blocks(full_summary)

    # Show summary
    st.subheader("Section-wise Summary with Preserved Numbering")
    st.text_area("Generated Summary:", value=full_summary, height=600)

    # Download button for PDF
    if full_summary:
        pdf_file = generate_pdf(full_summary)
        if pdf_file:
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_file,
                file_name="irdai_summary_with_numbering.pdf",
                mime="application/pdf"
            )
    
    # Additional validation section
    with st.expander("Numbering Validation"):
        original_numbers = re.findall(r'\b\d+\.\d+(?:\.\d+)*|\([a-z]\)|\b[ivx]+\)', english_text)
        summary_numbers = re.findall(r'\b\d+\.\d+(?:\.\d+)*|\([a-z]\)|\b[ivx]+\)', full_summary)
        
        st.write(f"Original document numbering instances: {len(original_numbers)}")
        st.write(f"Summary numbering instances: {len(summary_numbers)}")
        
        if len(summary_numbers) < len(original_numbers) * 0.7:  # If less than 70% retained
            st.warning("Some numbering may have been lost during summarization. Consider reviewing the output.")
        else:
            st.success("Good numbering retention achieved!")
