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

class UTF8PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_font('Arial', '', 12)
    
    def header(self):
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
    
    pdf.set_font('Arial', '', 12)
    
    # Split content into paragraphs
    paragraphs = summary_text.strip().split("\n\n")
    
    for para in paragraphs:
        clean_para = para.strip()
        if clean_para:
            # Clean the text to handle Unicode issues
            clean_para = pdf.clean_text(clean_para)
            
            # Enhanced header detection
            is_header = False
            font_size = 12
            
            # Check for different types of headers
            if (
                # Main document title
                any(title in clean_para for title in ['Final Order', 'IRDAI Circular Summary']) or
                # Section headers with numbers
                (len(clean_para.split()) < 15 and any(clean_para.startswith(prefix) for prefix in [
                    'CHAPTER', 'SECTION', 'PART', 'Article', 'Background', 'Charge-', 'Decision on Charge', 'Summary of Decisions'
                ])) or
                # Numbered sections (1., 2., 3., etc.)
                (len(clean_para) < 150 and clean_para.strip().split('.')[0].isdigit() and len(clean_para.strip().split('.')[0]) <= 2) or
                # Sub-sections (3.1., 4.2., etc.)
                (len(clean_para) < 150 and '.' in clean_para[:10] and clean_para.split('.')[0].replace(' ', '').replace('\t', '').isdigit()) or
                # Headers with #### markdown-style
                clean_para.startswith('####') or
                clean_para.startswith('###') or
                # All caps short lines
                (clean_para.isupper() and len(clean_para) < 100 and len(clean_para.split()) < 10) or
                # Lines ending with colons (likely headers)
                (clean_para.endswith(':') and len(clean_para) < 100)
            ):
                is_header = True
                
                # Determine font size based on header type
                if any(title in clean_para for title in ['Final Order', 'IRDAI Circular Summary']):
                    font_size = 16
                elif clean_para.startswith('####'):
                    font_size = 12
                    clean_para = clean_para.replace('####', '').strip()
                elif clean_para.startswith('###'):
                    font_size = 14
                    clean_para = clean_para.replace('###', '').strip()
                elif any(clean_para.startswith(prefix) for prefix in ['CHAPTER', 'SECTION', 'PART']):
                    font_size = 14
                else:
                    font_size = 13
            
            # Apply header formatting
            if is_header:
                pdf.set_font('Arial', 'B', font_size)
                pdf.ln(5)
                pdf.cell(0, 8, clean_para, 0, 1)
                pdf.ln(3)
                # Reset font
                pdf.set_font('Arial', '', 12)
            else:
                # Regular paragraph - handle line wrapping
                pdf.set_font('Arial', '', 12)
                
                # Check if it's a bold sub-header (like **6.1.2.1.**)
                if clean_para.startswith('**') and '**' in clean_para[2:]:
                    # Extract the bold part and regular part
                    end_bold = clean_para.find('**', 2)
                    if end_bold != -1:
                        bold_part = clean_para[2:end_bold]
                        regular_part = clean_para[end_bold+2:].strip()
                        
                        # Print bold part
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 6, bold_part, 0, 1)
                        pdf.set_font('Arial', '', 12)
                        
                        # Print regular part if exists
                        if regular_part:
                            words = regular_part.split(' ')
                            line = ''
                            for word in words:
                                test_line = line + ' ' + word if line else word
                                if pdf.get_string_width(test_line) < (pdf.w - pdf.l_margin - pdf.r_margin):
                                    line = test_line
                                else:
                                    if line:
                                        pdf.cell(0, 6, line, 0, 1)
                                    line = word
                            
                            if line:
                                pdf.cell(0, 6, line, 0, 1)
                        
                        pdf.ln(4)
                        continue
                
                # Regular text wrapping
                words = clean_para.split(' ')
                line = ''
                for word in words:
                    test_line = line + ' ' + word if line else word
                    if pdf.get_string_width(test_line) < (pdf.w - pdf.l_margin - pdf.r_margin):
                        line = test_line
                    else:
                        if line:
                            pdf.cell(0, 6, line, 0, 1)
                        line = word
                
                # Print remaining line
                if line:
                    pdf.cell(0, 6, line, 0, 1)
                
                pdf.ln(4)  # Space between paragraphs
    
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
    english_text = ""
    print(english_text)
    english_page_count = 0
    total_page_count = len(doc)

    for page in doc:
        page_text = page.get_text().strip()
        print(page_text)
        if page_text:
            try:
                lang = detect(page_text)
                if lang == "en":
                    english_text += "\n" + page_text
                    english_page_count += 1
            except:
                pass

    if english_page_count == 0:
        st.error("No English pages detected in the document.")
        st.stop()

    st.success(f"Total pages: {total_page_count} | English pages: {english_page_count}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=50)
    chunks = splitter.split_text(english_text)
    print(chunks)

    st.info(f"Summarizing {english_page_count} English pages across {len(chunks)} chunks...")

    full_summary = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1} of {len(chunks)}..."):
            messages = [
                SystemMessage(content="You are a professional IRDAI summarizer. Follow all instructions strictly."),
                HumanMessage(content=get_summary_prompt(chunk))
            ]
            response = llm(messages)
            full_summary += "\n\n" + response.content.strip()

    # Deduplication logic
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
