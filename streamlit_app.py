import streamlit as st
import fitz
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from dotenv import load_dotenv
from docx import Document
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

def remove_specific_text_pattern(text):
    """
    Remove the specific pattern: "Final Order – In the matter of M/s SBI Life Insurance Co. Ltd."
    and its reference line "Ref: IRDAI/E&C;/ORD/MISC/115/09/2024"
    """
    # Pattern to match the specific text with variations in spacing and punctuation
    pattern1 = r"Final\s+Order\s*[–-]\s*In\s+the\s+matter\s+of\s+M/s\s+SBI\s+Life\s+Insurance\s+Co\.\s+Ltd\."
    pattern2 = r"Ref:\s*IRDAI/E&C;/ORD/MISC/115/09/2024"
    
    # Remove both patterns (case insensitive)
    text = re.sub(pattern1, "", text, flags=re.IGNORECASE)
    text = re.sub(pattern2, "", text, flags=re.IGNORECASE)
    
    # Clean up any extra whitespace or newlines left behind
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newlines
    text = text.strip()
    
    return text

def extract_structured_text(doc):
    """
    Extract text while preserving structure and hierarchy
    """
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        
        if page_text.strip():
            # Add page break marker for context
            full_text += f"\n\n--- Page {page_num + 1} ---\n\n"
            full_text += page_text
    
    return full_text

def clean_and_structure_text(text):
    """
    Clean the extracted text while preserving structure
    """
    # Remove the specific unwanted patterns
    text = remove_specific_text_pattern(text)
    
    # Fix common PDF extraction issues
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Fix broken lines that should be connected
    # Connect lines that end with lowercase and next line starts with lowercase
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
    
    # Preserve numbered sections and bullet points
    # Ensure proper spacing around numbered items
    text = re.sub(r'\n(\d+\)|\d+\.|\([a-z]\)|\([i-v]+\))', r'\n\n\1', text)
    
    # Clean up page markers and excessive spaces
    text = re.sub(r'--- Page \d+ ---', '', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    return text.strip()

def is_primarily_english(text_chunk):
    """
    Check if a text chunk is primarily English
    """
    if not text_chunk or len(text_chunk.strip()) < 10:
        return False
    
    # Sample first 500 characters for language detection
    sample = text_chunk[:500]
    
    try:
        detected_lang = detect(sample)
        return detected_lang == 'en'
    except:
        # If detection fails, check for English indicators
        english_indicators = ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'as']
        words = sample.lower().split()
        english_word_count = sum(1 for word in words if word in english_indicators)
        return english_word_count >= 3

def structure_aware_split(text, max_chunk_size=3500):
    """
    Split text while trying to preserve structural boundaries
    """
    # First, try to split by major sections
    major_sections = re.split(r'\n\n(?=Chapter [IVX]+|SECTION [IVX]+|\d+\.\s+[A-Z])', text)
    
    chunks = []
    current_chunk = ""
    
    for section in major_sections:
        if len(current_chunk) + len(section) <= max_chunk_size:
            current_chunk += "\n\n" + section if current_chunk else section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single section is too long, split it further
            if len(section) > max_chunk_size:
                # Split by subsections
                subsections = re.split(r'\n\n(?=\d+\)|[a-z]\)|[i-v]+\))', section)
                temp_chunk = ""
                
                for subsection in subsections:
                    if len(temp_chunk) + len(subsection) <= max_chunk_size:
                        temp_chunk += "\n\n" + subsection if temp_chunk else subsection
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = subsection
                
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = section
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def get_summary_prompt(text):
    return f"""
You are acting as a **Senior Legal Analyst** and Regulatory Compliance Officer specializing in IRDAI, UIDAI, and eGazette circulars.
 
Your task is to generate a **legally precise, clause-preserving, structure-aligned summary** of the input regulatory document. Your summary will be reviewed for legal compliance, so accuracy is critical.

CRITICAL: You MUST preserve the EXACT numbering system and hierarchical structure from the original document.

Use clean plain text.
Do not use Markdown formatting (no **bold**, `code`, or extra spacing).

---
 
### LEGAL SUMMARIZATION RULES
 
**1. STRUCTURE PRESERVATION (Strict Order):**
- PRESERVE EXACT numbering: 1), 2), a), b), i), ii), (1), (2), (a), (b)
- Maintain ALL section headers EXACTLY as they appear
- Keep subsection indentation and hierarchy
- Do NOT renumber or reorganize sections
- If you see "Chapter I", "Chapter II", preserve these EXACTLY

**2. NUMBERING SYSTEM PRESERVATION:**
- Keep original numbering: If original has "1)", your summary must have "1)"
- Keep original lettering: If original has "a)", your summary must have "a)"  
- Keep original roman numerals: If original has "i)", your summary must have "i)"
- Preserve parentheses and periods exactly as in original

**3. CLAUSE-BY-CLAUSE SUMMARIZATION (NO MERGING):**
- Summarize each numbered/lettered clause separately
- Maintain the exact sequence from the original
- Do not combine adjacent points even if they seem similar

**4. PRESERVE LEGAL PHRASES & CAUSALITY TRIGGERS:**
- Never skip or simplify phrases like: "unless", "until", "after", "shall", "subject to", "provided that"
- These are legally binding conditions and must be retained with their meaning intact

**5. DEFINITIONS & EXPLANATORY SECTIONS:**
- If the section contains definitions or classifications:
  - List each term separately preserving original numbering
  - Do not merge multiple definitions into one block

**6. COMMITTEES, PANELS, AUTHORITIES (EXACT NAMES):**
- Retain every mention of committees and positions verbatim
- Never shorten or generalize names
- Repeat full names every time they appear

**7. TABLES – PRESERVE IN FULL:**
- Summarize column-by-column, row-by-row with original structure
- Do not omit any row or column

**8. NUMERIC LIMITS & ABBREVIATIONS:**
- Maintain correct expressions exactly as written
- Preserve currency formats and legal terminology

**9. OUTPUT FORMAT:**
- Use ONLY plain text with NO formatting symbols
- DO NOT use markdown headers, bold, italic, or special characters
- Use simple numbering and lettering exactly as in original
- Use line breaks and indentation to match original structure
- Use CAPITAL LETTERS for emphasis instead of bold

**10. EXAMPLE OF CORRECT STRUCTURE PRESERVATION:**
If original has:
```
Chapter I: General Information
1) Insurers are required to:
   a. make available products
   b. allow for customization
2) Products are available to cover:
   a. Uterine Artery Embolization
   b. Balloon Sinuplasty
```

Your summary should maintain:
```
Chapter I: General Information
1) Insurers must make products available and allow customization
   a. Products must be made available across all categories
   b. Customization must be allowed based on medical conditions
2) Products must cover technological advances including:
   a. Uterine Artery Embolization procedures
   b. Balloon Sinuplasty treatments
```

---

Now begin the section-wise clause-preserving summary, maintaining EXACT numbering and structure:
--------------------
{text}
"""

# Streamlit UI
st.set_page_config(layout="wide")

st.title("IRDAI Document Summarizer (Structure-Preserving)")
st.write("Upload an IRDAI circular PDF to generate a structured summary that preserves numbering and hierarchy.")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if llm is None:
        st.error("Cannot process document: Azure OpenAI client not properly configured.")
        st.stop()
    
    # Extract text with structure preservation
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    with st.spinner("Extracting text from PDF..."):
        # Extract full text maintaining structure
        raw_text = extract_structured_text(doc)
        
        # Clean and structure the text
        structured_text = clean_and_structure_text(raw_text)
        
        # Check if primarily English
        if not is_primarily_english(structured_text):
            st.warning("The document doesn't appear to be primarily in English. Results may vary.")
    
    st.success(f"Successfully extracted text from {len(doc)} pages")
    
    # Show preview of extracted text structure
    with st.expander("Preview Extracted Text Structure"):
        preview = structured_text[:2000] + "..." if len(structured_text) > 2000 else structured_text
        st.text_area("Structured Text Preview:", value=preview, height=300)
    
    # Split text with structure awareness
    with st.spinner("Preparing text chunks while preserving structure..."):
        chunks = structure_aware_split(structured_text, max_chunk_size=3500)
    
    st.info(f"Processing {len(chunks)} structured chunks...")
    
    # Process chunks and generate summary
    full_summary = ""
    
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing chunk {i + 1} of {len(chunks)}..."):
            try:
                messages = [
                    SystemMessage(content="You are a professional IRDAI document summarizer. Preserve ALL numbering and structural hierarchy EXACTLY as in the original."),
                    HumanMessage(content=get_summary_prompt(chunk))
                ]
                response = llm(messages)
                chunk_summary = response.content.strip()
                
                if chunk_summary:
                    full_summary += "\n\n" + chunk_summary
                    
            except Exception as e:
                st.error(f"Error processing chunk {i + 1}: {str(e)}")
                continue
    
    # Clean up the final summary
    def clean_final_summary(text):
        # Remove excessive newlines but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove duplicate section headers
        lines = text.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            if line.strip() != prev_line.strip() or not line.strip():
                cleaned_lines.append(line)
            prev_line = line
            
        return '\n'.join(cleaned_lines).strip()
    
    full_summary = clean_final_summary(full_summary)
    
    # Display the summary
    st.subheader("📋 Structure-Preserving Summary")
    st.text_area("Generated Summary:", value=full_summary, height=600)
    
    # Generate PDF
    def generate_pdf(summary_text):
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
            
            buffer = BytesIO()
            
            # Create the PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Get styles and create custom styles
            styles = getSampleStyleSheet()
            
            # Title style
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor='black'
            )
            
            # Body style
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leftIndent=0,
                rightIndent=0
            )
            
            # Heading style for sections
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
                alignment=TA_LEFT,
                textColor='black'
            )
            
            # Build the document content
            story = []
            
            # Add title
            story.append(Paragraph("IRDAI Circular Summary", title_style))
            story.append(Spacer(1, 20))
            
            # Process the summary text line by line to preserve structure
            lines = summary_text.split('\n')
            current_para = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_para:
                        story.append(Paragraph(current_para, body_style))
                        current_para = ""
                    story.append(Spacer(1, 6))
                elif (line.startswith('Chapter') or 
                      line.isupper() and len(line) < 100 or
                      line.endswith(':') and len(line) < 100):
                    if current_para:
                        story.append(Paragraph(current_para, body_style))
                        current_para = ""
                    story.append(Paragraph(line, heading_style))
                else:
                    if current_para and not line.startswith(('1)', '2)', 'a)', 'b)', 'i)', 'ii)')):
                        current_para += " " + line
                    else:
                        if current_para:
                            story.append(Paragraph(current_para, body_style))
                        current_para = line
            
            if current_para:
                story.append(Paragraph(current_para, body_style))
            
            # Build the PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError:
            st.error("ReportLab library is required for PDF generation. Please install it using: pip install reportlab")
            return None
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as text file
        st.download_button(
            label="📄 Download as Text File",
            data=full_summary,
            file_name="irdai_summary_structured.txt",
            mime="text/plain"
        )
    
    with col2:
        # Try to generate PDF
        try:
            pdf_file = generate_pdf(full_summary)
            if pdf_file:
                st.download_button(
                    label="📑 Download as PDF",
                    data=pdf_file,
                    file_name="irdai_summary_structured.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** This tool preserves the original document structure including numbering, sections, and legal hierarchy for accurate regulatory compliance.")
