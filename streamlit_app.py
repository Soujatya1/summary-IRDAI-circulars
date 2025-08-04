import streamlit as st
import fitz # PyMuPDF
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

# Streamlit UI
st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if llm is None:
        st.error("Cannot process document: Azure OpenAI client not properly configured.")
        st.stop()
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    english_text = ""
    print(english_text)
    english_text_count = 0
    total_page_count = len(doc)
    
    for page in doc:
        page_text = page.get_text().strip()
        print(page_text)
        if page_text:
            text_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            for text_line in text_lines:
                if len(text_line.split()) < 1:
                    continue
                try:
                    lang = detect(text_line)
                    if lang == "en":
                        english_text += "\n\n" + text_line
                        english_text_count += 1
                except:
                    pass
    
    # Remove the specific text pattern after extracting all English text
    english_text = remove_specific_text_pattern(english_text)
    
    st.success(f"Total pages: {total_page_count} | English paragraphs: {english_text_count}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=50)
    chunks = splitter.split_text(english_text)
    print(chunks)
    
    #st.info(f"Summarizing {english_page_count} English paragraphs across {len(chunks)} chunks...")
    
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
    
    # Replace your existing generate_pdf function with this updated version

    def generate_pdf(summary_text):
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            import re
            
            buffer = BytesIO()
            
            # Create the PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles and create custom styles
            styles = getSampleStyleSheet()
            
            # Title style
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor='black'
            )
            
            # Regular preformatted style
            preformatted_style = ParagraphStyle(
                'Preformatted',
                parent=styles['Normal'],
                fontSize=10,
                leading=12,
                spaceAfter=0,
                spaceBefore=0,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                fontName='Times-Roman',
                wordWrap='LTR'
            )
            
            # Bold style for chapter headers and uppercase lines
            bold_style = ParagraphStyle(
                'BoldText',
                parent=styles['Normal'],
                fontSize=10,
                leading=12,
                spaceAfter=0,
                spaceBefore=0,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                fontName='Times-Bold',  # Bold font
                wordWrap='LTR'
            )
            
            # Function to check if a line should be bold
            def should_be_bold(line):
                stripped_line = line.strip()
                
                # Condition 1: Line starts with "Chapter" (any case)
                if stripped_line.lower().startswith('chapter'):
                    return True
                
                # Condition 2: All letters in the line are uppercase
                # Check if line has letters and all letters are uppercase
                if stripped_line and any(c.isalpha() for c in stripped_line) and stripped_line.isupper():
                    return True
                    
                return False
            
            # Build the document content
            story = []
            
            # Add title
            story.append(Paragraph("IRDAI Circular Summary", title_style))
            story.append(Spacer(1, 20))
            
            # Split the text into lines and process each line
            lines = summary_text.split('\n')
            
            for line in lines:
                # Escape HTML characters to prevent ReportLab from interpreting them
                escaped_line = (line.replace('&', '&amp;')
                              .replace('<', '&lt;')
                              .replace('>', '&gt;')
                              .replace('"', '&quot;'))
                
                # Add each line as a separate paragraph
                if escaped_line.strip():
                    # Check if this line should be bold
                    if should_be_bold(line):
                        story.append(Paragraph(escaped_line, bold_style))
                    else:
                        story.append(Paragraph(escaped_line, preformatted_style))
                else:
                    # Add a small spacer for empty lines
                    story.append(Spacer(1, 6))
            
            # Build the PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError:
            st.error("ReportLab library is required for PDF generation. Please install it using: pip install reportlab")
            return None
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            return None
    
    # Download button for PDF
    try:
        pdf_file = generate_pdf(full_summary)
        if pdf_file:
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_file,
                file_name="irdai_summary.pdf",
                mime="application/pdf"
            )
    except ImportError:
        st.error("ReportLab library is required for PDF generation. Please install it using: pip install reportlab")
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        # Fallback to text file
        st.download_button(
            label="Download Summary as Text File",
            data=full_summary,
            file_name="irdai_summary.txt",
            mime="text/plain"
        )
