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

load_dotenv()
AZURE_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

if not AZURE_API_KEY:
    st.warning(" *AZURE_OPENAI_API_KEY environment variable not set!* Please set your API key to enable advanced formula extraction from documents.")
    MOCK_MODE = True
    llm = None
else:
    try:
        llm = AzureChatOpenAI(
            deployment_name=DEPLOYMENT_NAME,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            temperature=0.3
        )
        MOCK_MODE = False

    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        MOCK_MODE = True
        llm = None

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

st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if MOCK_MODE or llm is None:
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
                HumanMessage(content=get_summary_prompt(chunk, english_page_count))
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

    # Generate DOCX without bold
    def generate_docx(summary_text):
        doc = Document()
        doc.add_heading("IRDAI Circular Summary", level=1)

        paragraphs = summary_text.strip().split("\n\n")
        for para in paragraphs:
            clean_para = para.strip()
            if clean_para:
                doc.add_paragraph(clean_para)

        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    # Download button
    docx_file = generate_docx(full_summary)
    st.download_button(
        label="Download Summary as .docx",
        data=docx_file,
        file_name="irdai_summary.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
