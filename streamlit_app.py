import streamlit as st
import pdfplumber
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import black, blue, grey, darkblue
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


def create_custom_styles():
    """Create custom paragraph styles for different content types"""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=darkblue,
        fontName='Helvetica-Bold'
    ))
    
    # Main heading style
    styles.add(ParagraphStyle(
        name='MainHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=18,
        textColor=darkblue,
        fontName='Helvetica-Bold',
        keepWithNext=True
    ))
    
    # Sub heading style
    styles.add(ParagraphStyle(
        name='SubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textColor=black,
        fontName='Helvetica-Bold',
        keepWithNext=True
    ))
    
    # Clause style (for numbered items)
    styles.add(ParagraphStyle(
        name='Clause',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        spaceBefore=3,
        leftIndent=20,
        fontName='Helvetica'
    ))
    
    # Sub-clause style (for sub-items)
    styles.add(ParagraphStyle(
        name='SubClause',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        spaceBefore=2,
        leftIndent=40,
        fontName='Helvetica'
    ))
    
    # Definition style
    styles.add(ParagraphStyle(
        name='Definition',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        spaceBefore=3,
        leftIndent=30,
        fontName='Helvetica',
        textColor=black
    ))
    
    # Body text style
    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    ))
    
    return styles

def parse_summary_content(summary_text):
    """Parse the summary text and identify different content types"""
    lines = summary_text.split('\n')
    parsed_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identify content type based on patterns
        if re.match(r'^[A-Z\s]+:?\s*$', line) and len(line) > 10:
            # Main heading (all caps)
            parsed_content.append(('main_heading', line))
        elif re.match(r'^\d+\.', line):
            # Numbered clause
            parsed_content.append(('clause', line))
        elif re.match(r'^[a-z]\)', line) or re.match(r'^\([a-z]\)', line):
            # Sub-clause with letter
            parsed_content.append(('sub_clause', line))
        elif re.match(r'^\s*-\s*Definition:', line) or 'Definition:' in line:
            # Definition
            parsed_content.append(('definition', line))
        elif re.match(r'^[A-Z][^:]*:', line):
            # Sub heading (starts with capital, has colon)
            parsed_content.append(('sub_heading', line))
        elif line.startswith('•') or line.startswith('-'):
            # Bullet point
            parsed_content.append(('sub_clause', line))
        else:
            # Regular body text
            parsed_content.append(('body_text', line))
    
    return parsed_content

def create_header_footer(canvas, doc):
    """Create header and footer for each page"""
    canvas.saveState()
    
    # Header
    canvas.setFont('Helvetica-Bold', 10)
    canvas.setFillColor(darkblue)
    canvas.drawString(72, doc.height + 50, "IRDAI Regulatory Document Summary")
    canvas.drawRightString(doc.width + 72, doc.height + 50, 
                          f"Generated on {datetime.now().strftime('%d-%m-%Y')}")
    
    # Header line
    canvas.setStrokeColor(grey)
    canvas.setLineWidth(0.5)
    canvas.line(72, doc.height + 40, doc.width + 72, doc.height + 40)
    
    # Footer
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(grey)
    canvas.drawCentredText(doc.width/2 + 72, 30, 
                           f"Page {canvas.getPageNumber()}")
    
    # Footer line
    canvas.line(72, 50, doc.width + 72, 50)
    
    canvas.restoreState()

def generate_summary_pdf(summary_text, original_filename="document"):
    """Generate a well-formatted PDF from the LLM summary"""
    
    # Create a BytesIO buffer
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=100,
        bottomMargin=72
    )
    
    # Get custom styles
    styles = create_custom_styles()
    
    # Story list to hold all content
    story = []
    
    # Add title page
    story.append(Paragraph("REGULATORY DOCUMENT SUMMARY", styles['CustomTitle']))
    story.append(Spacer(1, 20))
    
    # Add document info
    doc_info = f"""
    <b>Original Document:</b> {original_filename}<br/>
    <b>Summary Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    <b>Processing Method:</b> AI-Powered Legal Analysis<br/>
    <b>Summary Type:</b> Clause-Preserving Structural Summary
    """
    story.append(Paragraph(doc_info, styles['BodyText']))
    story.append(Spacer(1, 30))
    
    # Add separator line
    story.append(Paragraph("_" * 80, styles['BodyText']))
    story.append(Spacer(1, 20))
    
    # Parse and add the summary content
    parsed_content = parse_summary_content(summary_text)
    
    for content_type, text in parsed_content:
        if content_type == 'main_heading':
            story.append(Spacer(1, 12))
            story.append(Paragraph(text, styles['MainHeading']))
        elif content_type == 'sub_heading':
            story.append(Paragraph(text, styles['SubHeading']))
        elif content_type == 'clause':
            story.append(Paragraph(text, styles['Clause']))
        elif content_type == 'sub_clause':
            story.append(Paragraph(text, styles['SubClause']))
        elif content_type == 'definition':
            # Format definitions with special highlighting
            formatted_text = text.replace('Definition:', '<b>Definition:</b>')
            story.append(Paragraph(formatted_text, styles['Definition']))
        else:
            story.append(Paragraph(text, styles['BodyText']))
    
    # Add footer note
    story.append(Spacer(1, 30))
    story.append(Paragraph("_" * 80, styles['BodyText']))
    story.append(Spacer(1, 10))
    
    footer_note = """
    <b>Disclaimer:</b> This summary has been generated using AI technology for regulatory compliance analysis. 
    While every effort has been made to preserve the legal accuracy and structure of the original document, 
    users should refer to the original regulatory text for authoritative legal interpretation.
    """
    story.append(Paragraph(footer_note, styles['BodyText']))
    
    # Build the PDF
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    
    # Get the PDF data
    buffer.seek(0)
    return buffer

def add_pdf_download_functionality():
    """Add this function to your main Streamlit app after getting the summary"""
    
    # Add this after you get the summary from summarize_text_with_langchain
    st.subheader("📄 Generate PDF Summary")
    
    if st.button("🔄 Generate Formatted PDF", type="primary"):
        with st.spinner("Generating PDF summary..."):
            try:
                # Generate the PDF
                pdf_buffer = generate_summary_pdf(
                    final_summary,  # Your summary variable
                    uploaded_file.name if uploaded_file else "regulatory_document"
                )
                
                # Success message
                st.success("✅ PDF generated successfully!")
                
                # Download button
                st.download_button(
                    label="📥 Download PDF Summary",
                    data=pdf_buffer,
                    file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}_" + 
                             f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
                # Show PDF info
                pdf_buffer.seek(0, 2)  # Seek to end
                pdf_size = pdf_buffer.tell()
                st.info(f"📊 PDF Size: {pdf_size / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                logger.error(f"PDF generation error: {e}")

def clean_extracted_text(text):
    text = re.sub(r'\n\n--- Page \d+ ---\n', '\n\n', text)
    text = re.sub(r'--- Page \d+ ---', '', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def generate_download_filename(original_filename):
    name_without_ext = os.path.splitext(original_filename)[0]
    return f"{name_without_ext}_summary.pdf"

if uploaded_file and llm:
    st.header("📄 PDF Processing & Summarization")
    
    with pdfplumber.open(uploaded_file) as pdf:
        st.info(f"📖 Processing PDF with {len(pdf.pages)} pages")
        
        english_text = ""
        page_count = 0
        
        for page in pdf.pages:
            try:
                text = page.extract_text()
                if text and is_english(text):
                    english_text += text + "\n"
                    page_count += 1
            except Exception as e:
                logger.warning(f"Error extracting text from page: {e}")
        
        if english_text.strip():
            st.success(f"✅ Extracted {len(english_text)} characters of English text from {page_count} pages")
            
            # Get summary from LLM
            with st.spinner("🤖 Generating AI summary..."):
                final_summary = summarize_text_with_langchain(english_text)
            
            # Display the summary
            st.markdown("---")
            st.header("📋 Generated Summary")
            
            # Show summary in expandable text area
            with st.expander("📖 View Full Summary", expanded=True):
                st.text_area("Summary Content", final_summary, height=400, key="summary_display")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Summary Length", f"{len(final_summary)} chars")
            with col2:
                st.metric("📉 Compression Ratio", f"{len(final_summary)/len(english_text):.1%}")
            with col3:
                st.metric("📄 Word Count", len(final_summary.split()))
            
            # PDF Generation Section
            st.markdown("---")
            st.header("📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 PDF Summary")
                if st.button("🔄 Generate Formatted PDF", type="primary", key="generate_pdf"):
                    with st.spinner("Creating beautifully formatted PDF..."):
                        try:
                            # Generate the PDF
                            pdf_buffer = generate_summary_pdf(
                                final_summary,
                                uploaded_file.name
                            )
                            
                            # Success message
                            st.success("✅ PDF generated successfully!")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download PDF Summary",
                                data=pdf_buffer,
                                file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}_" + 
                                         f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                type="primary",
                                key="download_pdf"
                            )
                            
                            # Show PDF info
                            pdf_buffer.seek(0, 2)  # Seek to end
                            pdf_size = pdf_buffer.tell()
                            st.info(f"📊 PDF Size: {pdf_size / 1024:.1f} KB")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            logger.error(f"PDF generation error: {e}")
            
            with col2:
                st.subheader("📝 Text Summary")
                st.download_button(
                    label="📥 Download Text Summary",
                    data=final_summary,
                    file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}_" + 
                             f"{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    type="secondary",
                    key="download_txt"
                )
                
                # Also provide original extracted text
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=english_text,
                    file_name=f"extracted_{uploaded_file.name.replace('.pdf', '')}_" + 
                             f"{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    key="download_original"
                )
            
        else:
            st.warning("⚠️ No English text found in the uploaded PDF.")
            st.info("Please ensure the PDF contains English text and is not image-based.")

else:
    # Show instructions when no file is uploaded or LLM not configured
    if not uploaded_file:
        st.info("👆 Please upload a PDF file to begin processing.")
    
    if not llm:
        st.warning("⚙️ Please configure Azure OpenAI settings in the sidebar.")
        
    # Show example of what the app can do
    st.markdown("---")
    st.header("🎯 What This App Does")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📤 Input
        - **PDF Documents** (IRDAI regulations, circulars)
        - **Multi-language support** (extracts English content)
        - **Complex document structures**
        """)
    
    with col2:
        st.markdown("""
        ### 📥 Output
        - **Structured summaries** preserving legal clauses
        - **Professional PDF reports** with formatting
        - **Plain text summaries** for easy sharing
        """)
    
    st.markdown("""
    ### ✨ Key Features
    - 🔍 **Smart text extraction** with header/footer filtering
    - 🧠 **AI-powered summarization** using Azure OpenAI
    - 📋 **Legal clause preservation** maintaining document structure
    - 🎨 **Professional PDF formatting** with headers, styles, and branding
    - 📊 **Processing transparency** showing chunks and progress
    """)
