import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import langdetect
from docx import Document
from io import BytesIO
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="wide"
)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Azure OpenAI Configuration
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
        index=0,
        help="Azure OpenAI API version"
    )

def initialize_azure_openai(endpoint, api_key, deployment_name, api_version):
    """Initialize Azure OpenAI client with provided credentials."""
    try:
        if not all([endpoint, api_key, deployment_name, api_version]):
            return None
            
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

def extract_english_text(text):
    """
    Enhanced function to extract English text from mixed language content.
    Uses langdetect library with fallback to keyword-based detection.
    Filters out headers, footers, and document metadata.
    """
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        english_sentences = []
        
        # Patterns to exclude (headers, footers, metadata)
        exclude_patterns = [
            r'^\d+\s+THE GAZETTE OF INDIA',
            r'PART\s+[IVX]+—SEC',
            r'EXTRAORDINARY',
            r'^Page\s+\d+',
            r'^www\.',
            r'@\w+\.(com|org|gov|in)',
            r'^[A-Z\s]{10,}
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip if too short
            if len(sentence) <= 10:
                continue
                
            # Skip if matches exclude patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
                
            # Check if it's English
            try:
                lang = langdetect.detect(sentence)
                if lang == 'en':
                    english_sentences.append(sentence)
            except LangDetectException:
                # Fallback: keyword-based detection for English
                if re.search(r'\b(the|and|or|of|to|in|for|with|by|from|at|is|are|was|were|this|that|these|those|will|shall|must|can|may|should|would|could)\b', sentence.lower()):
                    english_sentences.append(sentence)
        
        return '. '.join(english_sentences) + '.' if english_sentences else ""
    
    except Exception as e:
        st.warning(f"Language detection error: {e}. Using original text.")
        return text

def get_summary_prompt(text):
    """Generate prompt for summarization with specific instructions."""
    return f"""
You are a domain expert in insurance compliance and regulation.

Your task is to generate a **clean, concise, section-wise summary** of the input document while preserving the **original structure and flow** of the document.

---

### Mandatory Summarization Rules:

1. **Follow the original structure strictly** — maintain the same order of:
   - Section headings
   - Subheadings
   - Bullet points
   - Tables
   - Date-wise event history
   - UIDAI / IRDAI / eGazette circulars

2. **Do NOT rename or reformat section titles** — retain the exact headings from the original file.

3. **Each section should be summarized in 1–5 lines**, proportional to its original length:
   - Keep it brief, but **do not omit the core message**.
   - Avoid generalizations or overly descriptive rewriting.

4. If a section contains **definitions**, summarize them line by line (e.g., Definition A: …).

5. If the section contains **tabular data**, preserve **column-wise details**:
   - Include every row and column in a concise bullet or structured format.
   - Do not merge or generalize rows — maintain data fidelity.

6. If a section contains **violations, fines, or penalties**, mention each item clearly:
   - List out exact violation titles and actions taken or proposed.

7. For **date-wise circulars or history**, ensure that:
   - **No dates are skipped or merged.**
   - Maintain **chronological order**.
   - Mention full references such as "IRDAI Circular dated 12-May-2022".

---

### Output Format:

- Follow the exact **order and structure** of the input file.
- Do **not invent new headings** or sections.
- Avoid decorative formatting, markdown, or unnecessary bolding — use **clean plain text**.

---

Now, generate a section-wise structured summary of the document below:
--------------------
{text}
"""

def summarize_text_with_langchain(text, llm):
    """Summarize text using LangChain with chunking for large documents."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        summaries = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks, 1):
            status_text.text(f"Processing chunk {i} of {len(chunks)}...")
            progress_bar.progress(i / len(chunks))
            
            prompt = get_summary_prompt(chunk)
            response = llm([HumanMessage(content=prompt)])
            summaries.append(response.content.strip())

        progress_bar.empty()
        status_text.empty()
        
        return "\n\n".join(summaries)
    
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""

def generate_docx(summary_text):
    """Generate a Word document from the summary text."""
    try:
        doc = Document()
        doc.add_heading("Document Summary", level=1)
        
        for para in summary_text.split("\n\n"):
            para = para.strip()
            if para:
                doc.add_paragraph(para)
        
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"Error generating DOCX: {str(e)}")
        return None

# Main application
def main():
    st.title("📄 PDF Summarizer")
    st.markdown("Upload a PDF document to generate a structured summary with English text extraction.")
    
    # Initialize Azure OpenAI
    llm = initialize_azure_openai(azure_endpoint, api_key, deployment_name, api_version)
    
    if not llm:
        st.warning("⚠️ Please configure Azure OpenAI settings in the sidebar to continue.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your PDF file",
        type="pdf",
        help="Select a PDF file to summarize"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        
        # Process PDF
        with st.spinner("🔍 Extracting text from PDF..."):
            english_text = ""
            page_count = 0
            
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    total_pages = len(pdf.pages)
                    
                    for i, page in enumerate(pdf.pages, 1):
                        page_count += 1
                        text = page.extract_text()
                        
                        if text:
                            # Extract English content using enhanced function
                            english_content = extract_english_text(text)
                            
                            if english_content.strip():
                                english_text += f"\n\n--- Page {i} ---\n" + english_content
                            else:
                                st.info(f"📄 Page {i}: No English content found")
                        
                        # Update progress
                        progress = i / total_pages
                        st.progress(progress)
                
                st.success(f"📊 Processed {page_count} pages successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                return
        
        # Check if English content was found
        if english_text.strip():
            st.info(f"📝 Extracted English text from {page_count} pages")
            
            # Show text preview
            with st.expander("🔍 View Extracted Text Preview"):
                st.text_area("Extracted English Text", english_text[:1000] + "..." if len(english_text) > 1000 else english_text, height=300)
            
            # Generate summary
            if st.button("🚀 Generate Summary", type="primary"):
                with st.spinner("🤖 Generating summary..."):
                    full_summary = summarize_text_with_langchain(english_text, llm)
                
                if full_summary:
                    st.subheader("📋 Generated Summary")
                    st.text_area("Summary Preview", full_summary, height=500)
                    
                    # Generate and offer download
                    docx_file = generate_docx(full_summary)
                    if docx_file:
                        st.download_button(
                            label="💾 Download Summary (DOCX)",
                            data=docx_file,
                            file_name="Document_Summary.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.error("❌ Failed to generate summary. Please try again.")
        else:
            st.error("❌ No English content found in the uploaded PDF.")
            st.info("💡 Make sure your PDF contains English text and try again.")

if __name__ == "__main__":
    main()
