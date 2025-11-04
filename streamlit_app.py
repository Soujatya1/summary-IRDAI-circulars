import streamlit as st
import fitz
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    pattern1 = r"Final\s+Order\s*[–-]\s*In\s+the\s+matter\s+of\s+M/s\s+SBI\s+Life\s+Insurance\s+Co\.\s+Ltd\."
    pattern2 = r"Ref:\s*IRDAI/E&C;/ORD/MISC/115/09/2024"
    
    text = re.sub(pattern1, "", text, flags=re.IGNORECASE)
    text = re.sub(pattern2, "", text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def is_likely_header(text, font_size, avg_font_size, is_bold, is_centered=False):
    """Determine if text is likely a header based on multiple factors"""
    text_upper = text.strip().upper()
    text_stripped = text.strip()
    
    # Check for common header patterns
    header_patterns = [
        r'^CHAPTER[\s-]+[IVX\d]+',
        r'^PRELIMINARY$',
        r'^DEFINITIONS$',
        r'^SCOPE$',
        r'^[IVX]+\.\s+[A-Z]',
        r'^\d+\.\s+[A-Z][a-z].*:$',
    ]
    
    is_header_pattern = any(re.match(pattern, text_stripped, re.IGNORECASE) for pattern in header_patterns)
    
    # All caps and short (likely a section header)
    is_all_caps_short = (text_stripped == text_upper and len(text_stripped) < 50 and len(text_stripped) > 2)
    
    # Larger font size
    is_larger_font = font_size > avg_font_size * 1.1
    
    # Ends with colon (common for headers)
    ends_with_colon = text_stripped.endswith(':')
    
    return (is_header_pattern or 
            (is_all_caps_short and (is_bold or is_larger_font or is_centered)) or
            (is_bold and is_larger_font and len(text_stripped) < 100) or
            (ends_with_colon and is_bold))

def extract_text_with_formatting(pdf_document):
    """Extract text with formatting and structural information from PDF"""
    formatted_content = []
    
    # Calculate average font size for header detection
    all_font_sizes = []
    for page in pdf_document:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_font_sizes.append(span["size"])
    
    avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 11
    
    for page_num, page in enumerate(pdf_document):
        blocks = page.get_text("dict")["blocks"]
        page_width = page.rect.width
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            block_text = ""
            block_formats = []
            block_bbox = block.get("bbox", [0, 0, page_width, 0])
            
            # Check if block is centered
            block_center = (block_bbox[0] + block_bbox[2]) / 2
            is_centered = abs(block_center - page_width / 2) < page_width * 0.1
            
            for line in block["lines"]:
                line_text = ""
                line_formats = []
                line_font_sizes = []
                
                for span in line["spans"]:
                    text = span["text"]
                    font_flags = span["flags"]
                    font_size = span["size"]
                    
                    # Detect formatting
                    is_bold = font_flags & 2**4  # Bold flag
                    is_italic = font_flags & 2**1  # Italic flag
                    
                    line_font_sizes.append(font_size)
                    
                    # Store text with its formatting
                    line_formats.append({
                        "text": text,
                        "bold": bool(is_bold),
                        "italic": bool(is_italic),
                        "size": font_size
                    })
                    line_text += text
                
                if line_text.strip():
                    block_text += line_text
                    block_formats.extend(line_formats)
            
            if block_text.strip():
                # Determine if this is a header
                max_font_size = max([f["size"] for f in block_formats]) if block_formats else avg_font_size
                has_bold = any(f["bold"] for f in block_formats)
                
                is_header = is_likely_header(block_text.strip(), max_font_size, avg_font_size, has_bold, is_centered)
                
                formatted_content.append({
                    "text": block_text.strip(),
                    "formats": block_formats,
                    "page": page_num,
                    "is_header": is_header,
                    "is_centered": is_centered,
                    "max_font_size": max_font_size
                })
    
    return formatted_content

def format_to_markdown(formatted_content):
    """Convert formatting info to markdown while preserving structure and headers"""
    markdown_lines = []
    
    for item in formatted_content:
        line_text = item["text"]
        is_header = item.get("is_header", False)
        
        # Check if this line looks like a numbered/bulleted item
        is_structured = re.match(r'^\s*(\d+\.|\d+\.\d+\.?|\([a-z]\)|\([ivxlcdm]+\)|[a-z]\)|[•\-\*])\s', line_text)
        
        # Special handling for headers
        if is_header:
            # Determine header level
            if re.match(r'^CHAPTER[\s-]+[IVX\d]+', line_text, re.IGNORECASE):
                markdown_lines.append(f"\n## {line_text}\n")
            elif line_text.strip().upper() == line_text.strip() and len(line_text.strip()) < 50:
                # All caps header (like PRELIMINARY, DEFINITIONS)
                markdown_lines.append(f"\n### {line_text}\n")
            else:
                # Other headers - make them bold
                markdown_lines.append(f"\n**{line_text}**\n")
            continue
        
        # Format the text
        line_md = ""
        for span in item["formats"]:
            text = span["text"]
            if span["bold"] and span["italic"]:
                text = f"***{text}***"
            elif span["bold"]:
                text = f"**{text}**"
            elif span["italic"]:
                text = f"*{text}*"
            line_md += text
        
        markdown_lines.append(line_md)
    
    return "\n".join(markdown_lines)

def get_summary_prompt(text):
    return f"""
You are an expert legal analyst and summarization specialist.
You will be given the full content of a legal/regulatory circular WITH FORMATTING MARKUP.

**CRITICAL FORMATTING RULE**: The input text contains markdown formatting (**bold**, *italic*, ***bold-italic***, ## headers, ### subheaders).
YOU MUST PRESERVE THIS FORMATTING IN YOUR SUMMARY. When summarizing:
- Keep **bold** text as **bold** in the summary
- Keep *italic* text as *italic* in the summary
- Keep ***bold-italic*** text as ***bold-italic*** in the summary
- Keep ## headers as ## headers
- Keep ### subheaders as ### subheaders

**CRITICAL STRUCTURE RULE**: Preserve ALL structural elements:
- Chapter headers (## CHAPTER-I, ## CHAPTER-II, etc.)
- Section subheaders (### PRELIMINARY, ### DEFINITIONS, etc.)
- All numbered sections and subsections with their exact numbering

**CRITICAL NUMBERING RULE**: The input text contains numbered sections (1., 2., 1.1, 1.2, (a), (b), etc.).
YOU MUST PRESERVE THE EXACT NUMBERING SCHEME from the original document. 
- If the original uses "1.", "2.", "3." - use the same in summary
- If the original uses "1.1", "1.2" - use the same in summary
- If the original uses "(a)", "(b)" - use the same in summary
- DO NOT change the numbering scheme

Your task is to produce a structured, concise, but meaning-preserving summary that follows these strict rules:

1. General Summarization Rules
The essence and meaning of every section must be preserved exactly — legal words and intent must remain unchanged.

Do not omit or alter important legal/regulatory terms (e.g., "shall", "subject to", "unless", "after", etc.).

Summaries should be shorter than the original but still capture the complete meaning.

Keep ALL chapter and section headers exactly as in the original circular WITH THEIR ORIGINAL FORMATTING AND NUMBERING.

Maintain the order of sections and subsections exactly as they appear in the source.

2. Definitions
Definition summaries should be mid-length — not too short to lose meaning, not too long to be redundant.

If a definition starts on one page and continues on the next, do not repeat the "Definitions" heading again.
Continue under the same section.

3. Handling Subpoints
PRESERVE THE EXACT NUMBERING SCHEME from the original (1., 2., (a), (b), 1.1, 1.2, etc.)

If a section has subpoints (a, b, c, d):
You may combine the meaning of a and b into summary point a,
and c and d into summary point b — only if meaning remains intact.

If a point (e.g., b) has sub-subpoints (1, 2, 3, 4):
Summarize them strictly under the correct parent point.

4. Special Elements
Panel Names: Must always be included in the summary.

Tables: Must preserve all rows. No row can be omitted. Summarize only textual descriptions if needed, but keep table data intact.

Miscellaneous Sections: Even if short, must be summarized with original intent preserved.

Regulatory Timelines: Must be explicitly retained in summary.

5. Exclusions
Do not include:
Authorized signatories
File names in headers/footers
Page numbers
Decorative lines, symbols, or watermarks

6. Output Formatting
Keep ALL original structural elements:
- ## CHAPTER headers
- ### Section subheaders
- **Bold section titles**
- Original numbering for all points and subpoints

PRESERVE markdown formatting throughout.
PRESERVE original numbering throughout.
Keep tables in tabular format in the summary.

---

Now begin the **section-wise, clause-wise, interpretation-based summarization** of the following legal document:
--------------------
{text}
"""

st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Upload an IRDAI Circular PDF", type="pdf")

if uploaded_file:
    if llm is None:
        st.error("Cannot process document: Azure OpenAI client not properly configured.")
        st.stop()
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Extract text with formatting
    formatted_content = extract_text_with_formatting(doc)
    
    # Filter English content (but be more lenient with short headers)
    english_formatted_content = []
    for item in formatted_content:
        try:
            text = item["text"]
            # For short text (like headers), be more lenient
            if len(text.strip()) < 30 or item.get("is_header", False):
                english_formatted_content.append(item)
            else:
                lang = detect(text)
                if lang == "en":
                    english_formatted_content.append(item)
        except:
            # If detection fails, include it anyway (might be a header)
            if len(item["text"].strip()) < 50:
                english_formatted_content.append(item)
    
    # Convert to markdown
    markdown_text = format_to_markdown(english_formatted_content)
    markdown_text = remove_specific_text_pattern(markdown_text)
    
    st.success(f"Total pages: {len(doc)} | Content blocks: {len(english_formatted_content)}")
    
    # Show preview of extracted structure
    with st.expander("Preview Extracted Structure"):
        preview_lines = markdown_text.split('\n')[:50]
        st.code('\n'.join(preview_lines), language='markdown')
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500, 
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(markdown_text)
    
    full_summary = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1} of {len(chunks)}..."):
            messages = [
                SystemMessage(content="You are a professional IRDAI summarizer. Follow all instructions strictly. PRESERVE ALL MARKDOWN FORMATTING, HEADERS, AND ORIGINAL NUMBERING SCHEMES."),
                HumanMessage(content=get_summary_prompt(chunk))
            ]
            response = llm.invoke(messages)
            full_summary += "\n\n" + response.content.strip()
    
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
    
    st.subheader("Section-wise Summary")
    st.markdown(full_summary)
    
    def generate_pdf_with_formatting(summary_text):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            import re
            
            buffer = BytesIO()
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor='black'
            )
            
            h2_style = ParagraphStyle(
                'Heading2',
                parent=styles['Heading2'],
                fontSize=13,
                spaceAfter=12,
                spaceBefore=12,
                alignment=TA_LEFT,
                textColor='black'
            )
            
            h3_style = ParagraphStyle(
                'Heading3',
                parent=styles['Heading3'],
                fontSize=11,
                spaceAfter=8,
                spaceBefore=8,
                alignment=TA_LEFT,
                textColor='black'
            )
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                spaceAfter=6,
                alignment=TA_LEFT,
            )
            
            def markdown_to_reportlab(text):
                """Convert markdown formatting to ReportLab XML"""
                text = (text.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;')
                           .replace('"', '&quot;'))
                
                text = re.sub(r'\*{4,}', '***', text)
                text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
                
                text = re.sub(r'\*\*\*([^\*]+?)\*\*\*', r'<b><i>\1</i></b>', text)
                text = re.sub(r'\*\*([^\*]+?)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'\*([^\*]+?)\*', r'<i>\1</i>', text)
                
                text = text.replace('*', '')
                
                return text
            
            story = []
            
            story.append(Paragraph("IRDAI Circular Summary", title_style))
            story.append(Spacer(1, 20))
            
            lines = summary_text.split('\n')
            
            for line in lines:
                if not line.strip():
                    story.append(Spacer(1, 6))
                    continue
                
                # Check for headers
                if line.startswith('## '):
                    header_text = markdown_to_reportlab(line[3:])
                    story.append(Paragraph(header_text, h2_style))
                elif line.startswith('### '):
                    header_text = markdown_to_reportlab(line[4:])
                    story.append(Paragraph(header_text, h3_style))
                else:
                    formatted_line = markdown_to_reportlab(line)
                    try:
                        story.append(Paragraph(formatted_line, normal_style))
                    except Exception as e:
                        plain_line = re.sub(r'<[^>]+>', '', formatted_line)
                        story.append(Paragraph(plain_line, normal_style))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError:
            st.error("ReportLab library is required. Install: pip install reportlab")
            return None
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            return None
    
    try:
        pdf_file = generate_pdf_with_formatting(full_summary)
        if pdf_file:
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_file,
                file_name="irdai_summary.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.download_button(
            label="Download Summary as Text File",
            data=full_summary,
            file_name="irdai_summary.txt",
            mime="text/plain"
        )
