import streamlit as st
import fitz
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.text_splitters import RecursiveCharacterTextSplitter
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

def extract_text_with_formatting(pdf_document):
    """Extract text with formatting information from PDF"""
    formatted_content = []
    
    for page_num, page in enumerate(pdf_document):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            # Get block position for alignment detection
            block_bbox = block.get("bbox", [0, 0, 0, 0])
            block_x0 = block_bbox[0]
                
            for line in block["lines"]:
                line_text = ""
                line_formats = []
                
                # Get line position
                line_bbox = line.get("bbox", [0, 0, 0, 0])
                line_x0 = line_bbox[0]
                
                for span in line["spans"]:
                    text = span["text"]
                    font_flags = span["flags"]
                    font_size = span["size"]
                    
                    # Detect formatting
                    is_bold = font_flags & 2**4  # Bold flag
                    is_italic = font_flags & 2**1  # Italic flag
                    
                    # Store text with its formatting
                    line_formats.append({
                        "text": text,
                        "bold": bool(is_bold),
                        "italic": bool(is_italic),
                        "size": font_size
                    })
                    line_text += text
                
                if line_text.strip():
                    formatted_content.append({
                        "text": line_text.strip(),
                        "formats": line_formats,
                        "page": page_num,
                        "indent": line_x0  # Store indentation for alignment
                    })
    
    return formatted_content

def format_to_markdown(formatted_content):
    """Convert formatting info to markdown for LLM processing"""
    markdown_text = ""
    
    # Calculate indentation levels
    if formatted_content:
        indents = [item["indent"] for item in formatted_content]
        min_indent = min(indents)
        
        # Create indent levels (every 20 pixels is roughly one level)
        indent_threshold = 20
    
    for item in formatted_content:
        # Calculate indent level
        indent_level = 0
        if formatted_content:
            indent_diff = item["indent"] - min_indent
            indent_level = int(indent_diff / indent_threshold)
        
        # Add indent markers
        indent_prefix = "→" * indent_level if indent_level > 0 else ""
        
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
        
        # Preserve numbering patterns (1., a., i., (1), etc.)
        markdown_text += indent_prefix + line_md + "\n"
    
    return markdown_text

def get_summary_prompt(text):
    return f"""
You are an expert legal analyst and summarization specialist.
You will be given the full content of a legal/regulatory circular WITH FORMATTING MARKUP.

**CRITICAL FORMATTING RULES**: 
1. The input text contains markdown formatting (**bold**, *italic*, ***bold-italic***).
   YOU MUST PRESERVE THIS FORMATTING IN YOUR SUMMARY.
2. The input contains indent markers (→) that indicate hierarchical structure and indentation levels.
   YOU MUST PRESERVE THESE INDENT MARKERS in the summary to maintain the same alignment.
3. The input contains numbering patterns (1., 2., a., b., i., ii., (1), (a), etc.).
   YOU MUST PRESERVE THE EXACT NUMBERING PATTERN AND STYLE in the summary.

When summarizing:
- Keep **bold** text as **bold** in the summary
- Keep *italic* text as *italic* in the summary
- Keep ***bold-italic*** text as ***bold-italic*** in the summary
- Keep → indent markers at the beginning of lines to preserve indentation
- Keep numbering patterns exactly as they appear (e.g., if input has "1.", summary should use "1." not "1)")

Your task is to produce a structured, concise, but meaning-preserving summary that follows these strict rules:

1. General Summarization Rules
The essence and meaning of every section must be preserved exactly — legal words and intent must remain unchanged.

Do not omit or alter important legal/regulatory terms (e.g., "shall", "subject to", "unless", "after", etc.).

Summaries should be shorter than the original but still capture the complete meaning.

Keep chapter and section names as in the original circular WITH THEIR ORIGINAL FORMATTING AND INDENTATION.

Maintain the order of sections and subsections exactly as they appear in the source.

PRESERVE ALL NUMBERING PATTERNS (1., a., i., (1), etc.) EXACTLY AS THEY APPEAR.

2. Definitions
Definition summaries should be mid-length — not too short to lose meaning, not too long to be redundant.

If a definition starts on one page and continues on the next, do not repeat the "Definitions" heading again.
Continue under the same section.

3. Handling Subpoints
If a section has subpoints (a, b, c, d):

You may combine the meaning of a and b into summary point a,
and c and d into summary point b — only if meaning remains intact.

PRESERVE THE ORIGINAL NUMBERING STYLE (if original uses "a.", use "a." not "(a)")

If a point (e.g., b) has sub-subpoints (1, 2, 3, 4):

Summarize them strictly under the correct parent point WITH PROPER INDENTATION (→).

Example: Point b has subpoints 1, 2, 3, 4.
→ Summary under b should contain two points: one summarizing 1 and 2, the other summarizing 3 and 4.

If there are 5 subpoints, split grouping accordingly without changing meaning.

4. Special Elements
Panel Names: Must always be included in the summary.

Tables: Must preserve all rows. No row can be omitted. Summarize only textual descriptions if needed, but keep table data intact.

Miscellaneous Sections: Even if short, must be summarized with original intent preserved.

Regulatory Timelines: Must be explicitly retained in summary (do not shorten to the point of losing exact dates).

5. Exclusions
Do not include:

Authorized signatories

File names in headers/footers

Page numbers

Decorative lines, symbols, or watermarks

6. Output Formatting
Keep original section headings WITH THEIR FORMATTING AND INDENTATION (e.g., "**Section 1: Definitions**", "→**Chapter III**").

For each section:

Write the section title WITH FORMATTING AND INDENTATION.

Write the summarized content matching the original numbering style and indentation level.

PRESERVE markdown formatting (**bold**, *italic*, ***bold-italic***) throughout.

PRESERVE indent markers (→) for hierarchical structure.

PRESERVE numbering patterns (1., a., i., (1), etc.) exactly.

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
    
    # Filter English content
    english_formatted_content = []
    for item in formatted_content:
        try:
            lang = detect(item["text"])
            if lang == "en":
                english_formatted_content.append(item)
        except:
            pass
    
    # Convert to markdown
    markdown_text = format_to_markdown(english_formatted_content)
    markdown_text = remove_specific_text_pattern(markdown_text)
    
    st.success(f"Total pages: {len(doc)} | English paragraphs: {len(english_formatted_content)}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=50)
    chunks = splitter.split_text(markdown_text)
    
    full_summary = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1} of {len(chunks)}..."):
            messages = [
                SystemMessage(content="You are a professional IRDAI summarizer. Follow all instructions strictly. PRESERVE ALL MARKDOWN FORMATTING."),
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
    st.markdown(full_summary)  # Use markdown to display formatted text
    
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
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                spaceAfter=6,
                alignment=TA_LEFT,
            )
            
            # Styles for different indent levels
            indent_styles = {}
            for i in range(10):
                indent_styles[i] = ParagraphStyle(
                    f'Indent{i}',
                    parent=normal_style,
                    leftIndent=i * 20,  # 20 points per indent level
                )
            
            def markdown_to_reportlab(text):
                """Convert markdown formatting to ReportLab XML with proper validation"""
                # First escape special characters BEFORE adding tags
                text = (text.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;')
                           .replace('"', '&quot;'))
                
                # Clean up malformed markdown first
                # Remove stray asterisks and fix common issues
                text = re.sub(r'\*{4,}', '***', text)  # Reduce 4+ asterisks to 3
                text = re.sub(r'(?<!\*)\*(?!\*)', '', text)  # Remove single asterisks not part of pairs
                
                # Now convert markdown to XML tags in order
                # Bold and italic: ***text*** - do this FIRST
                text = re.sub(r'\*\*\*([^\*]+?)\*\*\*', r'<b><i>\1</i></b>', text)
                # Bold: **text**
                text = re.sub(r'\*\*([^\*]+?)\*\*', r'<b>\1</b>', text)
                # Italic: *text* (rare at this point)
                text = re.sub(r'\*([^\*]+?)\*', r'<i>\1</i>', text)
                
                # Remove any remaining asterisks
                text = text.replace('*', '')
                
                # Clean up malformed or nested tags
                def fix_nested_tags(t):
                    """Fix improperly nested tags"""
                    # Remove empty tags
                    t = re.sub(r'<b>\s*</b>', '', t)
                    t = re.sub(r'<i>\s*</i>', '', t)
                    t = re.sub(r'<b><i>\s*</i></b>', '', t)
                    
                    # Fix duplicate tags
                    t = re.sub(r'<b>(<b>)+', '<b>', t)
                    t = re.sub(r'(</b>)+</b>', '</b>', t)
                    t = re.sub(r'<i>(<i>)+', '<i>', t)
                    t = re.sub(r'(</i>)+</i>', '</i>', t)
                    
                    # Fix common nesting issues like <b><i></b></i>
                    t = re.sub(r'<b><i>([^<]*)</b></i>', r'<b><i>\1</i></b>', t)
                    t = re.sub(r'<i><b>([^<]*)</i></b>', r'<b><i>\1</i></b>', t)
                    
                    return t
                
                text = fix_nested_tags(text)
                
                return text
            
            story = []
            
            story.append(Paragraph("IRDAI Circular Summary", title_style))
            story.append(Spacer(1, 20))
            
            lines = summary_text.split('\n')
            
            for line in lines:
                if line.strip():
                    # Count indent markers (→)
                    indent_level = line.count('→')
                    # Remove indent markers from the text
                    clean_line = line.replace('→', '')
                    
                    formatted_line = markdown_to_reportlab(clean_line)
                    
                    # Choose appropriate style based on indent level
                    style = indent_styles.get(indent_level, normal_style)
                    
                    try:
                        story.append(Paragraph(formatted_line, style))
                    except Exception as e:
                        # If paragraph fails, try without any formatting
                        st.warning(f"Skipping formatting for line due to error: {str(e)[:100]}")
                        plain_line = re.sub(r'<[^>]+>', '', formatted_line)  # Strip all tags
                        story.append(Paragraph(plain_line, style))
                else:
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError:
            st.error("ReportLab library is required for PDF generation. Please install it using: pip install reportlab")
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
    except ImportError:
        st.error("ReportLab library is required for PDF generation. Please install it using: pip install reportlab")
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.download_button(
            label="Download Summary as Text File",
            data=full_summary,
            file_name="irdai_summary.txt",
            mime="text/plain"
        )
