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

# ... (keep all the existing configuration code and functions as they are)

def detect_english_sentences(text):
    """
    Extract only English sentences from the given text
    """
    # Split text into sentences using multiple delimiters
    # This regex splits on periods, exclamation marks, question marks, and newlines
    # while trying to preserve sentence structure
    sentence_endings = r'[.!?]+\s*'
    potential_sentences = re.split(sentence_endings, text)
    
    english_sentences = []
    
    for sentence in potential_sentences:
        sentence = sentence.strip()
        
        # Skip very short sentences (likely fragments)
        if len(sentence) < 10:
            continue
            
        # Skip sentences that are mostly numbers or special characters
        if len(re.sub(r'[^a-zA-Z]', '', sentence)) < 5:
            continue
            
        try:
            # Detect language of this sentence
            detected_lang = detect(sentence)
            
            if detected_lang == "en":
                english_sentences.append(sentence)
                
        except Exception as e:
            # If detection fails, try a simple heuristic
            # Check if sentence contains mostly English characters
            english_chars = len(re.findall(r'[a-zA-Z]', sentence))
            total_chars = len(re.sub(r'\s', '', sentence))
            
            if total_chars > 0 and (english_chars / total_chars) > 0.7:
                english_sentences.append(sentence)
    
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

# Modified main processing section
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
