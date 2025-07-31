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
    
    pdf.set_font('Arial', '', 11)  # Slightly smaller font for better fitting
    
    # Clean the entire text first
    clean_text = pdf.clean_text(summary_text.strip())
    
    # Split by lines to preserve exact structure
    lines = clean_text.split('\n')
    
    for line in lines:
        # Handle empty lines (preserve spacing)
        if not line.strip():
            pdf.ln(5)  # Add some vertical space for empty lines
            continue
        
        # Check if line fits on current page, if not add new page
        if pdf.get_y() > pdf.h - 30:  # 30mm from bottom
            pdf.add_page()
        
        # Handle long lines that need wrapping
        words = line.split(' ')
        current_line = ''
        
        for word in words:
            test_line = current_line + ' ' + word if current_line else word
            
            # Check if the test line fits within margins
            if pdf.get_string_width(test_line) <= (pdf.w - pdf.l_margin - pdf.r_margin):
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
