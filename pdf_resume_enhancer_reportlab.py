import os
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFResumeEnhancer:
    def __init__(self):
        # Initialize the text generation pipeline
        try:
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-large"
            )
        except Exception as e:
            print(f"Error initializing the model: {str(e)}")
            print("Falling back to a simpler model...")
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small"
            )
            
        # Job description from main.py
        self.job_description = """
        As a leading technology innovator, Qualcomm pushes the boundaries of what's possible to enable next-generation experiences and drives digital transformation to help create a smarter, connected future for all. As a Qualcomm Software Engineer, you will design, develop, create, modify, and validate embedded and cloud edge software, applications, and/or specialized utility programs that launch cutting-edge, world class products that meet and exceed customer needs.

        Minimum Qualifications:
        • Bachelor's degree in Engineering, Information Systems, Computer Science, or related field and 3+ years of Software Engineering or related work experience.
        OR
        Master's degree in Engineering, Information Systems, Computer Science, or related field and 2+ years of Software Engineering or related work experience.
        OR
        PhD in Engineering, Information Systems, Computer Science, or related field and 1+ year of Software Engineering or related work experience.

        • 2+ years of academic or work experience with Programming Language such as C, C++, Java, Python, etc.
        Location - Hyderabad
        Experience - 1-5 Years
        We are seeking an experienced Machine Learning Engineers specializing in Generative AI to join our core AI team. The ideal candidate will be responsible for designing, developing, and deploying cutting-edge generative AI solutions, with a focus on Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Intelligent agent systems.
        """

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
        return text

    def enhance_section(self, section_text, section_name):
        """Enhance a specific section of the resume"""
        prompt = f"""Enhance the following {section_name} section of a resume to better match the job description.
        Focus on highlighting relevant skills and experiences. Keep the format simple and professional.
        
        JOB DESCRIPTION:
        {self.job_description}
        
        ORIGINAL {section_name.upper()}:
        {section_text}
        
        ENHANCED {section_name.upper()}:"""
        
        try:
            response = self.generator(
                prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=False
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            print(f"Error enhancing {section_name} section: {str(e)}")
            return section_text

    def create_enhanced_pdf(self, input_pdf, output_pdf):
        """Create an enhanced version of the PDF resume"""
        # Extract text from PDF
        resume_text = self.extract_text_from_pdf(input_pdf)
        
        # Split into sections (simple approach)
        sections = {}
        current_section = "header"
        sections[current_section] = []
        
        for line in resume_text.split('\n'):
            line = line.strip()
            if line.upper() in ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS']:
                current_section = line.upper()
                sections[current_section] = []
            if current_section in sections:
                sections[current_section].append(line)
        
        # Enhance each section
        enhanced_sections = {}
        for section, content in sections.items():
            if section == 'header':
                enhanced_sections[section] = content
                continue
                
            section_text = '\n'.join(content)
            enhanced_section = self.enhance_section(section_text, section)
            enhanced_sections[section] = enhanced_section
        
        # Create PDF using ReportLab
        self._create_pdf_reportlab(enhanced_sections, output_pdf)
    
    def _create_pdf_reportlab(self, sections, output_path):
        """Create PDF using ReportLab with better Unicode support"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Create custom styles if they don't exist
        if not hasattr(styles, 'CustomSectionHeader'):
            styles.add(ParagraphStyle(
                name='CustomSectionHeader',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=24,
                textColor=colors.HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            ))
        
        if not hasattr(styles, 'CustomBodyText'):
            styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                spaceAfter=6,
                fontName='Helvetica'
            ))
        
        # Build the document
        elements = []
        
        # Add header
        if 'header' in sections:
            header_text = '<br/>'.join(sections['header'])
            elements.append(Paragraph(header_text, styles['CustomBodyText']))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Add sections
        for section, content in sections.items():
            if section == 'header':
                continue
                
            # Add section header
            elements.append(Paragraph(section, styles['CustomSectionHeader']))
            
            # Add section content
            if isinstance(content, list):
                content = '\n'.join(content)
            
            # Clean up content
            content = content.replace('•', '• ')
            content = content.replace('  ', ' ')
            
            # Split into paragraphs and add to elements
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    elements.append(Paragraph(para, styles['CustomBodyText']))
                    elements.append(Spacer(1, 0.1 * inch))
            
            elements.append(Spacer(1, 0.2 * inch))
        
        # Build the PDF
        doc.build(elements)
        print(f"✅ Success! Enhanced resume saved as {output_path}")

if __name__ == "__main__":
    enhancer = PDFResumeEnhancer()
    
    # Paths
    input_pdf = "resumes/GaneshResume.pdf"
    output_pdf = "enhanced_resume_reportlab.pdf"
    
    if not os.path.exists(input_pdf):
        print(f"Error: Input PDF file '{input_pdf}' not found.")
        print("Please update the 'input_pdf' variable with your PDF file path.")
    else:
        print("Enhancing your PDF resume with ReportLab...")
        enhancer.create_enhanced_pdf(input_pdf, output_pdf)
