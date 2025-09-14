import os
import re
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from dotenv import load_dotenv
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch
from typing import Optional

# Set random seeds for reproducibility
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

load_dotenv()
set_all_seeds(42)  # For reproducibility

class ResumeGenerator:
    def __init__(self, resume_path: Optional[str] = None, model_path: str = "Qwen/Qwen1.5-1.8B"):
        self.model_initialized = False
        self.resume = resume_path or "resumes/GaneshResume.pdf"
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Qwen 2B model with error handling"""
        try:
            print("Initializing Qwen 2B model...")
            
            # Check if we have CUDA available
            if torch.cuda.is_available():
                print("CUDA is available. Using GPU acceleration.")
                self.device = "cuda"
            else:
                print("CUDA not available. Using CPU (this will be slower).")
                self.device = "cpu"
            
            # Initialize tokenizer and model
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"device_map": "auto"}
            )
            
            self.model_initialized = True
            print("Qwen 2B model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            print("Make sure you have enough GPU memory (at least 12GB recommended)")
            print("or try running on CPU with more system RAM.")
            self.model_initialized = False
        
    def _create_prompt(self, job_desc, resume_content):
        """Create a prompt for the model to generate a tailored resume"""
        # Limit the resume content length to avoid hitting model's token limit
        max_resume_length = 100000  # Increased to preserve more context
        if len(resume_content) > max_resume_length:
            # Try to keep sections intact when truncating
            truncated = resume_content[:max_resume_length]
            last_section = truncated.rfind('\n\n')
            if last_section > 0:
                truncated = truncated[:last_section] + "\n... [sections truncated for length]"
            resume_content = truncated
            
        return f"""
        TASK: I need you to help me tailor my resume for a specific job application. 
        I want to maintain the EXACT same structure, formatting, and content as my original resume, 
        but make targeted improvements to better match the job description.
        
        JOB DESCRIPTION:
        {job_desc}
        
        MY ORIGINAL RESUME (PRESERVE THIS EXACT FORMATTING AND STRUCTURE):
        {resume_content}
        
        INSTRUCTIONS:
        1. PRESERVE ALL original content, formatting, section order, and styling exactly as is
        2. DO NOT change any contact information, dates, job titles, company names, or education details
        3. For the SKILLS section:
           - Keep all existing skills
           - Only add new skills if they are explicitly mentioned in the job description and you're confident I have them
           - Maintain the same categorization and formatting
        
        4. For WORK EXPERIENCE and PROJECTS:
           - Keep all original bullet points exactly as written
           - Only modify/expand bullet points that directly relate to the job requirements
           - When enhancing bullet points, maintain the same style and level of detail
           - Add 1-2 new bullet points per role if they strongly match the job requirements
        
        5. DO NOT make up any information or experience I don't have
        6. Keep the same professional tone and writing style throughout
        7. If the job description mentions specific technologies or requirements that I don't have in my resume, 
           highlight relevant transferable skills instead of adding new ones
        
        Return the ENTIRE resume with minimal changes, focusing only on the most relevant improvements.
        Maintain all original section headers, formatting, and structure.
        
        TAILORED RESUME (same format as original, with minimal targeted changes):
        """
    
    def _clean_generated_text(self, text, prompt):
        """Clean and format the generated text"""
        # Remove any text that might be part of the prompt
        text = text.replace(prompt, '').strip()
        
        # Remove any markdown code blocks if present
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove any leading/trailing whitespace and newlines
        text = text.strip('\n').strip()
        
        return text
        
    def _create_styles(self):
        """Create and return paragraph styles matching the original resume's format"""
        styles = getSampleStyleSheet()
        
        # Base styles that match the original resume
        base_style = {
            'fontName': 'Helvetica',
            'fontSize': 10,
            'leading': 12,
            'spaceAfter': 6,
            'textColor': 'black'
        }
        
        # Custom styles - only add if they don't already exist
        custom_styles = {
            'Name': {
                'fontName': 'Helvetica-Bold',
                'fontSize': 16,
                'spaceAfter': 6,
                'textColor': 'black',
                'alignment': TA_LEFT
            },
            'ContactInfo': {
                'fontName': 'Helvetica',
                'fontSize': 10,
                'leading': 14,
                'spaceAfter': 12,
                'textColor': 'black',
                'alignment': TA_LEFT
            },
            'SectionHeader': {
                'fontName': 'Helvetica-Bold',
                'fontSize': 12,
                'leading': 14,
                'spaceBefore': 12,
                'spaceAfter': 6,
                'textColor': 'black',
                'alignment': TA_LEFT,
                'borderWidth': 1,
                'borderColor': '#333333',
                'borderPadding': (0, 0, 0, 2)
            },
            'JobTitle': {
                'fontName': 'Helvetica-Bold',
                'fontSize': 10,
                'leading': 12,
                'spaceAfter': 2,
                'textColor': 'black',
                'alignment': TA_LEFT
            },
            'Company': {
                'fontName': 'Helvetica-Bold',
                'fontSize': 10,
                'leading': 12,
                'spaceAfter': 2,
                'textColor': 'black',
                'alignment': TA_LEFT
            },
            'Date': {
                'fontName': 'Helvetica-Oblique',
                'fontSize': 9,
                'leading': 12,
                'spaceAfter': 4,
                'textColor': 'black',
                'alignment': TA_RIGHT
            },
            'Bullet': {
                'fontName': 'Helvetica',
                'fontSize': 10,
                'leading': 12,
                'leftIndent': 10,
                'firstLineIndent': -10,
                'spaceBefore': 0,
                'spaceAfter': 4,
                'textColor': 'black',
                'bulletIndent': 0,
                'bulletFontSize': 10
            },
            'Skills': {
                'fontName': 'Helvetica',
                'fontSize': 10,
                'leading': 12,
                'spaceAfter': 4,
                'textColor': 'black',
                'alignment': TA_LEFT
            }
        }
        
        # Add styles only if they don't exist
        for name, style_params in custom_styles.items():
            if name not in styles:
                styles.add(ParagraphStyle(name=name, **style_params))
        
        return styles
        
    def _save_to_pdf(self, text, output_path):
        """Save the generated resume to a PDF file using ReportLab"""
        print(f"Starting PDF generation for: {output_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Output path (absolute): {os.path.abspath(output_path)}")
        
        try:
            # Create a PDF document with appropriate margins
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=40,  # Reduced from 72
                leftMargin=40,   # Reduced from 72
                topMargin=40,    # Reduced from 72
                bottomMargin=40  # Reduced from 72
            )
            
            # Create styles
            styles = self._create_styles()
            
            # Create story (content)
            story = []
            
            # Process each line of the resume text
            current_section = None
            current_list = []
            
            # Split text into sections
            sections = text.split('\n\n')
            
            for section in sections:
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                if not lines:
                    continue
                    
                # Check if this is the name section (first non-empty line)
                if not story and len(lines) == 1 and '|' not in lines[0]:
                    story.append(Paragraph(lines[0], styles['Name']))
                    continue
                    
                # Check if this is the contact info line (contains | separators)
                if any('|' in line for line in lines):
                    contact_info = ' '.join(lines)
                    story.append(Paragraph(contact_info, styles['ContactInfo']))
                    continue
                    
                # Handle section headers
                if len(lines) == 1 and (lines[0].isupper() or lines[0].endswith(':')):
                    section_title = lines[0].strip(' :')
                    story.append(Paragraph(section_title, styles['SectionHeader']))
                    continue
                    
                # Handle experience/project entries
                if len(lines) >= 2 and ('–' in lines[0] or 'present' in lines[0].lower()):
                    # This is a job/project title line
                    title_line = lines[0]
                    date_parts = [part.strip() for part in title_line.split('  ') if part.strip()]
                    
                    if len(date_parts) >= 2:
                        # Position and company are in the first part
                        position_company = date_parts[0]
                        date_range = date_parts[-1]
                        
                        # Add position and company
                        if ' at ' in position_company:
                            position, company = position_company.split(' at ', 1)
                            story.append(Paragraph(company.strip(), styles['Company']))
                            story.append(Paragraph(position.strip(), styles['JobTitle']))
                        else:
                            story.append(Paragraph(position_company, styles['JobTitle']))
                        
                        # Add date range
                        story.append(Paragraph(date_range, styles['Date']))
                    else:
                        story.append(Paragraph(title_line, styles['JobTitle']))
                    
                    # Process bullet points
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith(('•', '-', '*')):
                            bullet_text = line[1:].strip()
                            current_list.append(Paragraph(bullet_text, styles['Bullet']))
                        else:
                            if current_list:
                                story.append(ListFlowable(
                                    current_list,
                                    bulletType='bullet',
                                    bulletFontName='Helvetica',
                                    bulletFontSize=10,
                                    leftIndent=20,
                                    rightIndent=0,
                                    bulletIndent=0,
                                    spaceAfter=6
                                ))
                                current_list = []
                            if line:  # Only add non-empty lines
                                story.append(Paragraph(line, styles['Normal']))
                    
                    # Add any remaining bullet points
                    if current_list:
                        story.append(ListFlowable(
                            current_list,
                            bulletType='bullet',
                            bulletFontName='Helvetica',
                            bulletFontSize=10,
                            leftIndent=20,
                            rightIndent=0,
                            bulletIndent=0,
                            spaceAfter=6
                        ))
                        current_list = []
                    
                    # Add some space after each experience/project
                    story.append(Spacer(1, 12))
                
                # Handle skills section
                elif any(skill_indicator in section.lower() for skill_indicator in ['skills', 'languages', 'technologies']):
                    for line in lines:
                        if ':' in line:
                            # This is a skill category (e.g., "Languages:")
                            category, skills = line.split(':', 1)
                            story.append(Paragraph(f"<b>{category}:</b>{skills}", styles['Skills']))
                        else:
                            story.append(Paragraph(line, styles['Skills']))
                
                # Handle education and other sections
                else:
                    for line in lines:
                        if line.startswith(('•', '-', '*')):
                            bullet_text = line[1:].strip()
                            current_list.append(Paragraph(bullet_text, styles['Bullet']))
                        else:
                            if current_list:
                                story.append(ListFlowable(
                                    current_list,
                                    bulletType='bullet',
                                    bulletFontName='Helvetica',
                                    bulletFontSize=10,
                                    leftIndent=20,
                                    rightIndent=0,
                                    bulletIndent=0,
                                    spaceAfter=6
                                ))
                                current_list = []
                            if line:  # Only add non-empty lines
                                story.append(Paragraph(line, styles['Normal']))
            
            # Add any remaining bullet points
            if current_list:
                story.append(ListFlowable(
                    current_list,
                    bulletType='bullet',
                    bulletFontName='Helvetica',
                    bulletFontSize=10,
                    leftIndent=20,
                    rightIndent=0,
                    bulletIndent=0,
                    spaceAfter=6
                ))
            
            print("Building PDF document...")
            # Build the PDF
            doc.build(story)
            
            # Verify the file was created
            abs_path = os.path.abspath(output_path)
            print(f"Checking if file exists at: {abs_path}")
            
            if os.path.exists(abs_path):
                size = os.path.getsize(abs_path)
                print(f"PDF successfully generated at: {abs_path} ({size} bytes)")
                return True
            else:
                print("Error: PDF was not created. Checking directory contents...")
                print(f"Files in directory: {os.listdir(os.path.dirname(abs_path) or '.')}")
                return False
                
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def generate_resume(self, job_description: str, output_pdf: str = None) -> str:
        # Read the resume content
        try:
            # Check if file is a PDF
            if self.resume.lower().endswith('.pdf'):
                import PyPDF2
                print(f"Reading PDF file: {self.resume}")
                with open(self.resume, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    resume_content = ""
                    for page in reader.pages:
                        resume_content += page.extract_text() + "\n"
                print("Successfully extracted text from PDF")
            else:
                # For text files, try with utf-8 first, fall back to other common encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                resume_content = None
                for encoding in encodings:
                    try:
                        with open(self.resume, 'r', encoding=encoding) as f:
                            resume_content = f.read()
                        print(f"Successfully read resume with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        print(f"Failed to read with {encoding} encoding, trying next...")
                        continue
                
                if resume_content is None:
                    # If all encodings fail, try with error handling
                    with open(self.resume, 'r', encoding='utf-8', errors='replace') as f:
                        resume_content = f.read()
                    print("Used error handling to read resume")
            
            if not resume_content.strip():
                raise ValueError("Resume file is empty")
                
        except FileNotFoundError:
            error_msg = f"Resume file not found at: {self.resume}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error reading resume: {str(e)}"
            print(error_msg)
            return error_msg
        
        if not self.model_initialized:
            return "Error: Model initialization failed. Please check your setup and try again."
            
        try:
            print("Generating tailored resume...")
            
            # Create a focused prompt with limited input
            prompt = self._create_prompt(job_description, resume_content)
            
            # Generate response with controlled parameters
            try:
                result = self.generator(
                    prompt,
                    max_length=500,
                    min_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    truncation=True
                )
                
                if not result or not result[0].get('generated_text'):
                    raise ValueError("Received empty response from the model")
                    
                generated_text = result[0]['generated_text']
                
                # Clean and format the output
                generated_text = self._clean_generated_text(generated_text, prompt)
                
                # Ensure we have a minimum amount of text
                if len(generated_text) < 50:
                    raise ValueError("Generated response is too short")
                    
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                raise
            
            print("Successfully generated tailored resume")
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating resume: {str(e)}"
            print(error_msg)
            
            # Return a more helpful error message with the original content
            return f"""# Error Generating Resume

We encountered an issue generating a tailored resume. Here's what you can do:
1. Check if your resume file is not empty and properly formatted
2. Try again with a shorter job description
3. Make sure you have a stable internet connection

Original resume content:

{resume_content[:1000]}{'...' if len(resume_content) > 1000 else ''}"""


def main():
    """Main function to demonstrate resume generation"""
    try:
        # Initialize the resume generator
        generator = ResumeGenerator()
        
        # Sample job description (you can replace this with a real one)
        job_description = """
        We are looking for a skilled software engineer with experience in Python, 
        machine learning, and web development. The ideal candidate should have 
        experience with cloud platforms and modern software development practices.
        """
        
        # First, save the generated resume as a text file
        output_txt = 'generated_resume.txt'
        abs_output_txt = os.path.abspath(output_txt)
        print(f"Will save generated resume to: {abs_output_txt}")
        
        # Generate the resume
        print("Generating resume tailored to the job description...")
        generated_resume = generator.generate_resume(job_description)
        
        # Save the generated resume as a text file
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(generated_resume)
        
        print(f"\nResume has been tailored and saved to '{output_txt}'")
        
        # Then try to save as PDF if needed
        output_pdf = 'tailored_resume_output.pdf'
        abs_output_pdf = os.path.abspath(output_pdf)
        print(f"\nAttempting to save as PDF to: {abs_output_pdf}")
        
        try:
            generator._save_to_pdf(generated_resume, output_pdf)
            print(f"PDF successfully saved to: {abs_output_pdf}")
        except Exception as e:
            print(f"Warning: Could not save as PDF: {str(e)}")
            print("The resume has been saved as a text file. You can view it at:")
            print(f"  {abs_output_txt}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


    def _extract_text_from_pdf(self, pdf_path):
        """Extract and format text content from a PDF file"""
        try:
            print(f"Extracting text from PDF: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text while preserving structure
                        page_text = ' '.join(line.strip() for line in page_text.split('\n') if line.strip())
                        text.append(page_text)
                        
                full_text = '\n\n'.join(text)
                print(f"Extracted {len(full_text)} characters from PDF")
                return full_text
                
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None

    def _clean_resume_text(self, text):
        """Clean and normalize the resume text while preserving structure"""
        if not text:
            return ""
            
        # Normalize whitespace but keep paragraph breaks
        lines = []
        current_paragraph = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line:  # If line has content
                current_paragraph.append(line)
            elif current_paragraph:  # Empty line with previous content
                lines.append(' '.join(current_paragraph))
                current_paragraph = []
                
        # Add the last paragraph if exists
        if current_paragraph:
            lines.append(' '.join(current_paragraph))
            
        # Join with double newlines to preserve paragraph structure
        return '\n\n'.join(lines)

    def generate_resume(self, job_description: str) -> str:
        """
        Generate a tailored resume based on the job description
        
        Args:
            job_description: The job description to tailor the resume for
            
        Returns:
            str: The generated resume text
        """
        # Read the resume content from PDF
        if not os.path.exists(self.resume):
            error_msg = f"Resume file not found at: {self.resume}"
            print(error_msg)
            return error_msg
            
        print(f"\n{'='*50}")
        print("RESUME TAILORING PROCESS")
        print(f"{'='*50}")
        print(f"Input resume: {os.path.abspath(self.resume)}")
        
        # Extract text from PDF
        print("\n1. Extracting text from resume...")
        resume_content = self._extract_text_from_pdf(self.resume)
        if not resume_content:
            return "Error: Could not extract text from the resume PDF"
            
        # Clean up the resume text
        print("2. Cleaning and formatting resume content...")
        resume_content = self._clean_resume_text(resume_content)
        
        if not resume_content.strip():
            return "Error: Extracted resume content is empty"
            
        print(f"3. Resume content prepared ({len(resume_content)} characters)")
        
        # Create the prompt for the model
        print("4. Creating optimization prompt...")
        prompt = self._create_prompt(job_description, resume_content)
        
        if not self.model_initialized:
            return "Error: Model initialization failed. Please check your setup and try again."
            
        try:
            print("Generating tailored resume...")
            
            # Create a focused prompt with limited input
            prompt = self._create_prompt(job_description, resume_content)
            
            # Generate response with controlled parameters
            try:
                result = self.generator(
                    prompt,
                    max_length=500,
                    min_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    truncation=True
                )
                
                if not result or not result[0].get('generated_text'):
                    raise ValueError("Received empty response from the model")
                    
                generated_text = result[0]['generated_text']
                
                # Clean and format the output
                generated_text = self._clean_generated_text(generated_text, prompt)
                
                # Ensure we have a minimum amount of text
                if len(generated_text) < 50:
                    raise ValueError("Generated response is too short")
                    
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                raise
            
            print("Successfully generated tailored resume")
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating resume: {str(e)}"
            print(error_msg)
            
            # Return a more helpful error message with the original content
            return f"""# Error Generating Resume

We encountered an issue generating a tailored resume. Here's what you can do:
1. Check if your resume file is not empty and properly formatted
2. Try again with a shorter job description
3. Make sure you have a stable internet connection

Original resume content:

{resume_content[:1000]}{'...' if len(resume_content) > 1000 else ''}"""
        