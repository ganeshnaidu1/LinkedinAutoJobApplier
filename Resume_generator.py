import os
import re
import json
import requests
from dotenv import load_dotenv
import PyPDF2
from huggingface_hub.inference._client import InferenceClient
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch
from typing import Optional, Dict, Any

load_dotenv()

class ResumeGenerator:
    def __init__(self, resume_path: Optional[str] = None, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):  # Using a model that's known to work
        self.model_initialized = False
        self.resume = resume_path or "resumes/GaneshResume.pdf"
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Get API key from environment
        self.api_key = os.getenv('HUGGINGFACEHUB_API_KEY')
        if not self.api_key:
            print("Error: HUGGINGFACEHUB_API_KEY not found in environment variables")
            print("Please make sure your .env file contains HUGGINGFACEHUB_API_KEY=your_api_key")
            return
            
        # Initialize the Hugging Face Inference client
        try:
            self.client = InferenceClient(token=self.api_key, model=model_name)
            print(f"Successfully initialized model: {model_name}")
            self.model_initialized = True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            self.model_initialized = False
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF file: {str(e)}")
            return ""

    def _clean_resume_text(self, text: str) -> str:
        """Clean and normalize the resume text"""
        # Remove extra whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def _create_prompt(self, job_description: str, resume_content: str) -> str:
        """Create a prompt for the LLM to tailor the resume"""
        return f"""
        Please tailor the following resume for the job description below. 
        Focus on highlighting relevant skills and experiences.
        
        JOB DESCRIPTION:
        {job_description}
        
        CURRENT RESUME:
        {resume_content}
        
        TAILORED RESUME:
        """

    def generate_text(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text using the Hugging Face Inference API"""
        if not self.model_initialized:
            print("Error: Model not initialized")
            return ""
            
        try:
            response = self.client.text_generation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_full_text=False
            )
            return response
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""
    
    def generate_resume(self, job_description: str, output_path: Optional[str] = None) -> str:
        """Generate a tailored resume based on the job description"""
        if not self.model_initialized:
            print("Error: Model not initialized")
            return ""
            
        print("\nGenerating tailored resume...")
        
        try:
            # Extract text from the resume
            resume_content = self._extract_text_from_pdf(self.resume)
            if not resume_content:
                return "Error: Could not extract text from resume"
                
            # Clean the resume text
            resume_content = self._clean_resume_text(resume_content)
            
            # Create the prompt
            prompt = self._create_prompt(job_description, resume_content)
            
            # Generate the tailored resume
            tailored_resume = self.generate_text(prompt)
            
            # Save to file if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(tailored_resume)
                print(f"\n✅ Tailored resume saved to: {os.path.abspath(output_path)}")
            
            return tailored_resume
            
        except Exception as e:
            print(f"Error generating resume: {str(e)}")
            return ""
    
    def generate_resume_pdf(self, job_description: str, output_pdf: str) -> bool:
        """Generate a PDF of the tailored resume"""
        try:
            # Generate the text content first
            resume_text = self.generate_resume(job_description)
            if not resume_text:
                return False
            
            # Create a PDF document
            doc = SimpleDocTemplate(
                output_pdf,
                pagesize=letter,
                rightMargin=72, leftMargin=72,
                topMargin=72, bottomMargin=18
            )
            
            # Create styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='Normal_Center',
                parent=styles['Normal'],
                alignment=TA_CENTER,
            ))
            
            # Prepare the content
            content = []
            
            # Split the text into paragraphs and add to content
            for para in resume_text.split('\n\n'):
                if para.strip():
                    content.append(Paragraph(para.strip(), styles['Normal']))
                    content.append(Spacer(1, 12))
            
            # Build the document
            doc.build(content)
            print(f"\n✅ PDF resume saved to: {os.path.abspath(output_pdf)}")
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return False
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
        """
        Generate a tailored resume based on the job description using Hugging Face API
        
        Args:
            job_description: The job description to tailor the resume for
            output_pdf: Optional path to save the generated PDF
            
        Returns:
            str: The generated resume text
        """
        if not self.model_initialized:
            print("Error: Connection to Hugging Face API not initialized.")
            return None
            
        try:
            # Read the resume content
            resume_content = self._extract_text_from_pdf(self.resume)
            if not resume_content:
                print("Error: Could not read resume content.")
                return None
                
            # Clean the resume content
            resume_content = self._clean_resume_text(resume_content)
            
            # Create the prompt
            prompt = self._create_prompt(job_description, resume_content)
            
            try:
                # Generate the tailored resume using the API
                print("Generating tailored resume using Qwen 2.5 7B model via Hugging Face API...")
                
                # Use the generate_text method which handles the API call
                generated_text = self.generate_text(prompt, max_new_tokens=1024, temperature=0.7)
                
                if not generated_text:
                    raise ValueError("Failed to generate content from the API")
                
                # Clean and format the output
                generated_text = self._clean_generated_text(generated_text, prompt)
                
                # Ensure we have a minimum amount of text
                if len(generated_text) < 50:
                    raise ValueError("Generated response is too short")
                    
                # If output_pdf is provided, save as PDF
                if output_pdf:
                    try:
                        self._save_to_pdf(generated_text, output_pdf)
                        print(f"Successfully saved PDF to: {os.path.abspath(output_pdf)}")
                    except Exception as e:
                        print(f"Warning: Could not save as PDF: {str(e)}")
                
                print("Successfully generated tailored resume")
                return generated_text
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                raise
            
        except Exception as e:
            error_msg = f"Error generating resume: {str(e)}"
            print(error_msg)
            
            # Return a more helpful error message with the original content
            return f"""# Error Generating Resume

We encountered an issue generating a tailored resume. Here's what you can do:
1. Check if your resume file is not empty and properly formatted
2. Try again with a shorter job description
3. Make sure you have a stable internet connection
4. Verify your Hugging Face API token is valid and has access to the model
5. Check if the model is available at {self.api_url}

Original resume content:

{resume_content[:1000] if 'resume_content' in locals() else 'Could not load resume content'}{'...' if 'resume_content' in locals() and len(resume_content) > 1000 else ''}"""


def main():
    """Main function to demonstrate resume generation with Hugging Face API"""
    try:
        # Check if Hugging Face API token is set
        if not os.getenv('HUGGINGFACEHUB_API_KEY'):
            print("Error: HUGGINGFACEHUB_API_KEY environment variable is not set in .env file.")
            print("Please set it with your Hugging Face API token in the .env file.")
            print("You can get your API token from: https://huggingface.co/settings/tokens")
            return 1
            
        # Initialize the resume generator
        print("Initializing resume generator...")
        generator = ResumeGenerator()
        
        if not generator.model_initialized:
            print("Failed to initialize the model. Please check your API key and internet connection.")
            return 1
            
        # Sample job description (you can replace this with a real one)
        job_description = """
        We are looking for a skilled software engineer with experience in Python, 
        machine learning, and web development. The ideal candidate should have 
        experience with cloud platforms and modern software development practices.
        """
        
        # Output files
        output_txt = 'tailored_resume.txt'
        output_pdf = 'tailored_resume.pdf'
        
        print("\n" + "="*50)
        print("RESUME TAILORING TOOL")
        print("="*50)
        print(f"Resume: {os.path.abspath(generator.resume)}")
        print(f"Output will be saved to: {os.path.abspath(output_txt)}")
        
        # Generate the resume
        print("\nGenerating tailored resume...")
        generated_resume = generator.generate_resume(job_description, output_pdf=output_pdf)
        
        if not generated_resume or generated_resume.startswith("Error"):
            print("\n❌ Failed to generate resume. Please check the error message above.")
            return 1
        
        # Save the generated resume as a text file
        try:
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(generated_resume)
            print(f"\n✅ Success! Tailored resume has been saved to:")
            print(f"Text file: {os.path.abspath(output_txt)}")
            if os.path.exists(output_pdf):
                print(f"PDF file:  {os.path.abspath(output_pdf)}")
            
            print("\nYou can now open these files to view your tailored resume!")
            
        except Exception as e:
            print(f"\n⚠️  Generated resume but could not save to file: {str(e)}")
            print("\nHere's the generated resume content:")
            print("-"*50)
            print(generated_resume)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


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

    def generate_resume(self, job_description: str, output_pdf: str = None) -> str:
        """
        Generate a tailored resume based on the job description using Hugging Face API
        
        Args:
            job_description: The job description to tailor the resume for
            output_pdf: Optional path to save the generated PDF
            
        Returns:
            str: The generated resume text
        """
        if not self.model_initialized:
            error_msg = "Error: Connection to Hugging Face API not initialized."
            print(error_msg)
            return error_msg
            
        try:
            # Check if resume file exists
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
            
            try:
                # Generate the tailored resume using the API
                print("5. Generating tailored resume using Qwen 2.5 7B model...")
                
                # Use the generate_text method which handles the API call
                generated_text = self.generate_text(prompt, max_new_tokens=1024, temperature=0.7)
                
                if not generated_text:
                    raise ValueError("Failed to generate content from the API")
                
                # Clean and format the output
                generated_text = self._clean_generated_text(generated_text, prompt)
                
                # Ensure we have a minimum amount of text
                if len(generated_text) < 50:
                    raise ValueError("Generated response is too short")
                    
                # If output_pdf is provided, save as PDF
                if output_pdf:
                    try:
                        self._save_to_pdf(generated_text, output_pdf)
                        print(f"6. Successfully saved PDF to: {os.path.abspath(output_pdf)}")
                    except Exception as e:
                        print(f"Warning: Could not save as PDF: {str(e)}")
                
                print("\n✅ Successfully generated tailored resume!")
                return generated_text
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                raise
            
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
        