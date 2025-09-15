import os
import re
import PyPDF2
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch

load_dotenv()

class ResumeGeneratorGroq:
    def __init__(self, resume_path: Optional[str] = None, 
                 model_name: str = "gemma2-9b-it"):  # Using Gemma 2 9B model
        """Initialize the ResumeGenerator with a resume path and model name."""
        self.model_initialized = False
        self.resume = resume_path or "resumes/GaneshResume.pdf"
        self.model_name = model_name
        
        # Initialize Groq client
        load_dotenv()  # Make sure .env is loaded
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment variables")
            print("Please add your Groq API key to the .env file")
            return
            
        try:
            self.client = Groq(api_key=api_key)
            self.model_initialized = True
            print(f"Successfully initialized Groq client with model: {model_name}")
        except Exception as e:
            print(f"Error initializing Groq client: {str(e)}")
            self.model_initialized = False
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
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

    def _clean_text(self, text: str) -> str:
        """Clean and normalize the text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
        
    def _create_prompt(self, job_description: str, resume_content: str) -> str:
        """Create a detailed prompt for generating a well-structured resume."""
        return f"""
        Please create a professional, well-structured resume tailored to the job description below. 
        Focus on highlighting relevant skills, experiences, and projects that match the job requirements.
        
        INSTRUCTIONS:
        1. Format the resume with clear, distinct sections: Summary, Technical Skills, Work Experience, Projects, and Education.
        2. For the Skills section, include both technical and soft skills relevant to the job.
        3. For Work Experience and Projects, include 2-3 bullet points per role/project, focusing on achievements and impact.
        4. Include relevant technologies, frameworks, and tools used in each role/project.
        5. Add hyperlinks to your LinkedIn, GitHub, and any relevant project repositories.
        6. Keep the language professional and achievement-oriented.
        7. Ensure proper spacing and formatting for readability.
        
        JOB DESCRIPTION:
        {job_description}
        
        CURRENT RESUME CONTENT (use as reference but tailor for the job):
        {resume_content}
        
        Generate the tailored resume below. Use markdown formatting for section headers (##), bullet points (-), and links.
        Include all relevant sections even if not in the original resume.
        
        FORMAT EXAMPLE:
        ## [Your Name]
        [Email] | [Phone] | [LinkedIn] | [GitHub] | [Portfolio]
        
        ### Summary
        [2-3 sentences highlighting your experience and what you bring to the role]
        
        ### Technical Skills
        - **Category 1:** Skill 1, Skill 2, Skill 3
        - **Category 2:** Skill 4, Skill 5
        
        ### Work Experience
        **Job Title**  
        *Company Name, Location | Month Year - Present*  
        - Achievement 1 with impact and technologies used
        - Achievement 2 with metrics if possible
        
        ### Projects
        **Project Name** | [GitHub](link) | [Live Demo](link)  
        *Technologies: Python, Flask, React, etc.*  
        - Description of project and your contributions
        - Key features and impact
        
        ### Education
        **Degree in Field**  
        *University Name, Location | Graduation Year*  
        - Relevant coursework, achievements, or activities
        
        NOW GENERATE THE TAILORED RESUME:
        """

    def generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text using the Groq API."""
        if not self.model_initialized:
            print("Error: Groq client not initialized")
            return ""
            
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return ""
    
    def generate_resume(self, job_description: str, output_path: Optional[str] = None) -> str:
        """Generate a tailored resume based on the job description."""
        if not self.model_initialized:
            print("Error: Groq client not initialized")
            return ""
            
        print("\nExtracting and processing resume content...")
        
        try:
            # Extract and clean resume text
            resume_content = self._extract_text_from_pdf(self.resume)
            if not resume_content:
                return "Error: Could not extract text from resume"
                
            resume_content = self._clean_text(resume_content)
            
            # Create the prompt
            prompt = self._create_prompt(job_description, resume_content)
            
            print("Generating tailored resume (this may take a moment)...")
            tailored_resume = self.generate_text(prompt)
            
            # Save to file if output path is provided
            if output_path and tailored_resume:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(tailored_resume)
                print(f"\n✅ Tailored resume saved to: {os.path.abspath(output_path)}")
            
            return tailored_resume
            
        except Exception as e:
            print(f"Error generating resume: {str(e)}")
            return ""
    
    def generate_resume_pdf(self, job_description: str, output_pdf: str) -> bool:
        """Generate a PDF of the tailored resume."""
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
                    # Check if this is a section header
                    if para.strip().endswith(':'):
                        content.append(Spacer(1, 12))
                        content.append(Paragraph(para.strip(), styles['Heading2']))
                    else:
                        content.append(Paragraph(para.strip(), styles['Normal']))
                    content.append(Spacer(1, 6))
            
            # Build the document
            doc.build(content)
            print(f"\n✅ PDF resume saved to: {os.path.abspath(output_pdf)}")
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return False

def main():
    """Main function to demonstrate resume generation."""
    print("Initializing Groq Resume Generator...")
    
    # Create a sample job description
    job_description = """
    We are looking for a Python Developer with experience in:
    - Web development using Flask or Django
    - Data analysis with pandas and NumPy
    - Working with RESTful APIs
    - Version control with Git
    
    Key responsibilities:
    - Develop and maintain web applications
    - Write clean, maintainable code
    - Collaborate with cross-functional teams
    - Debug and fix issues
    """
    
    # Initialize the generator with Gemma 2 9B model
    generator = ResumeGeneratorGroq(model_name="gemma2-9b-it")
    
    if not generator.model_initialized:
        print("Failed to initialize the Groq client. Please check the error messages above.")
        return 1
    
    # Generate text resume
    output_txt = "tailored_resume_groq.txt"
    print(f"\nGenerating tailored resume for job description...")
    resume_text = generator.generate_resume(job_description, output_path=output_txt)
    
    if resume_text:
        print("\nGenerated Resume Preview:")
        print("-" * 50)
        print(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        print("-" * 50)
        
        # Generate PDF version
        output_pdf = "tailored_resume_groq.pdf"
        if generator.generate_resume_pdf(job_description, output_pdf):
            print(f"\n✅ Successfully generated PDF resume: {os.path.abspath(output_pdf)}")
    else:
        print("\n❌ Failed to generate resume. Please check the error messages above.")
    
    return 0

if __name__ == "__main__":
    exit(main())
