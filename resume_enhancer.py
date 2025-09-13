import os
from fpdf import FPDF
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

class ResumeEnhancer:
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
                model="google/flan-t5-small"  # Smaller model as fallback
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

        Key Responsibilities:
        • Design and implement RAG-based solutions to enhance LLM capabilities with external knowledge sources
        • Develop and optimize LLM fine-tuning strategies for domain adaptation
        • Create robust evaluation frameworks for measuring and improving model performance
        • Build and maintain agentic workflows for autonomous AI systems
        • Collaborate with cross-functional teams to implement AI solutions

        Required Skills:
        • Strong programming skills in Python, Java, or C++
        • Experience with machine learning frameworks (TensorFlow, PyTorch)
        • Knowledge of NLP and deep learning techniques
        • Experience with vector databases and embedding models
        • Familiarity with modern AI/ML infrastructure and cloud platforms (AWS, GCP, Azure)
        • Strong understanding of RAG architectures
        """

    def read_resume(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def enhance_resume(self, resume_text):
        # Split the resume into sections
        sections = {}
        current_section = "header"
        sections[current_section] = []
        
        # Simple section detection
        for line in resume_text.split('\n'):
            line = line.strip()
            if line.upper() in ['TECHNICAL SKILLS', 'EXPERIENCE', 'EDUCATION', 'PROJECTS', 'CERTIFICATIONS']:
                current_section = line.upper()
                sections[current_section] = []
            if current_section in sections:
                sections[current_section].append(line)
            else:
                sections[current_section].append(line)
        
        # Process each section with the model
        enhanced_sections = {}
        for section, content in sections.items():
            if section == 'header':
                enhanced_sections[section] = content
                continue
                
            section_text = '\n'.join(content)
            prompt = f"""Enhance the following resume section to better match the job description.
            Keep the original format and structure but update the content to highlight relevant experience.
            Preserve all links and technical terms.
            
            JOB DESCRIPTION:
            {self.job_description}
            
            SECTION: {section}
            {section_text}
            
            ENHANCED {section}:"""
            
            try:
                response = self.generator(
                    prompt,
                    max_length=1000,
                    num_return_sequences=1,
                    temperature=0.2,
                    do_sample=False
                )
                enhanced_sections[section] = response[0]['generated_text'].strip()
            except Exception as e:
                print(f"Error enhancing {section} section: {str(e)}")
                enhanced_sections[section] = section_text
        
        # Reconstruct the resume
        enhanced_resume = []
        for section in ['header'] + [s for s in enhanced_sections.keys() if s != 'header']:
            if section in enhanced_sections:
                if isinstance(enhanced_sections[section], list):
                    enhanced_resume.append('\n'.join(enhanced_sections[section]))
                else:
                    enhanced_resume.append(enhanced_sections[section])
        
        return '\n\n'.join(enhanced_resume)

    def markdown_to_pdf(self, markdown_text, output_path):
        # Convert markdown to HTML
        html = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
        
        # Create PDF with Unicode support
        pdf = FPDF()
        pdf.add_page()
        
        # Use built-in font with UTF-8 support
        pdf.set_font('Arial', size=11)
        
        # Set margins
        pdf.set_left_margin(20)
        pdf.set_right_margin(20)
        
        # Convert HTML to plain text for PDF
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator='\n')
        
        # Add text to PDF with proper encoding and wrapping
        for line in text.split('\n'):
            line = line.strip()
            if line:
                try:
                    # Replace problematic characters
                    line = line.replace('•', '-')
                    line = line.encode('latin-1', 'replace').decode('latin-1')
                    
                    # Split long lines to prevent overflow
                    if pdf.get_string_width(line) > pdf.w - 40:  # Account for margins
                        words = line.split()
                        current_line = ''
                        for word in words:
                            if pdf.get_string_width(current_line + word + ' ') < pdf.w - 40:
                                current_line += word + ' '
                            else:
                                if current_line:
                                    pdf.cell(0, 8, txt=current_line, ln=1, align='L')
                                current_line = word + ' '
                        if current_line:
                            pdf.cell(0, 8, txt=current_line, ln=1, align='L')
                    else:
                        pdf.cell(0, 8, txt=line, ln=1, align='L')
                except Exception as e:
                    print(f"Warning: Could not add line: {str(e)}")
                    continue
        
        # Save PDF
        try:
            pdf.output(output_path)
        except Exception as e:
            print(f"Error saving PDF: {str(e)}")
            # Fallback: Save as text file
            with open('enhanced_resume.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            print("Saved enhanced resume as text file: enhanced_resume.txt")

if __name__ == "__main__":
    enhancer = ResumeEnhancer()
    
    # Paths
    input_resume = "resumes/GaneshResume.pdf"
    output_pdf = "enhanced_resume.pdf"
    
    try:
        # Read the resume
        resume_text = enhancer.read_resume(input_resume)
        
        # Enhance the resume
        print("Enhancing your resume for the Machine Learning Engineer position...")
        enhanced_resume = enhancer.enhance_resume(resume_text)
        
        # Save enhanced resume as markdown
        with open("enhanced_resume.md", "w", encoding="utf-8") as f:
            f.write(enhanced_resume)
        
        # Convert to PDF
        print("Converting to PDF...")
        enhancer.markdown_to_pdf(enhanced_resume, output_pdf)
        
        print(f"\n✅ Success! Enhanced resume saved as {output_pdf}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
