from Resume_generator import ResumeGenerator
import os

def main():
    # Initialize the resume generator with the sample resume
    sample_resume_path = os.path.join("resumes", "GaneshResume.pdf")
    generator = ResumeGenerator(resume_path=sample_resume_path)
    
    # Sample job description
    job_description = """
    As a leading technology innovator, Qualcomm pushes the boundaries of what's possible to enable next-generation experiences and drives digital transformation to help create a smarter, connected future for all. As a Qualcomm Software Engineer, you will design, develop, create, modify, and validate embedded and cloud edge software, applications, and/or specialized utility programs that launch cutting-edge, world class products that meet and exceed customer needs. Qualcomm Software Engineers collaborate with systems, hardware, architecture, test engineers, and other teams to design system-level software solutions and obtain information on performance requirements and interfaces.

Minimum Qualifications:

• Bachelor's degree in Engineering, Information Systems, Computer Science, or related field and 3+ years of Software Engineering or related work experience.
OR
Master's degree in Engineering, Information Systems, Computer Science, or related field and 2+ years of Software Engineering or related work experience.
OR
PhD in Engineering, Information Systems, Computer Science, or related field and 1+ year of Software Engineering or related work experience.

• 2+ years of academic or work experience with Programming Language such as C, C++, Java, Python, etc.
Location - Hyderabad
Experience - 1-5 Years
We are seeking an experienced Machine Learning Engineers specializing in Generative AI to join our core AI team.

The ideal candidate will be responsible for designing, developing, and deploying cutting-edge generative AI solutions, with a focus on Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Intelligent agent systems.

Key Responsibilities:

Design and implement RAG-based solutions to enhance LLM capabilities with external knowledge sources

Develop and optimize LLM fine-tuning strategies for specific use cases and domain adaptation

Create robust evaluation frameworks for measuring and improving model performance

Build and maintain agentic workflows for autonomous AI systems

Collaborate with cross-functional teams to identify opportunities and implement AI solutions

Required Qualifications:

Bachelor's or Master's degree in Computer Science, or related technical field

3+ years of experience in Machine Learning/AI engineering

Strong programming skills in Python and experience with ML frameworks (PyTorch, TensorFlow)

Practical experience with LLM deployments and fine-tuning

Experience with vector databases and embedding models

Familiarity with modern AI/ML infrastructure and cloud platforms (AWS, GCP, Azure)

Strong understanding of RAG architectures and implementation

Preferred Qualifications:

Experience with popular LLM frameworks (Langchain, LlamaIndex, Transformers)

Knowledge of prompt engineering and chain-of-thought techniques

Experience with containerization and microservices architecture

Background in NLP and deep learning

Background in Reinforcement Learning

Contributions to open-source AI projects

Experience with ML ops and model deployment pipelines

Skills and Competencies:

Strong problem-solving and analytical skills

Excellent communication and collaboration abilities

Experience with agile development methodologies

Ability to balance multiple projects and priorities

Strong focus on code quality and best practices

Understanding of AI ethics and responsible AI development
    """
    
    # Generate the tailored resume
    print("Generating tailored resume...")
    tailored_resume = generator.generate_resume(job_description)
    
    # Save the generated resume
    output_file = "generated_resume.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tailored_resume)
    
    print(f"\nGenerated resume has been saved to: {output_file}")
    
    # Print a preview of the generated resume
    print("\n=== PREVIEW OF GENERATED RESUME ===")
    print(tailored_resume[:500] + "..." if len(tailored_resume) > 500 else tailored_resume)

if __name__ == "__main__":
    main()