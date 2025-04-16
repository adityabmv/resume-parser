import streamlit as st
from google import genai
from pydantic import BaseModel
from typing import List
import os
import json

# Define Pydantic models
class ResumeData(BaseModel):
    name: str
    email: str
    phone: str
    skills: List[str]
    education: str
    experience: str

class JobData(BaseModel):
    title: str
    requiredSkills: List[str]
    requiredEducation: str
    requiredExperience: str

# Define nested model for skills_match
class SkillsMatch(BaseModel):
    matched: List[str]
    missing: List[str]
    percentage: float

# Define model for structured analysis response
class AnalysisData(BaseModel):
    skills_match: SkillsMatch
    education_fit: str
    experience_fit: str
    suitability_score: int
    summary: str

# Initialize Streamlit app
st.title("Resume-Job Match Analyzer")
st.markdown("Upload a resume PDF and enter a job description to get a brutally honest analysis of the candidate’s fit.")

# Set up Gemini client
client = genai.Client(api_key=st.secrets.get("GEMINI_API_KEY", None))  # Use Streamlit secrets or replace with your key

# # Sidebar for API key input (optional, for local testing)
# with st.sidebar:
#     api_key = st.text_input("Enter Gemini API Key (optional if set in secrets)", type="password")
#     if api_key:
#         client = genai.Client(api_key=api_key)

# File uploader for resume PDF
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# Text area for job description
job_description = st.text_area("Enter Job Description", placeholder="e.g., Senior Software Engineer requiring Python, React, AWS, Bachelor's in CS, 3+ years experience")

# Session state for storing JSONs, messages, and analysis
if "resume_json" not in st.session_state:
    st.session_state.resume_json = None
if "job_json" not in st.session_state:
    st.session_state.job_json = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# Function to process resume PDF
def process_resume(file):
    try:
        # Save uploaded file temporarily
        with open("temp_resume.pdf", "wb") as f:
            f.write(file.read())
        
        # Upload to Gemini File API
        uploaded_file = client.files.upload(
            file="temp_resume.pdf",
            config={'display_name': 'Resume PDF'}
        )
        file_uri = uploaded_file.uri

        # Define prompt
        prompt = (
            "Extract the details from the provided resume PDF. "
            "Include the following fields: Name, Email, Phone, Skills (as a list), Education, and Experience. "
            "Return the data in a structured JSON format."
        )

        # Make API call
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                {'file_data': {'file_uri': file_uri, 'mime_type': 'application/pdf'}},
                {'text': prompt}
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': ResumeData,
            },
        )

        # Parse response
        resume_data = response.parsed
        return resume_data, response.text

    finally:
        # Clean up
        if 'uploaded_file' in locals():
            client.files.delete(name=uploaded_file.name)
        if os.path.exists("temp_resume.pdf"):
            os.remove("temp_resume.pdf")

# Function to process job description
def process_job_description(text):
    try:
        # Use Gemini to structure job description
        prompt = (
            f"Convert the following job description into structured JSON with fields: title, requiredSkills (as a list), requiredEducation, and requiredExperience.\n\n"
            f"Job Description: {text}"
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[{'text': prompt}],
            config={
                'response_mime_type': 'application/json',
                'response_schema': JobData,
            },
        )
        job_data = response.parsed
        return job_data, response.text
    except Exception as e:
        st.error(f"Error processing job description: {e}")
        return None, None

# Function to analyze match holistically using Gemini
def analyze_match(resume, job):
    if not resume or not job:
        return None
    
    try:
        # Convert resume and job data to strings for the prompt
        resume_str = json.dumps(resume.dict(), indent=2)
        job_str = json.dumps(job.dict(), indent=2)
        
        # Define a detailed prompt for holistic analysis with brutal honesty
        prompt = (
            "You are an uncompromising HR analyst who tells it like it is. Evaluate the candidate’s resume against the job description with brutal honesty, no sugarcoating. "
            "Consider:\n"
            "- Skills: Which ones match, which are missing, and how critical the gaps are.\n"
            "- Education: Does it meet the job’s needs, or is it irrelevant or underwhelming?\n"
            "- Experience: Is it sufficient in depth, relevance, and years, or does it fall short?\n"
            "- Overall fit: Can this candidate actually do the job, or are they out of their depth?\n\n"
            f"Resume Data:\n{resume_str}\n\n"
            f"Job Description Data:\n{job_str}\n\n"
            "Provide a structured JSON response with:\n"
            "- skills_match: Object with 'matched' (list of matched skills), 'missing' (list of missing skills), 'percentage' (number, e.g., 66.7).\n"
            "- education_fit: String, brutally honest (e.g., 'Completely inadequate').\n"
            "- experience_fit: String, brutally honest (e.g., 'Nowhere near enough').\n"
            "- suitability_score: Integer (0-100), reflecting your unfiltered judgment.\n"
            "- summary: String, blunt assessment of the candidate’s fit."
        )

        # Make API call for analysis
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[{'text': prompt}],
            config={
                'response_mime_type': 'application/json',
                'response_schema': AnalysisData,
            },
        )

        return response.parsed

    except Exception as e:
        st.error(f"Error analyzing match: {e}")
        return None

# Function to answer user questions using Gemini
def answer_question(question, resume, job, analysis):
    if not resume or not job or not question:
        return "Please process the resume and job description first, and enter a question."
    
    try:
        # Convert resume, job data, and analysis to strings for the prompt
        resume_str = json.dumps(resume.dict(), indent=2)
        job_str = json.dumps(job.dict(), indent=2)
        analysis_str = json.dumps(analysis.dict(), indent=2) if analysis else "No analysis available."
        
        # Define a prompt for answering the question with brutal honesty
        prompt = (
            "You are an uncompromising HR analyst who doesn’t hold back. Answer the question about the candidate’s suitability for the job based on their resume, job description, and previous analysis. "
            "Be brutally honest, direct, and clear—don’t soften the truth. If the question is broad (e.g., overall suitability), give a no-nonsense judgment. If specific (e.g., a skill), call out strengths or weaknesses bluntly.\n\n"
            f"Resume Data:\n{resume_str}\n\n"
            f"Job Description Data:\n{job_str}\n\n"
            f"Previous Analysis:\n{analysis_str}\n\n"
            f"Question: {question}\n\n"
            "Return the response as a plain text string."
        )

        # Make API call for the answer
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[{'text': prompt}],
            config={'response_mime_type': 'text/plain'},
        )

        return response.text

    except Exception as e:
        st.error(f"Error answering question: {e}")
        return "Failed to answer the question due to an error."

# Process button
if st.button("Analyze Match"):
    if resume_file and job_description:
        with st.spinner("Processing resume..."):
            resume_data, resume_raw = process_resume(resume_file)
            if resume_data:
                st.session_state.resume_json = resume_data.dict()
        
        with st.spinner("Processing job description..."):
            job_data, job_raw = process_job_description(job_description)
            if job_data:
                st.session_state.job_json = job_data.dict()
        
        if st.session_state.resume_json and st.session_state.job_json:
            # JSONs are processed but not displayed
            pass
            
            # Analyze match
            with st.spinner("Analyzing match..."):
                analysis = analyze_match(
                    ResumeData(**st.session_state.resume_json),
                    JobData(**st.session_state.job_json)
                )
                st.session_state.last_analysis = analysis

            # Display dashboard
            if analysis:
                st.subheader("Match Analysis Dashboard")
                
                # Layout with columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Skills Match Table
                    st.markdown("**Skills Match**")
                    skills_data = {
                        "Matched Skills": ", ".join(analysis.skills_match.matched) or "None",
                        "Missing Skills": ", ".join(analysis.skills_match.missing) or "None",
                        "Match Percentage": f"{analysis.skills_match.percentage:.1f}%"
                    }
                    st.dataframe(skills_data, use_container_width=True)
                    
                    # Education and Experience Fit
                    st.markdown("**Education Fit**")
                    st.write(analysis.education_fit)
                    st.markdown("**Experience Fit**")
                    st.write(analysis.experience_fit)
                
                with col2:
                    # Suitability Score
                    st.markdown("**Suitability Score**")
                    st.metric(label="Score (0-100)", value=analysis.suitability_score)
                
                # Summary
                st.markdown("**Summary**")
                st.write(analysis.summary)
                
                # Add to messages for chat history
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"Analysis completed: Suitability Score {analysis.suitability_score}/100"
                })
    else:
        st.error("Please upload a resume PDF and enter a job description.")

# Chat-like interface
st.subheader("Match Analysis Q&A")
question = st.text_input("Ask a question about the candidate (e.g., 'Is this candidate suitable for the job?' or 'Does the candidate have Python skills?')")
if st.button("Submit Question"):
    if question and st.session_state.resume_json and st.session_state.job_json:
        resume = ResumeData(**st.session_state.resume_json)
        job = JobData(**st.session_state.job_json)
        analysis = st.session_state.last_analysis
        
        # Get answer from Gemini
        with st.spinner("Generating answer..."):
            answer = answer_question(question, resume, job, analysis)
        
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "system", "content": answer})
    else:
        st.error("Please process the resume and job description first, and enter a question.")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])