import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from groq import Groq
from datetime import datetime
import PyPDF2
from PIL import Image
import pytesseract
from pathlib import Path
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter
import plotly.express as px
import warnings
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import os
from dotenv import load_dotenv

# Set page config at the very beginning
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=r"\[W008\] Evaluating Token.similarity based on empty vectors.")

# MongoDB connection
@st.cache_resource
def init_connection():
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client['resume_analyzer']
        logger.info("Connected to MongoDB Atlas")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        return None

db = init_connection()

# Initialize SkillExtractor
@st.cache_resource
def init_skill_extractor():
    nlp = spacy.load("en_core_web_sm")
    return SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

skill_extractor = init_skill_extractor()

# Set up Tesseract
tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Load CSV data
csv_path = Path(__file__).parent / 'data/job skill context(Core competencies).csv'

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

def extract_text_from_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type.startswith("image/"):
        return extract_text_from_image(file)
    else:  # Assume it's a text file
        return file.getvalue().decode("utf-8")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

@st.cache_data
def extract_skills(text):
    annotations = skill_extractor.annotate(text)
    skills = annotations['results']['full_matches'] + annotations['results']['ngram_scored']
    hard_skills = [skill['doc_node_value'].lower() for skill in skills if skill.get('type') != 'lowSurf']
    soft_skills = [skill['doc_node_value'].lower() for skill in skills if skill.get('type') == 'lowSurf']
    return list(set(hard_skills))[:10], list(set(soft_skills))[:10]

def save_job_data(job_description, hard_skills, soft_skills):
    if db is not None:
        job_data = {
            'description': job_description,
            'hard_skills': hard_skills,
            'soft_skills': soft_skills,
            'created_at': datetime.utcnow()
        }
        existing_job = db.jobs.find_one({'description': job_description})
        if existing_job is None:
            result = db.jobs.insert_one(job_data)
            return str(result.inserted_id)
        else:
            db.jobs.update_one(
                {'_id': existing_job['_id']},
                {'$set': {
                    'hard_skills': hard_skills,
                    'soft_skills': soft_skills,
                    'updated_at': datetime.utcnow()
                }}
            )
            return str(existing_job['_id'])
    return None

def save_resume_data(resume_content, hard_skills, soft_skills, job_id):
    if db is not None:
        resume_data = {
            'content': resume_content,
            'hard_skills': hard_skills,
            'soft_skills': soft_skills,
            'job_id': job_id,
            'created_at': datetime.utcnow()
        }
        existing_resume = db.resumes.find_one({
            'content': resume_content,
            'job_id': job_id
        })
        if existing_resume is None:
            result = db.resumes.insert_one(resume_data)
            return str(result.inserted_id)
        else:
            return str(existing_resume['_id'])
    return None

def get_resume_ranking(resume_content, resume_hard_skills, resume_soft_skills, job_id, global_comparison=True):
    if db is None:
        return 0, 0, 0
    
    job = db.jobs.find_one({'_id': ObjectId(job_id)})
    if not job:
        return 0, 0, 0
    
    job_skills = set(job.get('hard_skills', []) + job.get('soft_skills', []))
    resume_skills = set(resume_hard_skills + resume_soft_skills)
    
    if not job_skills:
        return 0, 0, 0
    
    match_percentage = len(resume_skills & job_skills) / len(job_skills) * 100
    
    query = {'job_id': job_id}
    if not global_comparison:
        query['content'] = resume_content
    
    other_resumes = list(db.resumes.find(query))
    other_scores = [
        len(set(resume.get('hard_skills', []) + resume.get('soft_skills', [])) & job_skills) / len(job_skills) * 100
        for resume in other_resumes
    ]
    
    if other_scores:
        percentile = sum(1 for score in other_scores if match_percentage > score) / len(other_scores) * 100
    else:
        percentile = 50  # If no other resumes, assume average
    
    return match_percentage, percentile, len(other_scores)

def display_skills(skills, color):
    cols = st.columns(2)
    for i, skill in enumerate(skills):
        cols[i % 2].markdown(f"<span style='background-color: {color}; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; margin: 0.1rem; display: inline-block;'>{skill}</span>", unsafe_allow_html=True)

def run_skill_analysis(resume_content, job_description, compare_globally):
    resume_hard_skills, resume_soft_skills = extract_skills(resume_content)
    
    if job_description:
        job_hard_skills, job_soft_skills = extract_skills(job_description)
        job_id = save_job_data(job_description, job_hard_skills, job_soft_skills)
        logger.debug(f"Job saved with ID: {job_id}")
        
        save_resume_data(resume_content, resume_hard_skills, resume_soft_skills, job_id)
        
        match_percentage, percentile, compared_resumes = get_resume_ranking(resume_content, resume_hard_skills, resume_soft_skills, job_id, compare_globally)
        
        return {
            'job_hard_skills': job_hard_skills,
            'job_soft_skills': job_soft_skills,
            'resume_hard_skills': resume_hard_skills,
            'resume_soft_skills': resume_soft_skills,
            'match_percentage': match_percentage,
            'percentile': percentile,
            'compared_resumes': compared_resumes
        }
    else:
        return {
            'resume_hard_skills': resume_hard_skills,
            'resume_soft_skills': resume_soft_skills,
            'job_hard_skills': [],
            'job_soft_skills': [],
            'match_percentage': None,
            'percentile': None,
            'compared_resumes': None
        }

def display_skill_analysis(analysis):
    st.sidebar.subheader("Skill Analysis")
    
    if analysis['match_percentage'] is not None:
        st.sidebar.write(f"Your resume matches {analysis['match_percentage']:.2f}% of the required skills for this job.")
        st.sidebar.write(f"Your resume is better than approximately {analysis['percentile']:.2f}% of other applicants for this job. (Compared against {analysis['compared_resumes']} resumes)")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.sidebar.write("Your Hard Skills:")
        display_skills(analysis['resume_hard_skills'], "#28a745")
    
    with col2:
        st.sidebar.write("Your Soft Skills:")
        display_skills(analysis['resume_soft_skills'], "#17a2b8")
    
    if analysis['job_hard_skills']:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.sidebar.write("Hard Skills to Develop:")
            display_skills(list(set(analysis['job_hard_skills']) - set(analysis['resume_hard_skills']))[:5], "#dc3545")
        
        with col2:
            st.sidebar.write("Soft Skills to Develop:")
            display_skills(list(set(analysis['job_soft_skills']) - set(analysis['resume_soft_skills']))[:5], "#dc3545")
        
        fig = px.pie(values=[analysis['match_percentage'], 100 - analysis['match_percentage']], 
                     names=['Matched Skills', 'Missing Skills'], 
                     title='Skill Match Analysis')
        st.sidebar.plotly_chart(fig, use_container_width=True)

def chatbot(resume_content):
    st.subheader("Resume Assistant Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your resume or skills:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if resume_content:
            GROQ_API_KEY = "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"
            client = Groq(api_key=GROQ_API_KEY) 
            completion = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": f"Help a user to understand how to gain soft skills for their resume. Here is the resume:{resume_content}\n\nUser question: {prompt}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            with st.chat_message("assistant"):
                st.markdown(completion.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        else:
            st.error("Please upload a resume.")



# ABOUT PAGE


def about():
    st.title("About")
    st.write("This is a resume assistant application that helps you analyze your skills and improve your resume.")

def main():
    # Initialize session state variables
    if 'uploaded_resume' not in st.session_state:
        st.session_state.uploaded_resume = None
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'skill_analysis' not in st.session_state:
        st.session_state.skill_analysis = None

    selected = option_menu(
        menu_title=None,
        options=["Home", "About"],
        icons=["house", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "nav-link-selected": {"background-color": "red"},
        }
    )

    if selected == "Home":
        st.sidebar.title("Resume Assistant")
        uploaded_resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt", "png", "jpg", "jpeg"])
        
        if uploaded_resume is not None and uploaded_resume != st.session_state.uploaded_resume:
            st.session_state.uploaded_resume = uploaded_resume
            st.session_state.resume_content = extract_text_from_file(uploaded_resume)

        st.session_state.job_description = st.sidebar.text_area("Enter job description (optional)", value=st.session_state.job_description)
        
        compare_globally = st.sidebar.checkbox("Compare against all applicants", value=True, help="Toggle to compare against all applicants or just your own resumes")
        
        if st.sidebar.button("Analyze Skills"):
            if st.session_state.resume_content:
                    analysis = run_skill_analysis(st.session_state.resume_content, st.session_state.job_description, compare_globally)
                    st.session_state.skill_analysis = analysis
            else:
                st.sidebar.warning("Please upload a resume to begin analysis.")

        if st.session_state.skill_analysis:
            display_skill_analysis(st.session_state.skill_analysis)

        # Main area for chatbot
        st.title("Resume Assistant Chatbot")
        if st.session_state.resume_content:
            chatbot(st.session_state.resume_content)
        else:
            st.info("Please upload a resume to start the conversation.")

    elif selected == "About":
        about()

if __name__ == "__main__":
    main()