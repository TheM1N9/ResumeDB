import json
from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
SQL_TABLE_NAME =   os.getenv('SQL_TABLE_NAME')

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Create the uploads folder parallel to the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
UPLOAD_FOLDER = os.path.join(parent_folder, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'py', 'ipynb'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Setup SQLite database in the uploads folder
DATABASE_URL = os.path.join(UPLOAD_FOLDER, "resumes.db")
engine = create_engine(f"sqlite:///{DATABASE_URL}")
Base = declarative_base()

class Resume(Base):
    __tablename__ = SQL_TABLE_NAME
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    contact_details = Column(JSON, nullable=False)
    skills = Column(JSON, nullable=False)
    projects = Column(JSON, nullable=True)
    education = Column(JSON, nullable=False)
    experience = Column(JSON, nullable=True)
    certifications = Column(JSON,nullable=True)
    achievements = Column(JSON,nullable=True)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

def allowed_file(filename):
    file_extension = get_file_extension(filename)
    return '.' in filename and file_extension in ALLOWED_EXTENSIONS

def pdf_file_loader(filename):
    loader = PyPDFLoader(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    pages = loader.load_and_split()
    text = '\n'.join([page.page_content for page in pages])
    return text

def doc_file_loader(filename):
    loader = Docx2txtLoader(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    text = loader.load()
    return text

@dataclass
class OrganizeInput:
    """Class to represent the summarized input."""
    name: str
    contact_details: Dict[str, str]
    skills: Dict[str, str]
    projects: List[Dict[str, str]]
    education: List[Dict[str, str]]
    experience: Dict[str, str]
    certifications: List[Dict[str,str]]
    achievements: List[Dict[str,str]]

def generate_data(text):
    template = """you will be given a resume of a person. 
    Please summarize it and identify 'name', 'contact_details', 'skills', 'education', 'experience', 'projects', 'certifications', 'achievements'.
    create a json format of the resume with all the key details. resume: {text}"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key=api_key, max_tokens=3000, temperature=0.7)
    json_llm = llm.bind(response_format={"type": "json_object"})
    llm_chain = prompt | json_llm
    question = 'create a json format of the resume with all the key details. resume: ' + text
    final = llm_chain.invoke(input=question)
    print('--------------------------------------------------')
    print(final.content)
    print('--------------------------------------------------')
    print(type(final.content))
    try:
        # Attempt to parse the summarized input as JSON
        final_dict = json.loads(final.content)
    except json.JSONDecodeError:
        # If parsing fails, return an empty OrganizeInput
        final_dict = {}

    final_dict = OrganizeInput(
        name=final_dict.get("name", ""),
        contact_details=final_dict.get("contact_details", {}),
        skills=final_dict.get("skills", {}),
        education=final_dict.get("education", []),
        experience=final_dict.get("experience", {}),
        projects=final_dict.get("projects", []),
        certifications=final_dict.get("certifications", []),
        achievements=final_dict.get("achievements", []),
    )
    return final_dict

def save_to_database(data, file):
    # Check if a resume with the given filename already exists
    existing_resume = session.query(Resume).filter_by(id=file).first()

    if existing_resume:
        # Update existing resume
        existing_resume.name = data.name
        existing_resume.contact_details = data.contact_details
        existing_resume.skills = data.skills
        existing_resume.education = data.education
        existing_resume.experience = data.experience
        existing_resume.projects = data.projects
        existing_resume.certifications = data.certifications
        existing_resume.achievements = data.achievements
    else:
        # Create a new resume
        resume = Resume(
            id=file,
            name=data.name,
            contact_details=data.contact_details,
            skills=data.skills,
            education=data.education,
            experience=data.experience,
            projects=data.projects,
            certifications=data.certifications,
            achievements=data.achievements
        )
        session.add(resume)
    
    session.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resumeFiles' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('resumeFiles')

    if not files:
        flash('No selected file')
        return redirect(request.url)

    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if get_file_extension(filename) == 'pdf':
                pdf_content = pdf_file_loader(filename)
                print(pdf_content)
                resume_data = generate_data(pdf_content)
                save_to_database(resume_data, filename)

            elif get_file_extension(filename) == 'doc':
                doc_content = doc_file_loader(filename)
                print(doc_content)
                resume_data = generate_data(doc_content)
                save_to_database(resume_data, filename)

    flash('Files uploaded successfully')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
