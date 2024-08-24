import json
import chromadb
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    send_from_directory,
    make_response,
)
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
import pandas as pd

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_collection1")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Create the uploads folder parallel to the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
UPLOAD_FOLDER = os.path.join(parent_folder, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "py", "ipynb"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Setup SQLite database in the uploads folder
DATABASE_URL = os.path.join(UPLOAD_FOLDER, "resumes.db")
engine = create_engine(f"sqlite:///{DATABASE_URL}")
Base = declarative_base()


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(String, primary_key=True)  # ID should be a string (filename)
    name = Column(String, nullable=False)  # Name should be a string
    contact_details = Column(JSON, nullable=False)  # Contact details should be JSON
    skills = Column(JSON, nullable=False)  # Skills should be JSON
    projects = Column(JSON, nullable=False)  # Projects should be JSON
    education = Column(JSON, nullable=False)  # Education should be JSON
    experience = Column(JSON, nullable=True)  # Experience should be JSON (nullable)
    certifications = Column(
        JSON, nullable=True
    )  # Certifications should be JSON (nullable)
    achievements = Column(JSON, nullable=True)  # Achievements should be JSON (nullable)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def get_file_extension(filename):
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename):
    file_extension = get_file_extension(filename)
    return "." in filename and file_extension in ALLOWED_EXTENSIONS


def pdf_file_loader(filename):
    loader = PyPDFLoader(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    pages = loader.load_and_split()
    text = "\n".join([page.page_content for page in pages])
    return text


def doc_file_loader(filename):
    loader = Docx2txtLoader(os.path.join(app.config["UPLOAD_FOLDER"], filename))
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
    certifications: Dict[str, str]
    achievements: Dict[str, str]


def generate_data(text):
    template = """you will be given a resume of a person. 
    Please summarize it and identify 'name', 'contact_details', 'skills', 'education', 'experience', 'projects', 'certifications', 'achievements'.
    create a json format of the resume with all the key details. resume: {text}"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key=api_key, max_tokens=3000, temperature=0.7)
    json_llm = llm.bind(response_format={"type": "json_object"})
    llm_chain = prompt | json_llm
    question = (
        "create a json format of the resume with all the key details. resume: " + text
    )
    final = llm_chain.invoke(input=question)
    print("--------------------------------------------------")
    print(final.content)
    print("--------------------------------------------------")
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
        certifications=final_dict.get("certifications", {}),
        achievements=final_dict.get("achievements", {}),
    )
    return final_dict


def save_to_database(data, file):
    # Serialize data to JSON
    contact_details = json.dumps(data.contact_details)
    skills = json.dumps(data.skills)
    education = json.dumps(data.education)
    experience = json.dumps(data.experience)
    projects = json.dumps(data.projects)
    certifications = json.dumps(data.certifications)
    achievements = json.dumps(data.achievements)

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
            contact_details=contact_details,
            skills=skills,
            education=education,
            experience=experience,
            projects=projects,
            certifications=certifications,
            achievements=achievements,
        )
        session.add(resume)

    session.commit()


def save_to_chroma(data, file):
    try:
        chroma_client = chromadb.PersistentClient(path="../chroma")
        collection_name = CHROMA_COLLECTION_NAME
        collections = [col.name for col in chroma_client.list_collections()]

        # Create or get existing collection
        if collection_name in collections:
            chroma_collection = chroma_client.get_collection(collection_name)
        else:
            chroma_collection = chroma_client.create_collection(collection_name)
            print(f"New collection '{collection_name}' created.")

        # Prepare data for upsert
        combined_text = str(
            [data.name, data.skills, data.projects, data.certifications]
        )

        # Serialize metadata to a JSON string
        meta_data = {
            "contact_details": data.contact_details,
            "education": data.education,
            "experience": data.experience,
            "achievements": data.achievements,
        }
        serialized_meta_data = json.dumps(meta_data)

        # Upsert into Chroma
        chroma_collection.upsert(
            documents=[combined_text],
            ids=[file],
            metadatas=[{"student_details": serialized_meta_data}],
        )
        print(f"Data for '{file}' successfully saved to Chroma.")

    except Exception as e:
        print(f"An error occurred while saving to Chroma: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Handling multiple file uploads
    files = request.files.getlist("resumeFiles")
    # Handling folder upload
    folder = request.files.getlist("resumeFolder")

    all_files = files + folder

    if not all_files:
        flash("No files or folder selected")
        return redirect(request.url)

    for file in all_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            file_extension = get_file_extension(filename)
            if file_extension == "pdf":
                content = pdf_file_loader(filename)
            elif file_extension in ["doc", "docx"]:
                content = doc_file_loader(filename)
            # Add other file type handling if necessary

            resume_data = generate_data(content)
            save_to_database(resume_data, filename)
            save_to_chroma(resume_data, filename)

    flash("Files and/or folder uploaded successfully")
    return redirect(url_for("index"))


@app.route("/resumes")
def list_resumes():
    # Get the page number from the URL query string
    page = request.args.get("page", 1, type=int)
    per_page = 10  # Number of resumes to display per page

    # Query all resumes from the database
    resumes = session.query(Resume).all()

    # Calculate the total number of pages
    total_pages = (len(resumes) + per_page - 1) // per_page

    # Slice the resumes list to get the resumes for the current page
    start = (page - 1) * per_page
    end = start + per_page
    resumes_paginated = resumes[start:end]

    # Render the template with the paginated resumes
    return render_template(
        "resumes.html", resumes=resumes_paginated, page=page, total_pages=total_pages
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download-data", methods=["GET"])
def download_data():
    # Query all resume data from the database
    resumes = session.query(Resume).all()

    # Convert data to a list of dictionaries
    resume_data = []
    for resume in resumes:
        resume_data.append(
            {
                "id": resume.id,
                "name": resume.name,
                "contact_details": resume.contact_details,  # No need to load it again if it's already a dict
                "skills": resume.skills,  # Assuming these are already dicts/arrays
                "education": resume.education,
                "experience": resume.experience if str(resume.experience) else None,
                "projects": resume.projects,
                "certifications": (
                    resume.certifications if str(resume.certifications) else None
                ),
                "achievements": (
                    resume.achievements if str(resume.achievements) else None
                ),
            }
        )

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(resume_data)

    # Convert DataFrame to CSV and prepare response
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=resumes.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
