from datetime import datetime
from fileinput import filename
import os
import json
from pathlib import Path
import chromadb
import pandas as pd
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    send_from_directory,
    make_response,
    session,
    jsonify,
    request,
)
from pydantic import SecretStr
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Table,
    MetaData,
    text,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    UserMixin,
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
import filetype
from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig
import re
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    DateTime,
)
import hashlib
from sqlalchemy.orm import relationship
import concurrent.futures
from functools import partial
import time
from threading import Lock

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Enter api key...")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key")
DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///resumes.db")

app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
app.secret_key = SECRET_KEY
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt"}

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"  # type: ignore

metadata = MetaData()

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Add this after your other database configurations
engine = create_engine("sqlite:///instance/resumes.db")


@app.template_filter("from_json")
def from_json(value):
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif value is None:
            return {}
        return value
    except (ValueError, TypeError) as e:
        print(f"JSON parsing error: {e}")
        return {}


# Function to create user-specific folder
def create_user_folder(username):
    user_folder = os.path.join("uploads", username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder


# Function to create user-specific table
def create_user_table(username):
    return Table(
        username,
        metadata,
        db.Column("id", db.String, primary_key=True),
        db.Column("name", db.String, nullable=False),
        db.Column("contact_details", db.JSON, nullable=False),
        db.Column("skills", db.JSON, nullable=False),
        db.Column("projects", db.JSON, nullable=False),
        db.Column("education", db.JSON, nullable=False),
        db.Column("experience", db.JSON, nullable=True),
        db.Column("certifications", db.JSON, nullable=True),
        db.Column("achievements", db.JSON, nullable=True),
        db.Column(
            "role_recommendation", db.JSON, nullable=True
        ),  # Update field name here
        extend_existing=True,
    )


# Function to create user-specific Chroma collection
def create_user_chroma_collection(username):
    collection_name = username
    collections = [col.name for col in chroma_client.list_collections()]

    if collection_name in collections:
        return chroma_client.get_collection(collection_name)
    else:
        return chroma_client.create_collection(collection_name)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(
            password, method="pbkdf2:sha256", salt_length=8
        )

    def check_password(self, password):
        return check_password_hash(self.password, password)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    messages = db.relationship(
        "ConversationHistory", backref="chat", lazy=True, cascade="all, delete-orphan"
    )

    def __init__(self, user_id, title):
        self.user_id = user_id
        self.title = title


class ConversationHistory(db.Model):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)

    def __init__(self, user_id, chat_id, timestamp, user_message, bot_response):
        self.user_id = user_id
        self.chat_id = chat_id
        self.timestamp = timestamp
        self.user_message = user_message
        self.bot_response = bot_response


class Resume(db.Model):
    __tablename__ = "resumes"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    contact_details = db.Column(db.JSON, nullable=False)
    skills = db.Column(db.JSON, nullable=False)
    projects = db.Column(db.JSON, nullable=False)
    education = db.Column(db.JSON, nullable=False)
    experience = db.Column(db.JSON, nullable=True)
    certifications = db.Column(db.JSON, nullable=True)
    achievements = db.Column(db.JSON, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


chroma_client = chromadb.PersistentClient()


def get_file_extension(filename):
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename):
    return "." in filename and get_file_extension(filename) in ALLOWED_EXTENSIONS


def pdf_file_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return "\n".join([page.page_content for page in pages])


def doc_file_loader(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()[0].page_content


@dataclass
class OrganizeInput:
    name: str
    contact_details: Dict[str, str]
    skills: Dict[str, str]
    projects: List[Dict[str, str]]
    education: List[Dict[str, str]]
    experience: Dict[str, str]
    certifications: Dict[str, str]
    achievements: Dict[str, str]
    role_recommendation: Dict[str, Dict[str, str]]


def generate_data(text):
    template = """You are a senior HR consultant and talent acquisition specialist. Analyze this resume and provide:

                1. Extract and structure these key components:
                - Personal Information:
                    * Full name, use pascal case for the name.
                    * Contact details (email, phone, location, LinkedIn)
                    * Professional summary

                - Professional Details:
                    * Technical and soft skills 
                    * Work experience (company, title, dates, key achievements)
                    * Education (degrees, institutions, dates)
                    * Notable projects
                    * Certifications
                    * Achievements
                    * Languages (if relevant)

                2. Career Role Analysis:
                - Identify 1-2 PRIMARY job roles that best match their profile based on:
                    * Core technical skills and expertise level
                    * Years and type of experience
                    * Educational background
                    * Industry exposure
                
                - For each recommended role, specify:
                    * Exact job title (use standard industry titles), use pascal case
                    * Seniority level
                    * Key qualifications they meet
                    * Any critical skill gaps
                    * Typical industry segments

                3. Evaluation criteria:
                - Roles must match their experience level exactly
                - At least 80% of their core skills should align with the role
                - Focus on their strongest demonstrated competencies
                - Consider both technical expertise and practical experience

                Output Format as JSON:
                ```json
                {{
                    "name": "string",
                    "contact_details": {{
                                "email": "string",
                                "phone": "string",
                                "location": "string",
                                "linkedin": "string",
                                "portfolio": "string"
                    }},
                    "skills": {{"technical_skills": [list of skills], "soft_skills": [list of skills]}},
                    "experience": [{{
                                "company": "string",
                                "title": "string",
                                "dates": "string",
                                "key_achievements": [list of achievements at the company]
                            }}],
                    "education": [{{
                                "degree": "string",
                                "institution": "string",
                                "dates": "string",
                                "cgpa": "string"
                                }}],
                    "projects": [{{"name": "string", "description": "string", "technologies": [list of technologies] }}],
                    "certifications": [{{"name": "string", "institution": "string" }}],
                    "achievements": [list of achievements],
                    "role_recommendation": {{
                                                "role_1": {{"job_title": "string", "seniority_level": "string", "key_qualifications": [list of skills], "skill_gaps": [list of skill gaps]}}, 
                                                "role_2": {{"job_title": "string", "seniority_level": "string", "key_qualifications": [list of skills], "skill_gaps": [list of skill gaps]}}
                                        }}
                }}
                ```
                You must not include any other information in the JSON output. You can skip any fields that are not present in the resume.

                Resume text: {text}"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(GEMINI_API_KEY),
        model="gemini-2.0-pro-exp",
        temperature=0.1,
    )
    llm_chain = prompt | llm | StrOutputParser()

    final = llm_chain.invoke({"text": text})
    final = re.search(r"```json\n(.*)\n```", final, re.DOTALL)
    if final:
        final = final.group(1)
    else:
        final = ""

    print(f"Candidate data: {final}")

    try:
        final_dict = json.loads(str(final))

        # Convert lists to dictionaries where needed
        if isinstance(final_dict.get("skills", {}), list):
            skills_dict = {
                str(i): skill for i, skill in enumerate(final_dict["skills"])
            }
        else:
            skills_dict = final_dict.get("skills", {})

        if isinstance(final_dict.get("experience", {}), list):
            experience_dict = {
                str(i): exp for i, exp in enumerate(final_dict.get("experience", []))
            }
        else:
            experience_dict = final_dict.get("experience", {})

        if isinstance(final_dict.get("certifications", {}), list):
            cert_dict = {
                str(i): cert
                for i, cert in enumerate(final_dict.get("certifications", []))
            }
        else:
            cert_dict = final_dict.get("certifications", {})

        if isinstance(final_dict.get("achievements", {}), list):
            achieve_dict = {
                str(i): achieve
                for i, achieve in enumerate(final_dict.get("achievements", []))
            }
        else:
            achieve_dict = final_dict.get("achievements", {})

        if isinstance(final_dict.get("role_recommendation", {}), dict):
            role_recommendation = final_dict.get("role_recommendation", {})
        else:
            role_recommendation = final_dict.get("role_recommendation", {})

        # Create instance of OrganizeInput with properly formatted data
        organized_data = OrganizeInput(
            name=final_dict.get("name", ""),
            contact_details=final_dict.get("contact_details", {}),
            skills=skills_dict,
            projects=final_dict.get("projects", []),
            education=final_dict.get("education", []),
            experience=experience_dict,
            certifications=cert_dict,
            achievements=achieve_dict,
            role_recommendation=role_recommendation,
        )

        return organized_data

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return OrganizeInput(
            name="",
            contact_details={},
            skills={},
            projects=[],
            education=[],
            experience={},
            certifications={},
            achievements={},
            role_recommendation={},
        )


def save_data(data, unique_id, username):
    """Refactored function to save data to both database and Chroma"""
    try:
        # Serialize data to JSON with error handling
        def safe_json_dumps(obj):
            if obj is None or obj == "" or isinstance(obj, type(None)):
                return json.dumps({})
            try:
                # Convert to dict first if it's a special object
                if hasattr(obj, "__dict__"):
                    obj = obj.__dict__
                return json.dumps(obj)
            except (TypeError, json.JSONDecodeError) as e:
                print(f"Error serializing object: {obj}, Error: {e}")
                return json.dumps({})

        # Serialize data to JSON with null checks
        contact_details = safe_json_dumps(getattr(data, "contact_details", {}))
        skills = safe_json_dumps(getattr(data, "skills", {}))
        education = safe_json_dumps(getattr(data, "education", []))
        experience = safe_json_dumps(getattr(data, "experience", {}))
        projects = safe_json_dumps(getattr(data, "projects", []))
        certifications = safe_json_dumps(getattr(data, "certifications", {}))
        achievements = safe_json_dumps(getattr(data, "achievements", {}))
        role_recommendation = safe_json_dumps(getattr(data, "role_recommendation", {}))

        print(f"Debug - Role recommendation before save: {role_recommendation}")

        # Create the table
        resume_table = create_user_table(username)
        metadata.create_all(db.engine)

        resume_data = {
            "id": unique_id,
            "name": str(data.name),
            "contact_details": str(contact_details),
            "skills": str(skills),
            "education": str(education),
            "experience": str(experience),
            "projects": str(projects),
            "certifications": str(certifications),
            "achievements": str(achievements),
            "role_recommendation": str(role_recommendation),
        }

        # Check if resume exists using unique_id
        existing_resume = db.session.query(resume_table).filter_by(id=unique_id).first()
        if existing_resume:
            # Update using update() method instead of direct attribute modification
            db.session.execute(
                resume_table.update()
                .where(resume_table.c.id == unique_id)
                .values(**resume_data)
            )
        else:
            # Insert a new resume
            insert_stmt = resume_table.insert().values(**resume_data)
            db.session.execute(insert_stmt)

        db.session.commit()
        print(f"Saved to database with ID: {unique_id}")
        print(f"Role recommendation data being saved: {role_recommendation}")

        # Save to Chroma using unique_id
        save_to_chroma(data, unique_id, username)

    except Exception as e:
        print(f"Error in save_data: {str(e)}")
        print(f"Error type: {type(e)}")
        db.session.rollback()
        raise


def generate_unique_id(data):
    """Generate a unique identifier for a resume based on name and contact details"""

    # Combine name and email (or other unique identifiers) to create a unique string
    unique_string = f"{data.name}_{data.contact_details.get('email', '')}"

    # Create a hash of the unique string
    hash_object = hashlib.md5(unique_string.encode())
    return hash_object.hexdigest()


def save_to_chroma(data, unique_key, username):
    try:
        chroma_collection = create_user_chroma_collection(username)

        combined_text = {
            "name": data.name,
            "contact_details": data.contact_details,
            "education": data.education,
            "skills": data.skills,
            "projects": data.projects,
            "experience": data.experience,
            "certifications": data.certifications,
            "achievements": data.achievements,
            "role_recommendation": data.role_recommendation,
        }

        if data is None:
            raise ValueError("No data found in the database.")

        primary_key = unique_key

        for column, column_data in combined_text.items():
            # Extract the text for the current column and row
            text = [str(column_data)]

            # Create the ID for the current column and row
            id = f"{column}_{primary_key}"

            print(f"id: {id}")

            # Create the metadata dictionary
            meta = {
                "id": primary_key,
                "table_name": username,
                "column_name": column,
            }

            chroma_collection.upsert(
                documents=text,
                ids=id,
                metadatas=meta,
            )
        print(f"Saved to Chroma: {unique_key}")
    except Exception as e:
        print(f"An error occurred while saving to Chroma: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    return render_template("upload.html")


# Add rate limiting for Gemini API
from threading import Lock
from collections import deque

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # Expire old requests in O(1) each
                while self.requests and now - self.requests[0] >= self.time_window:
                    self.requests.popleft()
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return
                sleep_time = (self.requests[0] + self.time_window) - now
            # Sleep outside the lock so other threads can proceed
            if sleep_time > 0:
                time.sleep(sleep_time)

# Create a rate limiter for Gemini API (10 requests per minute)
gemini_rate_limiter = RateLimiter(max_requests=10, time_window=60)


def process_single_file(file, user_folder, username):
    """Process a single file and return the result."""
    # Create a new application context for this thread
    with app.app_context():
        if not file.filename:
            return None, "No filename provided"

        # Get file extension from original filename
        original_filename = secure_filename(file.filename)
        file_extension = get_file_extension(original_filename)

        # File type validation using filetype library
        kind = filetype.guess(file)
        if kind is None:
            # Read the initial bytes for magic-number detection and reset stream
            file.seek(0)
            kind = filetype.guess(file.read(261))
            file.seek(0)

        # Final validation using a set for allowed MIME types
        if kind is None or kind.mime not in {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        }:
            return None, f"Unsupported file type: {original_filename}. Only PDF, DOC, DOCX, and TXT are allowed."

        # Save file temporarily for processing
        temp_filename = f"temp_{original_filename}"
        temp_file_path = os.path.join(user_folder, temp_filename)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        file.save(temp_file_path)

        try:
            if file_extension == "pdf":
                content = pdf_file_loader(temp_file_path)
            elif file_extension in ["doc", "docx"]:
                content = doc_file_loader(temp_file_path)
            elif file_extension == "txt":
                with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                return None, f"Unsupported file type: {file_extension}"

            # Acquire rate limit before calling Gemini API
            gemini_rate_limiter.acquire()

            # Generate structured data from the file content
            structured_data = generate_data(content)

            # Generate unique ID based on the data
            unique_id = generate_unique_id(structured_data)

            # Create new filename with hash and original extension
            new_filename = f"{unique_id}.{file_extension}"
            new_file_path = os.path.join(user_folder, new_filename)

            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            # Move the temporary file to its final location
            os.rename(temp_file_path, new_file_path)

            # Save structured data to the database and Chroma
            save_data(structured_data, unique_id, username)

            return (
                True,
                f"File {original_filename} uploaded and processed successfully!",
            )

        except Exception as e:
            # Clean up temporary file if there's an error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return None, f"Error processing file {original_filename}: {str(e)}"


@app.route("/uploadFile", methods=["POST"])
@login_required
def upload_form():
    # Combine all selected files and folders
    all_files = request.files.getlist("resumeFiles") + request.files.getlist(
        "resumeFolder"
    )

    # Filter out invalid or empty files
    valid_files = [file for file in all_files if file and file.filename]

    if not valid_files:
        flash("No valid files or folder selected", "error")
        return redirect(url_for("upload"))

    # Create user-specific folder
    user_folder = create_user_folder(current_user.username)

    # Process files in parallel with a limited number of workers
    # Use min(10, number of files) to avoid creating too many threads
    max_workers = min(10, len(valid_files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with the fixed arguments
        process_file = partial(
            process_single_file, user_folder=user_folder, username=current_user.username
        )
        # Submit all files for processing
        futures = [executor.submit(process_file, file) for file in valid_files]
        # Get results as they complete
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # Process results
    successful_uploads = []
    failed_uploads = []
    for success, message in results:
        if success:
            successful_uploads.append(message)
        else:
            failed_uploads.append(message)

    # Show appropriate flash messages
    if successful_uploads:
        flash(f"Successfully processed {len(successful_uploads)} files", "success")
    if failed_uploads:
        flash(
            f"Failed to process {len(failed_uploads)} files: {', '.join(failed_uploads)}",
            "error",
        )

    return redirect(url_for("upload"))


@app.route("/resumes")
@login_required
def list_resumes():
    try:
        print("Starting list_resumes function...")

        # Create user table if it doesn't exist
        user_table = create_user_table(current_user.username)
        metadata.create_all(db.engine)

        page = request.args.get("page", 1, type=int)
        role_filter = request.args.get("role", None)
        per_page = 10
        offset = (page - 1) * per_page

        with engine.connect() as conn:
            print("Connected to database...")

            # Debug: Print table structure
            table_info = conn.execute(
                text(f"PRAGMA table_info({current_user.username})")
            ).fetchall()
            print(f"Table structure for {current_user.username}:", table_info)

            # Modified roles query to handle nested JSON structure
            roles_query = text(
                f"""
                WITH RECURSIVE json_tree(role) AS (
                    SELECT json_extract(role_recommendation, '$') as role
                    FROM {current_user.username}
                ),
                extracted_roles(title) AS (
                    SELECT DISTINCT json_extract(value, '$.job_title')
                    FROM json_tree,
                    json_each(role)
                    WHERE json_extract(value, '$.job_title') IS NOT NULL
                )
                SELECT title FROM extracted_roles WHERE title IS NOT NULL
                """
            )
            available_roles = [row[0] for row in conn.execute(roles_query).fetchall()]
            print("Available roles:", available_roles)

            # Base query without pagination
            base_query = f"SELECT * FROM {current_user.username}"
            count_query = f"SELECT COUNT(*) FROM {current_user.username}"

            if role_filter:
                role_condition = """
                    WHERE (
                        SELECT COUNT(*)
                        FROM json_each(json_extract(role_recommendation, '$'))
                        WHERE LOWER(json_extract(value, '$.job_title')) LIKE LOWER(:role)
                        OR LOWER(json_extract(value, '$.job_title')) LIKE LOWER(:partial_role)
                    ) > 0
                """
                base_query += role_condition
                count_query += role_condition
                query_params = {
                    "role": f"%{role_filter}%",
                    "partial_role": f"%{' '.join(role_filter.split())}%",
                }
            else:
                query_params = {}

            # Get total count first
            count_result = conn.execute(text(count_query), query_params).scalar()
            total_count = count_result if count_result is not None else 0

            # Add pagination to base query
            base_query += " LIMIT :limit OFFSET :offset"
            query_params["limit"] = str(per_page)
            query_params["offset"] = str(offset)

            print("Executing query:", base_query)
            print("With parameters:", query_params)

            # Execute paginated query
            result = conn.execute(text(base_query), query_params)
            resumes = result.fetchall()
            print(f"Found {len(resumes)} resumes on current page")

            # Process results
            resumes_data = []
            for idx, row in enumerate(resumes):
                print(f"\nProcessing resume {idx + 1}:")
                resume_dict = dict(row._mapping)
                print("Initial resume data keys:", resume_dict.keys())

                print(
                    f"Type of role recommendation: {type(resume_dict['role_recommendation'])}"
                )

                # Parse all JSON fields
                json_fields = [
                    "contact_details",
                    "skills",
                    "education",
                    "experience",
                    "projects",
                    "certifications",
                    "achievements",
                    "role_recommendation",
                ]

                for field in json_fields:
                    print(f"\nProcessing field: {field}")
                    if field in resume_dict:
                        print(f"Raw {field} value:", resume_dict[field])
                        print(f"{field} type:", type(resume_dict[field]))

                        try:
                            if resume_dict[field] is None:
                                print(f"Setting {field} to default empty value")
                                resume_dict[field] = (
                                    []
                                    if field
                                    in ["projects", "education", "achievements"]
                                    else {}
                                )
                            elif isinstance(resume_dict[field], str):
                                print(f"Parsing {field} as JSON string")
                                parsed_value = json.loads(resume_dict[field])
                                print(
                                    f"Type after initial parsing: {type(parsed_value)}"
                                )

                                # Convert string JSON to Python objects
                                if isinstance(parsed_value, str):
                                    parsed_value = json.loads(parsed_value)

                                # Handle specific field types
                                if field == "projects":
                                    resume_dict[field] = (
                                        parsed_value
                                        if isinstance(parsed_value, list)
                                        else []
                                    )
                                elif field == "achievements":
                                    resume_dict[field] = (
                                        parsed_value
                                        if isinstance(parsed_value, list)
                                        else []
                                    )
                                elif field == "education":
                                    resume_dict[field] = (
                                        parsed_value
                                        if isinstance(parsed_value, list)
                                        else []
                                    )
                                else:
                                    resume_dict[field] = (
                                        parsed_value
                                        if isinstance(parsed_value, dict)
                                        else {}
                                    )

                                print(f"Final parsed type: {type(resume_dict[field])}")

                            print(f"Final {field} value:", resume_dict[field])
                            print(f"Final {field} type: {type(resume_dict[field])}")

                        except json.JSONDecodeError as e:
                            print(f"Error parsing {field}: {e}")
                            resume_dict[field] = (
                                []
                                if field in ["projects", "education", "achievements"]
                                else {}
                            )
                        except Exception as e:
                            print(f"Unexpected error processing {field}: {e}")
                            resume_dict[field] = (
                                []
                                if field in ["projects", "education", "achievements"]
                                else {}
                            )

                        print(f"Final {field} value:", resume_dict[field])

                print("\nFinal resume dict keys:", resume_dict.keys())
                resumes_data.append(resume_dict)

            # Calculate total pages based on total count
            total_pages = (total_count + per_page - 1) // per_page

            if not resumes_data:
                return render_template(
                    "resumes.html",
                    resumes=[],
                    page=page,
                    per_page=per_page,
                    total=total_count,
                    total_pages=total_pages,
                    available_roles=available_roles,
                    current_role=role_filter,
                )

            return render_template(
                "resumes.html",
                resumes=resumes_data,
                page=page,
                per_page=per_page,
                total=total_count,
                total_pages=total_pages,
                available_roles=available_roles,
                current_role=role_filter,
            )
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback

        print("Full traceback:")
        print(traceback.format_exc())
        flash("Error loading resumes", "error")
        return redirect(url_for("index"))


@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], current_user.username)

    # Check if the file exists with any of the allowed extensions
    for ext in ALLOWED_EXTENSIONS:
        file_path = os.path.join(user_folder, f"{filename}.{ext}")
        if os.path.exists(file_path):
            return send_from_directory(user_folder, f"{filename}.{ext}")

    flash("The requested file does not exist.")
    return redirect(url_for("list_resumes"))


@app.route("/download-data", methods=["GET"])
@login_required
def download_data():
    # Use a raw SQL query to select from the user's specific table
    with db.engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {current_user.username}"))
        resume_data = [dict(row._mapping) for row in result]

    df = pd.DataFrame(resume_data)
    response = make_response(df.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=resumes.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        email_check = User.query.filter_by(email=email).first()
        username_check = User.query.filter_by(username=username).first()
        if email_check:
            flash("Email address already exists")
            return redirect(url_for("signup"))
        if username_check:
            flash("Username already exists")
            return redirect(url_for("signup"))

        new_user = User(username=username, email=email)  # type: ignore
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful, please log in.")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            # flash("Login successful!")
            return redirect(url_for("index"))

        flash("Invalid username or password")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/chat", defaults={"chat_id": None}, methods=["GET", "POST"])
@app.route("/chat/<int:chat_id>", methods=["GET", "POST"])
@login_required
def chat(chat_id):
    # Initialize configurations
    chroma_config = ChromaDBConfig(
        collection_name=current_user.username, persist_path=Path("chroma")
    )

    sqlite_config = SQLiteConnectionConfig(
        db_file=Path("./instance/resumes.db"),
        dataset_table_name=current_user.username,
    )

    # Get user's chats
    user_chats = (
        Chat.query.filter_by(user_id=current_user.id)
        .order_by(Chat.created_at.desc())
        .all()
    )

    # Get current chat or create a new one
    current_chat = None
    if chat_id:
        current_chat = Chat.query.filter_by(
            id=chat_id, user_id=current_user.id
        ).first_or_404()
    elif user_chats:
        current_chat = user_chats[0]
    else:
        # Create a default chat if user has none
        current_chat = Chat(user_id=current_user.id, title="New Chat")
        db.session.add(current_chat)
        db.session.commit()
        user_chats = [current_chat]

    if request.method == "POST":
        user_query = request.form.get("query")

        if not user_query:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": "No query provided"}), 400
            return render_template(
                "chat.html", messages=[], chats=user_chats, current_chat=current_chat
            )

        # Process the user query using the chatbot
        nlqs_instance = NLQS(sqlite_config, chroma_config, chroma_client)
        queried_data = nlqs_instance.execute_nlqs_workflow(user_query, [])

        # Get chat history for context
        chat_history = []
        if current_chat:
            history = (
                db.session.query(ConversationHistory)
                .filter_by(chat_id=current_chat.id)
                .order_by("timestamp")
                .all()
            )
            for msg in history:
                chat_history.extend(
                    [
                        {"type": "user", "text": msg.user_message},
                        {"type": "bot", "text": msg.bot_response},
                    ]
                )

        response = generate_final_response(
            data=queried_data, query=user_query, chat_history=chat_history
        )

        # Save conversation to the current chat
        new_message = ConversationHistory(
            user_id=current_user.id,
            chat_id=current_chat.id,
            timestamp=datetime.utcnow(),
            user_message=user_query,
            bot_response=response,
        )
        db.session.add(new_message)
        db.session.commit()

        # Check if the request is an AJAX request
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"response": response})

        # If not AJAX, redirect back to the chat page
        return redirect(url_for("chat", chat_id=current_chat.id))

    # Get chat messages for GET request
    messages = []
    if current_chat:
        # Use the session instance to query
        chat_history = (
            db.session.query(ConversationHistory)
            .filter_by(chat_id=current_chat.id)
            .order_by("timestamp")
            .all()
        )

        for msg in chat_history:
            messages.extend(
                [
                    {"type": "user", "text": msg.user_message},
                    {"type": "bot", "text": msg.bot_response},
                ]
            )

    return render_template(
        "chat.html", messages=messages, chats=user_chats, current_chat=current_chat
    )


@app.route("/chat/new", methods=["POST"])
@login_required
def new_chat():
    data = request.get_json()
    title = data.get("title", "New Chat")

    chat = Chat(user_id=current_user.id, title=title)
    db.session.add(chat)
    db.session.commit()

    return jsonify({"success": True, "chat_id": chat.id})


@app.route("/chat/<int:chat_id>/rename", methods=["POST"])
@login_required
def rename_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()

    data = request.get_json()
    chat.title = data.get("title", chat.title)
    db.session.commit()

    return jsonify({"success": True})


@app.route("/chat/<int:chat_id>/delete", methods=["POST"])
@login_required
def delete_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    db.session.delete(chat)
    db.session.commit()

    return jsonify({"success": True})


def generate_final_response(data, query, chat_history):
    print(f"User Query {query}")
    # Get the last n messages for LLM context
    # n = 5  # Number of recent messages to use
    # llm_history = [
    #     f"User: {msg['text']}" if msg["type"] == "user" else f"Bot: {msg['text']}"
    #     for msg in chat_history[-n:]
    # ]

    # llm_history_text = "\n".join(llm_history)
    print("-------------------------------------------------")
    print(f"LLM history: {chat_history}")
    print("-------------------------------------------------")

    template = """
    ResumeDB Assistant - An AI-powered Resume Analysis System, created by M1N9. Your task is to answer the user's question with the data you have received.
    Make sure you provide the accurate answers to the users questions.


    OPERATING GUIDELINES:
    1. Data Processing:
    - Analyze incoming data before addressing queries
    - Provide concise data summaries when relevant
    - Never fabricate responses - use only provided data

    2. Query Handling:
    - User Query: {query}
    - Reference database: {data}
    - Consider context (Chat history): {chat_history}

    3. Response Protocol:
    - Format all outputs in Markdown
    - Include relevant resume URLs when referencing specific documents
    - Structure responses for clarity and readability
    - Avoid using code formatting in the Markdown.
    - Never talk anything about the database.

    4. Special Cases:
    - Respond to the user's query in a professional manner.
    - Basic greetings ("hello", "hi", "hey", "greetings"):
        * Respond with contextual greeting
        * Skip detailed analysis
        * Maintain professional tone

    5. Error Handling:
    - Clearly indicate when data is missing or incomplete
    - Request clarification when queries are ambiguous
    - Provide guidance for malformed queries

    The link for the resumes is "http://localhost:5000/uploads/id" example: "http://localhost:5000/uploads/44bfdf9a8bc32d6e453a46....". 
    When outputing the resume link generate it in markdown. example: Resume: [Resume Link](http://localhost:5000/uploads/44bfdf9a8bc32d6e453a46....)

    OUTPUT REQUIREMENTS:
    - Always use Markdown formatting
    - Include data summaries when relevant
    - Maintain consistent response structure
    - Sort the resumes according to the given query.
    User Query: {query}
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(GEMINI_API_KEY),
        temperature=0.1,
        model="gemini-2.0-pro-exp",
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"query": query, "data": data, "chat_history": chat_history}
    )

    print(f"Response: {response.strip()}")

    return response.strip()


@app.route("/logout")
@login_required
def logout():
    logout_user()
    # flash("You have been logged out.")
    return redirect(url_for("index"))


def run():
    with app.app_context():
        db.create_all()
    app.run(debug=True)


@app.template_filter("debug")
def debug_filter(value):
    print(f"Debug - Value type: {type(value)}")
    print(f"Debug - Value: {value}")
    return value


if __name__ == "__main__":
    run()
