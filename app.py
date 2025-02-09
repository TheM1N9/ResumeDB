from datetime import datetime
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

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, MetaData, text
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


class ConversationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)

    user = db.relationship("User", backref=db.backref("conversations", lazy=True))

    def __init__(self, user_id, timestamp, user_message, bot_response):
        self.user_id = user_id
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
                    * Full name
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
                    * Exact job title (use standard industry titles)
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
        model="gemini-2.0-flash-exp",
        temperature=0.5,
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


def save_data(data, file, username):
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

        # Dynamically create table
        resume_table = create_user_table(username)
        metadata.create_all(db.engine)

        resume_data = {
            "name": data.name,
            "contact_details": contact_details,
            "skills": skills,
            "education": education,
            "experience": experience,
            "projects": projects,
            "certifications": certifications,
            "achievements": achievements,
            "role_recommendation": role_recommendation,
        }

        if db.session.query(resume_table).filter_by(id=file).first():
            # Update existing resume using update()
            db.session.query(resume_table).filter_by(id=file).update(resume_data)
        else:
            # Insert a new resume
            insert_stmt = resume_table.insert().values(id=file, **resume_data)
            db.session.execute(insert_stmt)

        db.session.commit()
        print(f"Saved to database: {file}")
        print(f"Role recommendation data being saved: {role_recommendation}")

        # Save to Chroma
        save_to_chroma(data, file, username)

    except Exception as e:
        print(f"Error in save_data: {str(e)}")
        print(f"Error type: {type(e)}")
        db.session.rollback()
        raise


def save_to_chroma(data, file, username):
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

        primary_key = file

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
        print(f"Saved to Chroma: {file}")
    except Exception as e:
        print(f"An error occurred while saving to Chroma: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    return render_template("upload.html")


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

    for file in valid_files:
        if not file.filename:
            continue

        filename = secure_filename(file.filename)
        file_path = os.path.join(user_folder, filename)

        # File type validation using filetype library
        kind = filetype.guess(file)
        if kind is None:
            flash(f"Invalid file type: {filename}", "error")
            continue
        elif kind.mime not in [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ]:
            flash(
                f"Unsupported file type: {filename}. Only PDF, DOC, DOCX, and TXT are allowed.",
                "error",
            )
            continue

        # Save file
        file.save(file_path)

        # Process based on file extension
        file_extension = get_file_extension(filename)
        if file_extension == "pdf":
            content = pdf_file_loader(file_path)
        elif file_extension in ["doc", "docx"]:
            content = doc_file_loader(file_path)
        else:
            flash(f"Unsupported file type: {file_extension}", "error")
            continue

        # print(f"type of content: {type(content)}")

        # print(f"extracted content: {content}")
        # print(content)

        # Generate structured data from the file content
        structured_data = generate_data(content)

        # Save structured data to the database and Chroma
        save_data(structured_data, filename, current_user.username)

        flash(f"File {filename} uploaded and processed successfully!", "success")

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

            # Modified base query to properly handle JSON string parsing
            base_query = f"SELECT * FROM {current_user.username}"
            if role_filter:
                base_query += """
                    WHERE (
                        SELECT COUNT(*)
                        FROM json_each(json_extract(role_recommendation, '$'))
                        WHERE LOWER(json_extract(value, '$.job_title')) LIKE LOWER(:role)
                        OR LOWER(json_extract(value, '$.job_title')) LIKE LOWER(:partial_role)
                    ) > 0
                """
                params = {
                    "role": f"%{role_filter}%",
                    "partial_role": f"%{' '.join(role_filter.split())}%",
                    "limit": per_page,
                    "offset": offset,
                }
            else:
                params = {"limit": per_page, "offset": offset}

            base_query += " LIMIT :limit OFFSET :offset"
            print("Executing query:", base_query)
            print("With parameters:", params)

            # Execute query
            result = conn.execute(text(base_query), params)
            resumes = result.fetchall()
            print(f"Found {len(resumes)} resumes")

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

            print("\nTotal processed resumes:", len(resumes_data))

            # Add these lines before the template rendering
            count = len(resumes_data)
            total_pages = (count + per_page - 1) // per_page

            if not resumes_data:
                return render_template(
                    "resumes.html",
                    resumes=[],
                    page=page,
                    per_page=per_page,
                    total=0,
                    total_pages=0,
                    available_roles=available_roles,
                    current_role=role_filter,
                )

            return render_template(
                "resumes.html",
                resumes=resumes_data,
                page=page,
                per_page=per_page,
                total=count,
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
    file_path = os.path.join(user_folder, filename)
    if not os.path.exists(file_path):
        flash("The requested file does not exist.")
        return redirect(url_for("list_resumes"))  # Or another appropriate page
    return send_from_directory(user_folder, filename)


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


@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    # Get the user's conversation history from the database
    messages_db = (
        db.session.query(ConversationHistory)
        .filter_by(user_id=current_user.id)
        .order_by(ConversationHistory.timestamp)
        .all()
    )

    # Create a list of messages where both user and bot messages are included
    messages = []
    for msg in messages_db:
        if msg.user_message:
            messages.append({"type": "user", "text": msg.user_message})
        if msg.bot_response:
            messages.append({"type": "bot", "text": msg.bot_response})

    if current_user.is_authenticated:
        chroma_config = ChromaDBConfig(
            collection_name=current_user.username, persist_path=Path("chroma")
        )

        sqlite_config = SQLiteConnectionConfig(
            db_file=Path("./instance/resumes.db"),
            dataset_table_name=current_user.username,
        )

        if request.method == "POST":
            user_query = request.form.get("query")

            if not user_query:
                return render_template("chat.html", messages=messages)

            # Process the user query using the chatbot
            nlqs_instance = NLQS(sqlite_config, chroma_config, chroma_client)
            queried_data = nlqs_instance.execute_nlqs_workflow(user_query, messages)

            response = generate_final_response(
                data=queried_data, query=user_query, chat_history=messages
            )
            print(f"response: {response}")

            # Update conversation history in the database
            new_message = ConversationHistory(
                user_id=current_user.id,
                timestamp=datetime.utcnow(),
                user_message=user_query,
                bot_response=response,
            )
            db.session.add(new_message)
            db.session.commit()

            # Check if the request is an AJAX request, return JSON response
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"response": response})

            # For normal requests, render the HTML template
            messages.append({"type": "user", "text": user_query})
            messages.append({"type": "bot", "text": response})

    return render_template("chat.html", messages=messages)


def generate_final_response(data, query, chat_history):
    # Get the last n messages for LLM context
    n = 5  # Number of recent messages to use
    llm_history = [
        f"User: {msg['text']}" if msg["type"] == "user" else f"Bot: {msg['text']}"
        for msg in chat_history[-n:]
    ]

    llm_history_text = "\n".join(llm_history)
    template = """
    ResumeDB Assistant - An AI-powered Resume Analysis System by M1N9


    OPERATING GUIDELINES:
    1. Data Processing:
    - Analyze incoming data before addressing queries
    - Provide concise data summaries when relevant
    - Never fabricate responses - use only provided data

    2. Query Handling:
    - Process user input: {query}
    - Reference database: {data}
    - Consider context: {chat_history}

    3. Response Protocol:
    - Format all outputs in Markdown
    - Include relevant resume URLs when referencing specific documents
    - Structure responses for clarity and readability

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

    The link for the resumes is "http://localhost:5000/uploads/id(or)filename". 

    OUTPUT REQUIREMENTS:
    - Always use Markdown formatting
    - Include data summaries when relevant
    - Maintain consistent response structure
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(GEMINI_API_KEY),
        temperature=0.3,
        model="gemini-2.0-flash-exp",
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"data": data, "query": query, "chat_history": llm_history_text}
    )

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
