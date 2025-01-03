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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Enter api key...")
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


def generate_data(text):
    template = """you will be given a resume of a person. 
    Please summarize it and identify 'name', 'contact_details', 'skills', 'education', 'experience', 'projects', 'certifications', 'achievements'.
    create a json format of the resume with all the key details. resume: {text}"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(OPENAI_API_KEY),
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
    print("Raw LLM response:", final)  # Debug print

    try:
        # Attempt to parse the summarized input as JSON
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

        # Create instance of OrganizeInput with properly formatted data
        organized_data = OrganizeInput(
            name=final_dict.get("name", ""),
            contact_details=final_dict.get("contact_details", {}),
            skills=skills_dict,
            projects=final_dict.get(
                "projects", []
            ),  # Keep as list since defined that way
            education=final_dict.get(
                "education", []
            ),  # Keep as list since defined that way
            experience=experience_dict,
            certifications=cert_dict,
            achievements=achieve_dict,
        )
        print("Organized data:", organized_data)  # Debug print
        return organized_data

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")  # Debug print
        # If parsing fails, return an empty OrganizeInput
        return OrganizeInput(
            name="",
            contact_details={},
            skills={},
            projects=[],
            education=[],
            experience={},
            certifications={},
            achievements={},
        )


def save_data(data, file, username):
    """Refactored function to save data to both database and Chroma"""
    # Serialize data to JSON
    contact_details = json.dumps(data.contact_details)
    skills = json.dumps(data.skills)
    education = json.dumps(data.education)
    experience = json.dumps(data.experience)
    projects = json.dumps(data.projects)
    certifications = json.dumps(data.certifications)
    achievements = json.dumps(data.achievements)

    # Dynamically create table
    resume_table = create_user_table(username)
    metadata.create_all(db.engine)

    # Check if a resume with the given filename already exists
    existing_resume = db.session.query(resume_table).filter_by(id=file).first()

    if existing_resume:
        # Update existing resume
        db.session.query(resume_table).filter_by(id=file).update(
            {
                "name": data.name,
                "contact_details": contact_details,
                "skills": skills,
                "education": education,
                "experience": experience,
                "projects": projects,
                "certifications": certifications,
                "achievements": achievements,
            }
        )
    else:
        # Insert a new resume
        insert_stmt = resume_table.insert().values(
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
        db.session.execute(insert_stmt)

    print(f"Saved to database: {file}")
    db.session.commit()

    # Save to Chroma
    save_to_chroma(data, file, username)


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

        # Generate structured data from the file content
        structured_data = generate_data(content)

        # Save structured data to the database and Chroma
        save_data(structured_data, filename, current_user.username)

        flash(f"File {filename} uploaded and processed successfully!", "success")

    return redirect(url_for("upload"))


@app.route("/resumes")
@login_required
def list_resumes():
    page = request.args.get("page", 1, type=int)
    per_page = 10

    # Use a raw SQL query to select from the user's specific table
    with db.engine.connect() as conn:
        result = conn.execute(
            text(
                f"SELECT * FROM {current_user.username} LIMIT {per_page} OFFSET {(page-1)*per_page}"
            )
        )
        resumes = [dict(row._mapping) for row in result]

    # Get the total count of resumes for pagination
    with db.engine.connect() as conn:
        count_result = conn.execute(
            text(f"SELECT COUNT(*) FROM {current_user.username}")
        )
        total_count = count_result.scalar()

    if not resumes or not total_count:
        flash("No resumes found.")
        return redirect(url_for("upload"))

    return render_template(
        "resumes.html",
        resumes=resumes,
        page=page,
        per_page=per_page,
        total_pages=(total_count // per_page) + (total_count % per_page > 0),
    )


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
        ConversationHistory.query.filter_by(user_id=current_user.id)
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

    template = """ResumeDB ATS - Advanced Resume Matching System by M1N9

                    CORE FUNCTIONS:
                    1. Resume Analysis:
                    - Parse and evaluate candidate resumes
                    - Extract key skills, experience, and qualifications
                    - Calculate skill match percentages
                    - Identify experience alignment
                    - Flag relevant certifications

                    2. Job Requirement Processing:
                    - Parse job descriptions (JD)
                    - Identify required/preferred qualifications
                    - Extract key technical/soft skills
                    - Determine experience requirements
                    - Note specific certifications needed

                    3. Matching Algorithm:
                    - Calculate overall match scores
                    - Weight criteria based on JD priorities
                    - Consider years of experience
                    - Evaluate skill overlap
                    - Factor in education requirements
                    - Account for location preferences

                    SYSTEM CONFIGURATION:
                    - Resume format: http://localhost:5000/uploads/id(or)filename
                    - Input structure:
                    * Original JD: {query}
                    * Candidate Resumes: {data}
                    * Interaction Context: {chat_history}

                    RESPONSE PROTOCOL:
                    1. Analysis Steps:
                    - Summarize job requirements
                    - List qualified candidates
                    - Provide match percentages
                    - Highlight key alignments/gaps
                    - Recommend top matches

                    2. Output Format:
                    - Use Markdown consistently
                    - Structure data hierarchically
                    - Include match scores
                    - Provide evidence-based recommendations
                    - Link to relevant resumes

                    3. Special Queries:
                    - Handle basic greetings naturally
                    - Process skill-specific searches
                    - Support experience-based filtering
                    - Enable qualification comparisons

                    ERROR HANDLING:
                    1. Data Validation:
                    - Verify resume accessibility
                    - Check JD completeness
                    - Validate matching criteria
                    - Flag missing information

                    2. Query Processing:
                    - Request clarification when needed
                    - Handle ambiguous requirements
                    - Process partial matches
                    - Manage conflicting criteria

                    MATCHING CRITERIA:
                    1. Technical Alignment:
                    - Required skills match
                    - Technology stack overlap
                    - Tool/platform experience
                    - Programming languages

                    2. Experience Fit:
                    - Years in relevant roles
                    - Industry experience
                    - Project scope alignment
                    - Leadership requirements

                    3. Education/Certification:
                    - Degree requirements
                    - Professional certifications
                    - Continuing education
                    - Specialized training

                    4. Soft Skills:
                    - Communication abilities
                    - Team collaboration
                    - Project management
                    - Problem-solving

                    OUTPUT REQUIREMENTS:
                    1. Standard Response:
                    - Match percentage
                    - Key qualifications met
                    - Notable gaps
                    - Specific recommendations

                    2. Detailed Analysis:
                    - Skill-by-skill breakdown
                    - Experience comparison
                    - Education alignment
                    - Certification matches

                    3. Documentation:
                    - Resume access links
                    - Match justification
                    - Alternative candidates
                    - Improvement suggestions
                    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(OPENAI_API_KEY),
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


if __name__ == "__main__":
    run()
