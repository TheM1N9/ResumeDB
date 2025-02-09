from dataclasses import dataclass
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Dict, List
import chromadb
import pandas as pd
from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import (
    RedirectResponse,
    FileResponse,
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
)
from fastapi_login import LoginManager
from pydantic import BaseModel
from pydantic.v1 import SecretStr
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    String,
    Integer,
    JSON,
    text,
    DateTime,
)
from sqlalchemy.orm import declarative_base, Session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from nlqs.database.sqlite import SQLiteConnectionConfig
from nlqs.nlqs import NLQS, ChromaDBConfig
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from io import StringIO
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Enter api key...")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key")
DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///resumes.db")

# App initialization
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
engine = create_engine(DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# User authentication management
login_manager = LoginManager(SECRET_KEY, token_url="/login")
# Set up Jinja2Templates
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
# Ensure the upload directory exists
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
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False),
        Column("contact_details", JSON, nullable=False),
        Column("skills", JSON, nullable=False),
        Column("projects", JSON, nullable=False),
        Column("education", JSON, nullable=False),
        Column("experience", JSON, nullable=True),
        Column("certifications", JSON, nullable=True),
        Column("achievements", JSON, nullable=True),
        extend_existing=True,
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to create user-specific Chroma collection
def create_user_chroma_collection(username):
    collection_name = username
    collections = [col.name for col in chroma_client.list_collections()]

    if collection_name in collections:
        return chroma_client.get_collection(collection_name)
    else:
        return chroma_client.create_collection(collection_name)


# User Model
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(150), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(
            password, method="pbkdf2:sha256", salt_length=8
        )

    def check_password(self, password):
        return check_password_hash(self.password, password)


class ConversationHistory(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    user_message = Column(String, nullable=False)
    bot_response = Column(String, nullable=False)


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    contact_details = Column(JSON, nullable=False)
    skills = Column(JSON, nullable=False)
    projects = Column(JSON, nullable=False)
    education = Column(JSON, nullable=False)
    experience = Column(JSON, nullable=True)
    certifications = Column(JSON, nullable=True)
    achievements = Column(JSON, nullable=True)


manager = LoginManager(SECRET_KEY, token_url="/login")


@manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


chroma_client = chromadb.PersistentClient()


def get_file_extension(filename):
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_file_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return "\n".join([page.page_content for page in pages])


def doc_file_loader(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()[0].page_content


def get_user_table(username):
    return Table(
        username,
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False),
        Column("contact_details", JSON, nullable=False),
        Column("skills", JSON, nullable=False),
        Column("projects", JSON, nullable=False),
        Column("education", JSON, nullable=False),
        Column("experience", JSON, nullable=True),
        Column("certifications", JSON, nullable=True),
        Column("achievements", JSON, nullable=True),
    )


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


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", name="upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# Define your upload endpoint
@app.post("/uploadFile")
async def upload_form(
    files: list[UploadFile] = File(...), user: User = Depends(load_user)
):
    user_folder = create_user_folder(user.username)
    for file in files:
        if allowed_file(file.filename):
            file_path = os.path.join(user_folder, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    return {"message": "Files uploaded successfully!"}


@app.get("/resumes")
async def list_resumes(
    page: int = 1,
    per_page: int = 10,
    user: User = Depends(load_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * per_page
    resumes = db.execute(
        text(f"SELECT * FROM {user.username} LIMIT {per_page} OFFSET {offset}")
    ).fetchall()
    return resumes


@app.get("/uploads/{filename}")
async def uploaded_file(filename: str, user: User = Depends(load_user)):
    user_folder = create_user_folder(user.username)
    file_path = os.path.join(user_folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)


@app.get("/download-data")
async def download_data(user: User = Depends(load_user), db: Session = Depends(get_db)):
    # Use a raw SQL query to select from the user's specific table
    query = text(f"SELECT * FROM {user.username}")
    result = db.execute(query)
    resume_data = [dict(row._mapping) for row in result]

    df = pd.DataFrame(resume_data)

    # Use StringIO to handle CSV data as a string buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    response = StreamingResponse(
        iter([csv_buffer.getvalue()]),  # Convert buffer to iterator
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resumes.csv"},
    )
    return response


@app.post("/signup")
async def signup(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    if (
        db.query(User).filter_by(email=email).first()
        or db.query(User).filter_by(username=username).first()
    ):
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


class LoginForm(BaseModel):
    username: str
    password: str


@app.get("/login", name="login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def post_login(username: str = Form(...), password: str = Form(...)):
    # Handle login logic here
    return {"message": "Login successful"}


@app.get("/chat", name="chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat")
async def post_chat(
    query: str = Form(...),
    user: User = Depends(load_user),
    db: Session = Depends(get_db),
):
    messages_db = (
        db.query(ConversationHistory)
        .filter_by(user_id=user.id)
        .order_by(ConversationHistory.timestamp)
        .all()
    )
    messages = [{"type": "user", "text": msg.user_message} for msg in messages_db] + [
        {"type": "bot", "text": msg.bot_response} for msg in messages_db
    ]

    chroma_config = ChromaDBConfig(
        collection_name=user.username, persist_path=Path("chroma")
    )
    sqlite_config = SQLiteConnectionConfig(
        db_file=Path("./instance/resumes.db"), dataset_table_name=user.username
    )

    nlqs_instance = NLQS(sqlite_config, chroma_config, chroma_client)
    queried_data = nlqs_instance.execute_nlqs_workflow(query, messages)
    response = generate_final_response(queried_data, query, messages)

    new_message = ConversationHistory(
        user_id=user.id,
        timestamp=datetime.utcnow(),
        user_message=query,
        bot_response=response,
    )
    db.add(new_message)
    db.commit()

    return templates.TemplateResponse(
        "chat.html",
        {
            "messages": messages
            + [{"type": "user", "text": query}, {"type": "bot", "text": response}]
        },
    )


@app.get("/logout")
async def logout(user: User = Depends(load_user)):
    # Invalidate the user session or token
    # For session-based auth, this might involve clearing session data or cookies
    # For token-based auth, it might involve blacklisting the token or client-side removal

    # Assuming token-based auth, you can just redirect to the login page
    # Client should handle token removal

    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


# def run():
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)


def generate_data(text):
    template = """you will be given a resume of a person. 
    Please summarize it and identify 'name', 'contact_details', 'skills', 'education', 'experience', 'projects', 'certifications', 'achievements'.
    create a json format of the resume with all the key details. resume: {text}"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key=SecretStr(GEMINI_API_KEY), temperature=0.5)
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
        final_dict = json.loads(str(final.content))
    except json.JSONDecodeError:
        # If parsing fails, return an empty OrganizeInput
        final_dict = {}

    # **Create instance of OrganizeInput with parsed data**
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


@app.post("/save-data")
async def save_data_endpoint(
    data: OrganizeInput,
    file: str,
    db: Session = Depends(get_db),
    user: User = Depends(load_user),
):
    """API endpoint to save data to both the database and Chroma"""

    # Serialize the input data to JSON
    contact_details = json.dumps(data.contact_details)
    skills = json.dumps(data.skills)
    education = json.dumps(data.education)
    experience = json.dumps(data.experience)
    projects = json.dumps(data.projects)
    certifications = json.dumps(data.certifications)
    achievements = json.dumps(data.achievements)

    # Dynamically create user-specific table
    resume_table = create_user_table(user.username)
    metadata.create_all(engine)

    # Check if a resume with the given file ID already exists
    existing_resume = db.execute(
        resume_table.select().where(resume_table.c.id == file)
    ).fetchone()

    if existing_resume:
        # Update the existing resume
        db.execute(
            resume_table.update()
            .where(resume_table.c.id == file)
            .values(
                name=data.name,
                contact_details=contact_details,
                skills=skills,
                education=education,
                experience=experience,
                projects=projects,
                certifications=certifications,
                achievements=achievements,
            )
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
        db.execute(insert_stmt)

    # Commit changes to the database
    db.commit()
    print(f"Saved to database: {file}")

    # Save to Chroma
    save_to_chroma(data, file, user.username)

    return {"message": "Data saved successfully to both database and Chroma"}


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


def generate_final_response(data, query, chat_history):
    # Get the last n messages for LLM context
    n = 5  # Number of recent messages to use
    llm_history = [
        f"User: {msg['text']}" if msg["type"] == "user" else f"Bot: {msg['text']}"
        for msg in chat_history[-n:]
    ]

    llm_history_text = "\n".join(llm_history)

    template = """ You are a perfect Automated Tracking System (named ResumeDB) created by M1N9. Your task is to provide an answer based on the user input and the data retrieved from the database.
    Provide an answer based on the user input and the data retrieved from the database. 
    The link for the resumes is "http://localhost:5000/uploads/id(or)filename". Do not add ' before and after the id of the resume.
    Also, briefly summarize and discuss the data you received before addressing the specific user query.
    Do not make answers on your own.

    User input: {data}\n\ndata : {query}\n\nChat history: {chat_history}
    Handle Basic Greetings: If the user input is a simple greeting (e.g., "hello", "hi", "hey", "greetings"), respond with a friendly greeting message. Skip the structured analysis and JSON output for these cases.
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(
        api_key=SecretStr(GEMINI_API_KEY), temperature=0.7, model="gpt-4-turbo"
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke(
        {"data": data, "query": query, "chat_history": llm_history_text}
    )

    return response.strip()


# Helper function for starting the app
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
