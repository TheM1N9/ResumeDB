from datetime import datetime
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask and SQLAlchemy
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URI", "sqlite:///resumes.db"
)
db = SQLAlchemy(app)


# Define all models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    messages = db.relationship(
        "ConversationHistory", backref="chat", lazy=True, cascade="all, delete-orphan"
    )


class ConversationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey("chat.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)


class Resume(db.Model):
    __tablename__ = "resumes"
    id = db.Column(db.String, primary_key=True)
    filename = db.Column(db.String, nullable=False)
    name = db.Column(db.String, nullable=False)
    contact_details = db.Column(db.JSON, nullable=False)
    skills = db.Column(db.JSON, nullable=False)
    projects = db.Column(db.JSON, nullable=False)
    education = db.Column(db.JSON, nullable=False)
    experience = db.Column(db.JSON, nullable=True)
    certifications = db.Column(db.JSON, nullable=True)
    achievements = db.Column(db.JSON, nullable=True)


def init_db():
    with app.app_context():
        # Drop all existing tables
        db.drop_all()
        print("Dropped all existing tables")

        # Create all tables
        db.create_all()
        print("Created all tables")

        # Verify tables were created
        tables = db.inspect(db.engine).get_table_names()
        print("\nCreated tables:", tables)

        # Create a test admin user
        # admin = User(username="admin", email="admin@example.com")
        # admin.password = generate_password_hash(
        #     "admin123", method="pbkdf2:sha256", salt_length=8
        # )

        # try:
        #     db.session.add(admin)
        #     db.session.commit()
        #     print("\nCreated admin user:")
        #     print("Username: admin")
        #     print("Password: admin123")
        # except Exception as e:
        #     print(f"Error creating admin user: {e}")
        #     db.session.rollback()


if __name__ == "__main__":
    init_db()
    print("\nDatabase initialization completed!")
