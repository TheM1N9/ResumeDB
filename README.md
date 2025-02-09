# AI-Powered Resume Screening System

## Problem Statement

When recruiters approach colleges or organizations with job requirements and descriptions, the typical process involves selecting candidates based primarily on a high CGPA threshold (e.g., CGPA >= 8 or 8.5). In rare cases, candidates are chosen manually. This approach often overlooks other essential skills and qualifications that may be more relevant to the job description.

To address this issue, I propose developing an AI agent that automatically evaluates each resume against the recruiter's specific job requirements. This AI agent will ensure that candidates are selected based on a comprehensive analysis of their skills, experience, and qualifications, rather than solely on their CGPA. This will help recruiters find the most suitable candidates efficiently and effectively.

## Introduction

Recruiters often face challenges in selecting candidates based solely on high CGPA thresholds, potentially overlooking essential skills and qualifications relevant to the job description. To address this issue, this project aims to develop an AI agent that automatically evaluates resumes against specific job requirements, ensuring a comprehensive analysis of skills, experience, and qualifications.

This AI agent helps recruiters find the most suitable candidates efficiently and effectively.

## Prerequisites

- **OpenAI API Key**
- Python 3.10+
- [Poetry](https://python-poetry.org/) (for dependency management and virtual environments)
- Flask
- Gradio
- PyPDF
- langchain-community
- langchain-openai
- sqlalchemy

## Installation and Setup

### Step 1: Install virtual environment

If you haven't installed virtual environment yet, install it using pip:

```sh
pip install virtualenv
```

### Step 2: Install Dependencies

Navigate to the project directory and install the required packages using Poetry:

```sh
python -m venv .venv
```

Activate the virtual environment:

```sh
.venv/scripts/activate
```

### Step 3: Set up .env File


```sh
copy sample.env  to .env and update the values accordingly
```

### Step 4: Run the Flask App

Start the Flask application:

```sh
flask run
```

### Step 5: Upload Resumes

Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or the localhost port mentioned in your terminal. Upload the folder containing resumes (ensure you upload a folder, not individual files) and make sure every file has a unique name.

You will see a pop-up message confirming the resumes are saved successfully. You can verify this by checking the uploaded resumes in the `uploads` folder.

You can go to [http://127.0.0.1:5000/resumes](http://127.0.0.1:5000/resumes) to view the uploaded resumes.

### Step 6: Start the AI Bot

Open a new terminal in your IDE and run the following command:

```sh
poetry run start-bot
```

### Step 7: Access the Gradio Interface

Open your browser and go to [http://127.0.0.1:7860/](http://127.0.0.1:7860/). You will see a Gradio webpage.

Enter the job description in the provided text box and click `Enter`. The AI agent will analyze the uploaded resumes and provide a list of suitable candidates based on the job requirements.

## Project Structure

- `organisation/`: Contains the main Flask application and related scripts.
- `uploads/`: Folder where uploaded resumes are stored.
- `recruiters/`: Folder where recruiter scripts are stored.
- `start-bot`: Script to start the AI bot.

## Usage

1. Upload a folder containing resumes through the Flask web interface.
2. Enter the job description in the Gradio interface.
3. The AI agent will process the resumes and display the most suitable candidates based on the job requirements.

## Contributing

We welcome contributions to improve the project. Feel free to open issues or submit pull requests on our [GitHub repository](https://github.com/TheM1N9/ResumeDB).

## Issues

1. **UI**: We need to shift the user interface from Gradio to Flask to have better control over the URLs. This will eliminate the need to open two different pages and allow us to run both the resume collection page and the chat interface directly using Poetry.

## Future Updates

1. Display the database in the form of a table on a webpage.
2. We need to make use of other open and free models.
3. Create a separate page for students to upload their resume file and enter the job description to check the ATS score and the details of their resume.
