# AI-Powered Assamese Reading Fluency Assessment Platform using a Fine-Tuned Whisper Model

## Project Overview

The AI-Powered Assamese Reading Fluency Assessment Platform is a web-based application developed to evaluate the reading fluency of students in the Assamese language. The system uses a fine-tuned Whisper speech recognition model to convert a student's speech into text and compares it with the expected reading passage to calculate reading accuracy.

The application also assigns a reading fluency rating based on the student's performance and stores the results for teachers to review through a dashboard. The dashboard allows teachers to monitor student performance and download the assessment results in CSV format for future reference.

This project was developed using Python, Flask, HTML, CSS, JavaScript, and OpenAI's Whisper model.


## Features

### Student Features

- Record and submit Assamese speech through the web interface.
- Convert speech to text using a fine-tuned Whisper model.
- Compare the transcribed text with the expected reading passage.
- Calculate reading accuracy automatically.
- Display a reading fluency rating (Excellent, Good, Average, or Needs Practice).

### Teacher Features

- View assessment results through a teacher dashboard.
- Monitor the reading performance of multiple students.
- Download student assessment results in CSV format.

### System Features

- Web-based interface developed using Flask.
- Fine-tuned Whisper model for Assamese speech recognition.
- Automatic storage of assessment results.
- Simple and user-friendly interface for classroom use.


## System Workflow

The overall workflow of the application is shown below:

```text
Student Reads the Assamese Passage
                │
                ▼
        Voice Recording
                │
                ▼
 Fine-Tuned Whisper Model
                │
                ▼
     Speech Transcription
                │
                ▼
Comparison with Expected Text
                │
                ▼
     Accuracy Calculation
                │
                ▼
 Reading Fluency Rating
                │
                ▼
   Store Result in CSV File
                │
                ▼
Teacher Dashboard & Download Results
```


## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Backend development and application logic |
| Flask | Web framework for building the application |
| OpenAI Whisper | Assamese speech recognition |
| PyTorch | Running and fine-tuning the Whisper model |
| HTML | Structure of the web pages |
| CSS | Styling the user interface |
| JavaScript | Audio recording and client-side interactions |
| CSV | Storing student assessment results |


## Project Structure

```text
assamese-reading-fluency/
│
├── static/
│   └── style.css                 # Stylesheet
│
├── templates/
│   ├── home.html                 # Home page
│   ├── index.html                # Reading Fluency Analyzer
│   └── dashboard.html            # Teacher Dashboard
│
├── app.py                        # Main Flask application
├── create_csv.py                 # Creates the CSV file
├── dataset.csv                   # Dataset used for training
├── train_whisper.py              # Whisper fine-tuning script
├── train_whisper_full.py         # Complete training script
├── test_model.py                 # Test the trained model
├── requirements.txt              # Required Python packages
├── README.md
└── .gitignore
```


## Prerequisites

Before running this project, make sure the following software is installed on your system:

- Python 3.10 or later
- Git
- FFmpeg
- A virtual environment (recommended)
- The required Python packages listed in `requirements.txt`
- A fine-tuned Whisper model placed in the project directory


## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kundanika238/assamese-reading-fluency.git
```

### Step 2: Navigate to the Project Directory

```bash
cd assamese-reading-fluency
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Step 5: Install the Required Python Packages

```bash
pip install -r requirements.txt
```

### Step 6: Install FFmpeg

Download and install FFmpeg, then make sure it is added to your system's PATH.

### Step 7: Place the Fine-Tuned Whisper Model

Copy the fine-tuned Whisper model into the project directory and ensure that the model path in `app.py` points to the correct location before running the application.


## Running the Project

After completing the installation, start the Flask application using the following command:

```bash
python app.py
```

If the application starts successfully, open your web browser and visit:

```text
http://127.0.0.1:5001
```

You should see the home page of the Assamese Reading Fluency Assessment Platform.


## Using the Application

1. Launch the application and open it in your web browser.
2. Click **Reading Fluency Analyzer** on the home page.
3. Enter the student's name.
4. Enter the expected Assamese reading passage.
5. Click the **Record** button and read the passage aloud.
6. Wait for the speech to be transcribed by the fine-tuned Whisper model.
7. View the generated transcription, reading accuracy, and fluency rating.
8. Open the **Teacher Dashboard** to view all assessment records.
9. Download the assessment results as a CSV file if required.


## Model Information

This project uses a fine-tuned OpenAI Whisper model for Assamese speech recognition.

### Base Model

- OpenAI Whisper Tiny

### Fine-Tuning

The Whisper Tiny model was fine-tuned using an Assamese speech dataset prepared specifically for this project. The training process helps the model improve its ability to recognize Assamese speech for reading fluency assessment.

### Purpose of Fine-Tuning

The fine-tuned model is used to:

- Transcribe Assamese speech into text.
- Improve recognition accuracy for the reading passages used in the application.
- Evaluate students based on the similarity between the expected text and the transcribed text.

## Dataset Information

The dataset used in this project was created specifically for training the Assamese speech recognition model.

- All speech recordings were recorded by me.
- The recordings were collected using my own voice under different reading conditions.
- The dataset contains Assamese reading passages prepared for reading fluency assessment.
- The recorded audio was preprocessed and used to fine-tune the Whisper Tiny model for this project.

## Future Improvements

The following features can be added in future versions of the project:

- Expand the speech dataset by collecting recordings from multiple speakers of different ages, genders, and accents to improve the model's robustness and generalization.
- Support for multiple Assamese reading passages based on different grade levels.
- Student login and individual progress tracking.
- Graphical visualization of reading performance over time.
- Integration with a database for long-term record management.
- Real-time pronunciation feedback during reading.
- Support for additional Indian languages.
- Deployment on a cloud platform for online accessibility.
- Mobile-friendly interface for use on smartphones and tablets.
