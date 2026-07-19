# AI-Powered Assamese Reading Fluency Assessment Platform using a Fine-Tuned Whisper Model

A web-based application that evaluates the reading fluency of Assamese-speaking students using a fine-tuned Whisper speech recognition model. The system transcribes spoken audio, compares it with the expected text, calculates reading accuracy, assigns a fluency rating, and provides a teacher dashboard for monitoring student performance.


## Features

- Assamese speech recognition using a fine-tuned Whisper model
- Reading fluency assessment based on expected text
- Automatic accuracy calculation
- Performance rating (Excellent, Good, Average, Needs Practice)
- Student performance stored in CSV format
- Teacher dashboard for viewing student results
- Download results as a CSV file
- Simple and user-friendly web interface


## Technologies Used

- Python
- Flask
- OpenAI Whisper
- PyTorch
- HTML
- CSS
- JavaScript
- CSV


## Project Structure

```
assamese-reading-fluency/
│
├── static/                 # CSS files
├── templates/              # HTML templates
├── app.py                  # Main Flask application
├── create_csv.py           # Creates CSV file for storing results
├── dataset.csv             # Training dataset
├── test_model.py           # Test the trained model
├── train_whisper.py        # Whisper fine-tuning script
├── train_whisper_full.py   # Full model training script
├── README.md
└── .gitignore
```