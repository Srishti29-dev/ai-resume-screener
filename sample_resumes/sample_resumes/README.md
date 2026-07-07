# AI-Powered Resume Screening Assistant

An AI/NLP-based tool that compares a resume against a job description, computes
a match score, identifies matched and missing skills, and generates
improvement suggestions (using Google's Gemini API, with a rule-based fallback
if no API key is provided).

## Features
- **Text extraction** from `.txt` and `.pdf` resumes.
- **Skill extraction** using a customizable keyword dictionary.
- **Match scoring** with TF-IDF vectorization + cosine similarity.
- **AI-generated suggestions** via the Gemini API (optional).
- **Batch mode** to screen an entire folder of resumes against one job
  description and export results to CSV.

## Tech Stack
Python, Pandas, Scikit-learn (TF-IDF, cosine similarity), Gemini API,
Regex-based NLP, pypdf

## Setup

```bash
git clone https://github.com/<your-username>/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
```

To enable AI-generated suggestions, set your Gemini API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```
(Without a key, the tool automatically falls back to rule-based suggestions.)

## Usage

**Screen a single resume:**
```bash
python resume_screener.py --resume sample_resumes/resume_1.txt --jd sample_data/job_description.txt
```

**Batch screen a folder of resumes:**
```bash
python resume_screener.py --batch_dir sample_resumes --jd sample_data/job_description.txt --output results.csv
```

## Author
Srishti Sinha
