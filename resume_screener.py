"""
AI Resume Screening Assistant
------------------------------
Analyzes a resume against a job description, computes a match score,
extracts matched/missing skills, and (optionally) generates AI-powered
improvement suggestions using the Gemini API.

Author: Srishti Sinha
"""

import os
import re
import sys
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ---------------------------------------------------------------------------
# 1. TEXT EXTRACTION
# ---------------------------------------------------------------------------

def load_text(file_path: str) -> str:
    """Load text content from a .txt or .pdf file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        if not PDF_SUPPORT:
            raise ImportError("Install 'pypdf' to read PDF files: pip install pypdf")
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# 2. SKILL EXTRACTION
# ---------------------------------------------------------------------------

# A reasonably broad skill dictionary for tech/data/analyst roles.
# Extend this list to match the domains you're screening for.
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "mysql", "postgresql", "mongodb",
    "power bi", "tableau", "excel", "pandas", "numpy", "matplotlib",
    "scikit-learn", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "data visualization", "data analysis",
    "data cleaning", "statistics", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "google cloud", "rest api", "flask", "django",
    "html", "css", "javascript", "react", "node.js", "prompt engineering",
    "gemini api", "openai api", "etl", "data warehousing", "spark",
    "hadoop", "linux", "problem solving", "debugging", "agile", "scrum",
]


def extract_skills(text: str, skill_list=SKILL_KEYWORDS) -> set:
    """Return the set of known skills found in the given text."""
    text_lower = text.lower()
    found = set()
    for skill in skill_list:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


# ---------------------------------------------------------------------------
# 3. MATCH SCORE (TF-IDF + Cosine Similarity)
# ---------------------------------------------------------------------------

def compute_match_score(resume_text: str, jd_text: str) -> float:
    """Compute a 0-100 similarity score between resume and job description."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


# ---------------------------------------------------------------------------
# 4. OPTIONAL: AI-GENERATED SUGGESTIONS (Gemini API)
# ---------------------------------------------------------------------------

def get_ai_suggestions(resume_text: str, jd_text: str, missing_skills: set) -> str:
    """
    Generate improvement suggestions using Google's Gemini API.
    Requires GEMINI_API_KEY to be set as an environment variable.
    Falls back to a rule-based suggestion if no key is available.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return rule_based_suggestions(missing_skills)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are an expert resume reviewer. Compare the resume and job description
        below. In under 120 words, give 3 concise, actionable suggestions to
        improve the resume's match for this job.

        RESUME:
        {resume_text[:3000]}

        JOB DESCRIPTION:
        {jd_text[:2000]}
        """
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"[Info] Gemini API call failed ({e}); using rule-based suggestions instead.")
        return rule_based_suggestions(missing_skills)


def rule_based_suggestions(missing_skills: set) -> str:
    """Fallback suggestion generator when no API key is configured."""
    if not missing_skills:
        return "Great match! Your resume already covers the key skills in this job description."

    top_missing = ", ".join(list(missing_skills)[:5])
    return (
        f"Consider adding or highlighting these skills if you have experience with them: "
        f"{top_missing}. Where possible, quantify achievements (e.g., '%', 'time saved') "
        f"and mirror exact keywords from the job description to improve ATS matching."
    )


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def screen_resume(resume_path: str, jd_path: str) -> dict:
    resume_text = load_text(resume_path)
    jd_text = load_text(jd_path)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = resume_skills & jd_skills
    missing_skills = jd_skills - resume_skills

    score = compute_match_score(resume_text, jd_text)
    suggestions = get_ai_suggestions(resume_text, jd_text, missing_skills)

    return {
        "match_score": score,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "suggestions": suggestions,
    }


def print_report(result: dict):
    print("\n" + "=" * 50)
    print(" AI RESUME SCREENING REPORT")
    print("=" * 50)
    print(f"\nMatch Score: {result['match_score']}%")
    print(f"\nMatched Skills ({len(result['matched_skills'])}):")
    print(", ".join(result["matched_skills"]) or "None found")
    print(f"\nMissing Skills ({len(result['missing_skills'])}):")
    print(", ".join(result["missing_skills"]) or "None — great coverage!")
    print(f"\nSuggestions:\n{result['suggestions']}")
    print("=" * 50 + "\n")


def batch_screen(resume_dir: str, jd_path: str, output_csv: str = "results.csv"):
    """Screen every resume in a folder against one job description."""
    rows = []
    for file in Path(resume_dir).glob("*"):
        if file.suffix.lower() not in [".txt", ".pdf"]:
            continue
        result = screen_resume(str(file), jd_path)
        rows.append({
            "resume_file": file.name,
            "match_score": result["match_score"],
            "matched_skills": ", ".join(result["matched_skills"]),
            "missing_skills": ", ".join(result["missing_skills"]),
        })

    df = pd.DataFrame(rows).sort_values("match_score", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nBatch results saved to {output_csv}\n")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Resume Screening Assistant")
    parser.add_argument("--resume", help="Path to a single resume (.txt or .pdf)")
    parser.add_argument("--jd", required=True, help="Path to the job description (.txt)")
    parser.add_argument("--batch_dir", help="Folder of resumes to screen in bulk")
    parser.add_argument("--output", default="results.csv", help="Output CSV for batch mode")
    args = parser.parse_args()

    if args.batch_dir:
        batch_screen(args.batch_dir, args.jd, args.output)
    elif args.resume:
        result = screen_resume(args.resume, args.jd)
        print_report(result)
    else:
        print("Please provide either --resume <file> or --batch_dir <folder>")
        sys.exit(1)


if __name__ == "__main__":
    main()
