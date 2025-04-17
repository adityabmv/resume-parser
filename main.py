import os
import json
import pandas as pd
from llm_adapters.ollama_adapter import OllamaAdapter
from utils import load_pdfs_from_attachments, confirm_fields

CONFIG_PATH = os.path.expanduser("~/.cv_config.json")

def get_rating_from_score(score: int) -> str:
    if score is None:
        return "Unknown"
    if score >= 80:
        return "Strong Match"
    elif score >= 60:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Not Recommended"


def get_or_ask_path(key: str, prompt_msg: str, is_file: bool = True) -> str:
    config = load_config()

    # Use stored path if exists and valid
    if key in config and (
        (is_file and os.path.isfile(config[key])) or (not is_file and os.path.isdir(config[key]))
    ):
        print(f"🔁 Using saved path for {key}: {config[key]}")
        return config[key]

    # Otherwise, prompt and save
    while True:
        path = input(f"{prompt_msg}: ").strip()
        if (is_file and os.path.isfile(path)) or (not is_file and os.path.isdir(path)):
            config[key] = path
            save_config(config)
            return path
        else:
            print(f"❌ Invalid {'file' if is_file else 'directory'} path. Please try again.")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def main():
    print("📂 Welcome to the CV Analyzer CLI Tool")

    # Step 1: Get paths from config or prompt
    resume_dir = get_or_ask_path("resume_dir", "Enter path to the folder containing resumes (PDFs)", is_file=False)
    candidate_csv_path = get_or_ask_path("candidate_csv", "Enter path to the Candidates CSV file", is_file=True)
    attachment_csv_path = get_or_ask_path("attachment_csv", "Enter path to the Attachments CSV file", is_file=True)

    # Step 2: Load data
    print("\n🔄 Loading CSVs...")
    candidates_df = pd.read_csv(candidate_csv_path)
    attachments_df = pd.read_csv(attachment_csv_path)

    # Step 3: Confirm columns to use
    candidate_fields = confirm_fields(candidates_df, "candidate")
    attachment_fields = confirm_fields(attachments_df, "attachment")

    # Step 4: Set up LLM backend (Ollama)
    llm_backend = OllamaAdapter(model_name="gemma3:1b")

    # Step 5: Process resumes
    print("\n📄 Processing resumes and parsing...")
    resumes_data = load_pdfs_from_attachments(
        candidates_df[candidate_fields],
        attachments_df[attachment_fields],
        resume_dir=resume_dir,
        llm=llm_backend
    )

    # Step 6: Analyze resumes against a job description
    job_description = input("\n📝 Enter job description for evaluation: ").strip()

    print("\n📊 Generating analysis...")
    for entry in resumes_data:
        result = llm_backend.analyze_resume_against_job(
            resume_data=entry['parsed_resume'],
            candidate_meta=entry['candidate'],
            job_description=job_description
        )
        entry['analysis'] = result
        print(f"\n📌 Analysis for {entry['resume_file']}:")
        print(json.dumps(result, indent=2))

    # Step 7: Create candidate-focused summary
    output_summary = []

    for entry in resumes_data:
        candidate = entry['candidate']
        analysis = entry['analysis']
        
        # Use keys safely in case of variation
        application_id = candidate.get("Application Id") or candidate.get("Application ID")
        full_name = candidate.get("Full Name") or f"{candidate.get('First Name', '')} {candidate.get('Last Name', '')}".strip()

        summary = {
            "application_id": application_id,
            "full_name": full_name,
            "analysis_summary": {
                "score": analysis.get("score") or analysis.get("suitability_score"),
                "rating": get_rating_from_score(analysis.get("score") or analysis.get("suitability_score")),
                "skill_match_pct": analysis.get("skill_match_pct") or analysis.get("skills_match", {}).get("percentage"),
                "matched_skills": analysis.get("matched_skills") or analysis.get("skills_match", {}).get("matched"),
                "missing_skills": analysis.get("missing_skills") or analysis.get("skills_match", {}).get("missing"),
                "education_fit": analysis.get("education_fit"),
                "experience_fit": analysis.get("experience_fit"),
                "summary": analysis.get("summary")
            }
        }

        output_summary.append(summary)

    # Save to final JSON
    os.makedirs("output", exist_ok=True)
    with open("output/analysis_results.json", "w") as f:
        json.dump(output_summary, f, indent=2)

    print("\n✅ Final summarized results saved to `output/analysis_results.json`")


if __name__ == "__main__":
    main()
