import os
import json
import pandas as pd
from llm_adapters.ollama_adapter import OllamaAdapter
from utils import load_pdfs_from_attachments, confirm_fields

CONFIG_PATH = os.path.expanduser("~/.cv_config.json")
PARSED_RESUMES_PATH = os.path.join("output", "parsed_resumes.json")  # 🔄 New


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


def load_cached_resumes():
    if os.path.exists(PARSED_RESUMES_PATH):
        with open(PARSED_RESUMES_PATH, "r") as f:
            return json.load(f)
    return None


def save_parsed_resumes(resume_data):
    os.makedirs("output", exist_ok=True)
    with open(PARSED_RESUMES_PATH, "w") as f:
        json.dump(resume_data, f, indent=2)


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

    # Step 5: Handle resume parsing with cache support
    resumes_data = None
    if os.path.exists(PARSED_RESUMES_PATH):
        reuse = input("\n🔄 Cached parsed resumes found. Do you want to reparse all PDFs? (y/n): ").strip().lower()
        if reuse == 'n':
            print("✅ Using cached parsed resume data.")
            resumes_data = load_cached_resumes()

    if not resumes_data:
        print("\n📄 Parsing resumes with LLM...")
        resumes_data = load_pdfs_from_attachments(
            candidates_df[candidate_fields],
            attachments_df[attachment_fields],
            resume_dir=resume_dir,
            llm=llm_backend
        )
        save_parsed_resumes(resumes_data)
        print("💾 Parsed resumes cached to reuse in future runs.")

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

    # Step 7: Save results
    os.makedirs("output", exist_ok=True)
    with open("output/analysis_results.json", "w") as f:
        json.dump(resumes_data, f, indent=2)

    print("\n✅ All resumes analyzed. Results saved to `output/analysis_results.json`")


if __name__ == "__main__":
    main()
