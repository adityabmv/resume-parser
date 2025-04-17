import os
import json
import pandas as pd
from llm_adapters.ollama_adapter import OllamaAdapter
from utils import load_pdfs_from_attachments, confirm_fields


def get_valid_path(prompt_msg, is_file=True):
    while True:
        path = input(f"{prompt_msg}: ").strip()
        if (is_file and os.path.isfile(path)) or (not is_file and os.path.isdir(path)):
            return path
        else:
            print(f"❌ Invalid {'file' if is_file else 'directory'} path. Please try again.")


def main():
    print("📂 Welcome to the CV Analyzer CLI Tool")

    # Step 1: Get paths from user
    resume_dir = get_valid_path("Enter path to the folder containing resumes (PDFs)", is_file=False)
    candidate_csv_path = get_valid_path("Enter path to the Candidates CSV file", is_file=True)
    attachment_csv_path = get_valid_path("Enter path to the Attachments CSV file", is_file=True)

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

    # Step 7: Save results
    os.makedirs("output", exist_ok=True)
    with open("output/analysis_results.json", "w") as f:
        json.dump(resumes_data, f, indent=2)

    print("\n✅ All resumes processed. Results saved to `output/analysis_results.json`")


if __name__ == "__main__":
    main()
