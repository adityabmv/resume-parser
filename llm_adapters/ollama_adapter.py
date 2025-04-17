from llm_adapters.base import LLMAdapter
import ollama
import fitz  # PyMuPDF for PDF parsing
import json
import re


class OllamaAdapter(LLMAdapter):
    def __init__(self, model_name="mistral"):
        self.model = model_name

    def _read_pdf_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def parse_resume(self, pdf_path: str) -> dict:
        resume_text = self._read_pdf_text(pdf_path)

        prompt = (
            "Extract the following details from the candidate resume below and return as JSON:\n"
            "- name\n- email\n- phone\n- skills (as list)\n- education\n- experience\n\n"
            f"Resume:\n{resume_text}\n\n"
            "Respond in JSON format with keys exactly: name, email, phone, skills, education, experience."
        )

        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are a resume parsing assistant."},
            {"role": "user", "content": prompt}
        ])

        return self._safe_json_parse(response['message']['content'])

    def analyze_resume_against_job(self, resume_data: dict, candidate_meta: dict, job_description: str) -> dict:
        resume_str = json.dumps(resume_data, indent=2)
        candidate_str = json.dumps(candidate_meta, indent=2)

        prompt = (
            "You're an HR analyst. Evaluate the resume data and candidate metadata against the job description."
            "Be brutally honest and return a structured JSON with:\n"
            "- matched_skills[]\n- missing_skills[]\n- skill_match_pct (number)\n"
            "- education_fit (string)\n- experience_fit (string)\n- summary (string)\n- score (0-100)\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Resume Data:\n{resume_str}\n\n"
            f"Candidate Metadata:\n{candidate_str}\n\n"
            "Respond ONLY with the JSON object."
        )

        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are a strict HR evaluator. Respond with brutally honest JSON analysis."},
            {"role": "user", "content": prompt}
        ])

        return self._safe_json_parse(response['message']['content'])

    def _safe_json_parse(self, raw_text: str) -> dict:
        try:
            # Remove any junk before/after the JSON (some LLMs add text)
            json_text = re.search(r'\{.*\}', raw_text, re.DOTALL).group(0)
            return json.loads(json_text)
        except Exception as e:
            print("❌ Failed to parse LLM JSON:", e)
            print("🔎 Raw output:\n", raw_text)
            return {
                "error": "Failed to parse JSON",
                "raw_output": raw_text
            }
