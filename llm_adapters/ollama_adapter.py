from llm_adapters.base import LLMAdapter
import ollama
import fitz  # PyMuPDF for PDF parsing
import json
import re

from prompts import build_prompt


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

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a resume parsing assistant."},
                {"role": "user", "content": prompt},
            ],
            format={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "education": {"type": "string"},
                    "experience": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "name",
                    "email",
                    "phone",
                    "skills",
                    "education",
                    "experience",
                ],
            },
        )

        return self._safe_json_parse(response["message"]["content"])

    def analyze_resume_against_job(
        self, resume_data: dict, candidate_meta: dict, job_description: str
    ) -> dict:
        resume_str = json.dumps(resume_data, indent=2)
        candidate_str = json.dumps(candidate_meta, indent=2)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(
                        job_description=job_description,
                        resume=resume_str,
                        output_format="JSON",
                    ),
                },
            ],
            format={
                "type": "object",
                "properties": {
                    "Overall Match Assessment": {
                        "type": "object",
                        "properties": {
                            "Score": {"type": "integer", "minimum": 1, "maximum": 10},
                            "Justification": {"type": "string"},
                        },
                        "required": ["Score", "Justification"],
                    },
                    "Skill and Experience Alignment": {
                        "type": "object",
                        "properties": {
                            "Required Skills": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "Candidate Skills": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Skill": {"type": "string"},
                                        "Evidence": {"type": "string"},
                                        "Proficiency Level": {
                                            "type": "string",
                                            "enum": [
                                                "Beginner",
                                                "Intermediate",
                                                "Advanced",
                                            ],
                                        },
                                        "Gap Analysis": {"type": "string"},
                                    },
                                    "required": [
                                        "Skill",
                                        "Evidence",
                                        "Proficiency Level",
                                    ],
                                },
                            },
                            "Desired Skills": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["Required Skills", "Candidate Skills"],
                    },
                    "Experience Depth & Relevance": {
                        "type": "object",
                        "properties": {
                            "Relevant Experience": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "Impact & Quantifiable Results": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "Years of Experience": {"type": "integer"},
                        },
                        "required": [
                            "Relevant Experience",
                            "Impact & Quantifiable Results",
                        ],
                    },
                    "Action Verb and Achievement Focus": {
                        "type": "object",
                        "properties": {
                            "Action Verb Strength": {
                                "type": "object",
                                "properties": {
                                    "Strong Verbs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "Weak Verbs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "Suggestions": {"type": "string"},
                                },
                            },
                            "Achievement-Oriented Language": {"type": "string"},
                        },
                        "required": [
                            "Action Verb Strength",
                            "Achievement-Oriented Language",
                        ],
                    },
                    "Red Flags & Concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "Overall Recommendation": {
                        "type": "object",
                        "properties": {
                            "Recommendation": {
                                "type": "string",
                                "enum": [
                                    "Strongly Recommend for Interview",
                                    "Recommend with Reservations",
                                    "Do Not Recommend",
                                ],
                            },
                            "Justification": {"type": "string"},
                        },
                        "required": ["Recommendation", "Justification"],
                    },
                },
                "required": [
                    "Overall Match Assessment",
                    "Skill and Experience Alignment",
                    "Experience Depth & Relevance",
                    "Action Verb and Achievement Focus",
                    "Red Flags & Concerns",
                    "Overall Recommendation",
                ],
            },
        )

        return self._safe_json_parse(response["message"]["content"])

    def _safe_json_parse(self, raw_text: str) -> dict:
        try:
            # Remove any junk before/after the JSON (some LLMs add text)
            json_text = re.search(r"\{.*\}", raw_text, re.DOTALL).group(0)
            return json.loads(json_text)
        except Exception as e:
            print("❌ Failed to parse LLM JSON:", e)
            print("🔎 Raw output:\n", raw_text)
            return {"error": "Failed to parse JSON", "raw_output": raw_text}
