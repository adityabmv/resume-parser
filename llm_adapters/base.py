from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    @abstractmethod
    def parse_resume(self, pdf_path: str) -> dict:
        pass

    @abstractmethod
    def analyze_resume_against_job(self, resume_data: dict, candidate_meta: dict, job_description: str) -> dict:
        pass
