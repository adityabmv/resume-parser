import os
import pandas as pd
from typing import List
from tqdm import tqdm

def confirm_fields(df: pd.DataFrame, label: str) -> List[str]:
    print(f"\n📝 Fields available in {label} data:")
    for i, col in enumerate(df.columns):
        print(f"{i}. {col}")
    use_all = input("Use all fields? (y/n): ").strip().lower()
    if use_all == 'y':
        return df.columns.tolist()
    else:
        selected = input("Enter comma-separated column indices to use: ").strip()
        indices = [int(i) for i in selected.split(',')]
        return [df.columns[i] for i in indices]

def load_pdfs_from_attachments(candidates_df, attachments_df, resume_dir, llm):
    data = []
    for _, row in tqdm(attachments_df.iterrows(), total=len(attachments_df)):
        parent_id = row.get('Parent Id')
        pdf_name = row.get('File Name')

        if not parent_id or not pdf_name:
            continue

        file_path = os.path.join(resume_dir, pdf_name)
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        candidate_row = candidates_df[candidates_df['Application Id'] == parent_id]
        if candidate_row.empty:
            print(f"⚠️ No candidate metadata for {parent_id}")
            continue

        candidate_meta = candidate_row.iloc[0].to_dict()
        parsed_resume = llm.parse_resume(file_path)

        data.append({
            "resume_file": pdf_name,
            "candidate": candidate_meta,
            "parsed_resume": parsed_resume
        })

    return data
