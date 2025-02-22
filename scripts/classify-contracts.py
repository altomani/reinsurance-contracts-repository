import os
import json
import time
import pandas as pd
import openai
import re
import html2text
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is missing in environment variables.")
client = openai.OpenAI()
model_name = "gpt-4o-mini"


DOWNLOAD_DIR = "../download"
INDEX_DOWNLOAD_DIR = "../index-download"
INDEX_CLASSIFICATION_DIR = "../index-classification"
os.makedirs(INDEX_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_CLASSIFICATION_DIR, exist_ok=True)

def get_trimmed_content(content):
    encoder = tiktoken.encoding_for_model(model_name)
    tokens = encoder.encode(content)
    if len(tokens) > 120000:
        tokens = tokens[:120000]
        content = encoder.decode(tokens)
        content = content + "\n\n[...]\n"
    return content

def classify_contract(content, metadata):
    # Trim content to ensure token count does not exceed the limit
    content = get_trimmed_content(content)
    
    prompt = f"""Read carefully the following contract text and answer the questions at the end.

<contract-metadata>

{metadata}

</contract-metadata>

<contract-text>

{content}

</contract-text>

Answer the following questions based on the contract text provided:
1. Is this a reinsurance contract (including retrocession contracts)? (Yes/No)
2. Is this a Life (including Health) or a Non-Life contract? (Answer: Life/Non-Life/NA)
3. Is this an obligatory reinsurance contract (Treaty or Automatic, including Fac-Oblig and Fac Facilities) or Facultative contract? (Answer: Treaty/Facultative/NA)
4. Is this a Proportional or Non-Proportional contract? (Answer: Proportional/Non-Proportional/NA)
5. What is the main class of business of the contract? (Answer: Property/Casualty/Specialty/Health/Life/Multi-line/NA)

The answer must be formatted in the following way:

First your reasoning for each question, enclosed in <think> tags:

<think>
[Your reasoning here]
</think>

Then, the answer to the question, in JSON format, enclosed in <answer> tags:

<answer>
{{
    "reinsurance": "Yes",  // "Yes" | "No"
    "contractType": "Non-Life",  // "Life" | "Non-Life" | "NA"
    "obligatoryType": "Treaty",  // "Treaty" | "Facultative" | "NA"
    "proportional": "Proportional"  // "Proportional" | "Non-Proportional" | "NA"
    "classOfBusiness": "Specialty"  // "Property" | "Casualty" | "Specialty" | "Health" | "Life" | "Multi-line" | "NA"
}}
</answer>
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # store=True  # Uncomment to store the conversation in OpenAI for debugging and evaluations
        )
        message = response.choices[0].message.content
        answers = re.findall(r"<answer>(.*?)</answer>", message, re.DOTALL)
        result = json.loads(answers[0])
        return result
    except Exception as e:
        print(f"Error during classification: {e}")
        return {"reinsurance": "", "contractType": "", "obligatoryType": "", "proportional": "", "classOfBusiness": ""}

def main():
    for year in range(2024, 2025):
        print(f"\nProcessing year: {year}")
        index_path = os.path.join(INDEX_DOWNLOAD_DIR, f"index-{year}.csv")
        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            continue

        df = pd.read_csv(index_path)
        print(f"Loaded index file {index_path} containing {len(df)} rows.")

        # Prepare new columns
        df["classifyingModel"] = model_name
        df["reinsurance"] = ""
        df["contractType"] = ""
        df["obligatoryType"] = ""
        df["proportional"] = ""
        df["classOfBusiness"] = ""

        for i, row in df.iterrows():
            download_filename = row.get("downloadFilename")
            file_path = os.path.join(DOWNLOAD_DIR, download_filename)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}. Skipping row {i}.")
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if os.path.splitext(file_path)[1].lower() in [".txt"]:
                        content = content.strip()
                    elif os.path.splitext(file_path)[1].lower() in [".htm", ".html"]:
                        h = html2text.HTML2Text()
                        h.body_width = 0
                        content = h.handle(content).strip()
                    else:
                        print(f"Unsupported file type: {file_path}. Skipping row {i+1}.")
                        continue
                metadata = row[[
                    "companyNameLong", "description", "formType", "type", "filedAt"
                ]].to_json()
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Skipping row {i}.")
                continue

            print(f"Classifying file n. {i+1}: {download_filename}")
            classification = classify_contract(content, metadata)
            df.at[i, "reinsurance"] = classification.get("reinsurance", "")
            df.at[i, "contractType"] = classification.get("contractType", "")
            df.at[i, "obligatoryType"] = classification.get("obligatoryType", "")
            df.at[i, "proportional"] = classification.get("proportional", "")
            df.at[i, "classOfBusiness"] = classification.get("classOfBusiness", "")
            # To avoid hitting rate limits
            time.sleep(1)

        output_path = os.path.join(INDEX_CLASSIFICATION_DIR, f"index-classification-{year}.csv")
        df.to_csv(output_path, index=False)
        print(f"Updated index saved to {output_path}")

if __name__ == "__main__":
    main()