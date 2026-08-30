import time
import dspy
import dotenv
from typing import Literal
import html2text
import logging
import os
import pandas as pd
import asyncio
from textwrap import dedent

max_document_tokens = 70_000
# MAX_REQUEST_TOKENS = 128_000
MAX_RESPONSE_TOKENS = 20_000

ROOT_DIR = "."
OPENROUTER_API_KEY = dotenv.get_key(
    os.path.join(ROOT_DIR, ".env"), "OPENROUTER_API_KEY"
)
if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

model_name, max_document_tokens, lm = (  # Qwen3 MoE
    "qwen3",
    180_000,
    dspy.LM(
        model="openrouter/qwen/qwen3-235b-a22b-2507",
        api_key=OPENROUTER_API_KEY,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    ),
)

# model_name, max_document_tokens, lm = (
#     "gpt-oss",
#     85_000,
#     dspy.LM(
#         model="openrouter/openai/gpt-oss-120b",
#         api_key=OPENROUTER_API_KEY,
#         max_tokens=MAX_RESPONSE_TOKENS,
#         temperature=1.0,
#         top_p=1.0,
#     ),
# )  # GPT Open Source

# model_name, max_document_tokens, lm = (
#     "gemini-flash-lite",
#     400_000,
#     dspy.LM(
#         model="openrouter/google/gemini-2.5-flash-lite",
#         api_key=OPENROUTER_API_KEY,
#         max_tokens=MAX_RESPONSE_TOKENS,
#         temperature=1.0,
#     ),
# )  # Gemini 2.5 Flash Lite

SOURCE_DIR = "download"
INDEX_DIR = "index-download"
DEST_DIR = f"index-classification-{model_name}"

dspy.configure(lm=lm)

NUM_WORKERS = 20


class Tokenizer:
    """Tokenize text for processing"""

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "gpt-oss" or self.model_name == "gemini-flash-lite":
            import tiktoken

            self.tokenizer = tiktoken.get_encoding("o200k_harmony")
        # elif model_name=="gemini-flash-lite":  # The Gemini/Gemma tokenizer and DSPy have incompatible dependencies. We use tiktoken with a large safety margin for Gemini models.
        #     from gemma import gm
        #     self.tokenizer = gm.text.Gemma3Tokenizer()
        elif model_name == "qwen3":
            from litellm import create_pretrained_tokenizer

            self.tokenizer = create_pretrained_tokenizer(
                "Qwen/Qwen3-235B-A22B-Instruct-2507"
            )["tokenizer"]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def encode(self, text: str) -> list[int]:
        if self.model_name == "gpt-oss" or self.model_name == "gemini-flash-lite":
            return self.tokenizer.encode(text)
        elif self.model_name == "qwen3":
            return self.tokenizer.encode(text).ids
        else:
            return []

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class DocumentClassifier(dspy.Signature):
    """Classify documents according to their content"""

    document: str = dspy.InputField()
    metadata: str = dspy.InputField()
    is_reinsurance: bool = dspy.OutputField(
        desc=dedent("""\
                    Is the document a reinsurance agreement? 
                    The following documents are considered reinsurance agreements:
                        - Reinsurance agreements
                        - Retrocession agreements
                        - Reinstatement Premium Protection contracts if they refer to an underlying reinsurance contract
                        - Capital Maintenance agreements
                        - Loss Portfolio Transfer, Adverse Development Cover, Reinsurance To Close and other retrospective reinsurance agreements
                        - Portfolio Transfers in connection to mergers, acquisitions, sales and other corporate restructurings, if they include provisions similar to standard reinsurance agreements
                        - Catastrophe Bonds and other Insurance Linked Securities if they fulfill a risk transfer objective similar to standard reinsurance
                        - Insurance pooling agreements and government insurance schemes that fulfill a risk transfer objective similar to standard reinsurance, like the FHCF, NFIP, CEA in the US
                        - All documents that renew, cancel, novate, terminate, settle, commute, amend, or otherwise modify the terms of an existing reinsurance agreement
                    The following documents are NOT considered reinsurance agreements:
                        - Insurance policies and other direct insurance contracts
                        - Letters of credit
                        - Bonds, loans and credit instruments except for Catastrophe Bonds and other Insurance Linked Securities as described above
                        - Financial guarantees and other agreements that secure the payment obligations arising from insurance or reinsurance contracts
                        - Referral Agreements, Underwriting Agency Agreements between reinsurance companies and reinsurance underwriting management companies or agents
                    """)
    )
    is_main_contract: bool | None = dspy.OutputField(
        desc=dedent("""\
                    Does the document establish a reinsurance or retrocession contract or is it merely an ancillary agreement?                                                          
                    The following documents are considered main contracts:
                        - Entire reinsurance contracts
                        - Contract slips, if they contain all the principal terms of the reinsurance agreement
                        - Proposals for a reinsurance contract, if they contain all the principal terms
                        - Term sheets, if they contain all the principal terms of a reinsurance agreement
                        - Renewals of an existing reinsurance contract
                    Ancillary agreements include endorsement, extension, addendum, commutations, novations, and other documents which modify the terms of an existing reinsurance agreement.
                    If the document contains a main contract and ancillary agreements, answer the question positively.
                    Respond 'None' if not a reinsurance agreement at all.
                    """)
    )
    is_obligatory: bool | None = dspy.OutputField(
        desc=dedent("""\
                    Is the document an obligatory reinsurance agreement?
                    Obligatory reinsurance contracts include:
                    - Reinsurance and retrocession treaties 
                    - Facultative-obligatory contracts and facultative facilities with automatic acceptance
                    - Portfolio transfers in connection to merger, acquisitions or sales of companies, if they include an effective transfer of risk
                    - Other agreements which transfer risk from all insureds in one or more specific portfolios
                    - Reinstatement Premium Protection relative to obligatory contracts
                    Non-obligatory reinsurance agreements include:
                    - Facultative reinsurance contracts
                    - Other agreements which only transfer risks arising from a single or a small number of named insureds.
                    Reply 'None' if not a reinsurance contract or unknown.
                    """)
    )
    proportional_or_non_proportional: (
        Literal["proportional", "non-proportional", "hybrid"] | None
    ) = dspy.OutputField(
        desc=dedent("""\
                    Is the reinsurance proportional or non-proportional?
                    Proportional reinsurance includes Quota Share, Surplus and Variable Quota Share contracts, proportional retrocession of proportional and non-proportional reinsurance, and other agreements transferring an equal share of the risks included in a portfolio, even if some minor non-proportional terms are present, such as a loss ratio cap or a loss participation corridor.
                    Non-proportional reinsurance includes Excess of Loss, Aggregate Excess of Loss and Stop Loss contracts, non-proportional retrocession of proportional and non-proportional reinsurance, Reinstatement Premium Protection on non-proportional underlying contracts, and other reinsurance contracts that are not proportional contracts.
                    Reply 'hybrid' if the document includes separate contracts or separate sections within the contract and some of the contracts or sections are proportional and some non-proportional. Inuring contracts should generally not be considered when classifying the document.
                    Reply 'None' if not a full reinsurance contract or unknown.
                    """)
    )
    insurance_type: Literal["Life", "Non-Life"] | None = dspy.OutputField(
        desc=dedent("""\
                    Does the document pertain to Life or Non-Life insurance? 
                    Life insurance includes Term or Full Life policies, Pension and Annuity contracts, Long-Term Care policies, critical illness and disability insurance, and other related products.
                    Non-Life insurance includes Property, Casualty, Specialty, Marine, Workers' Compensation, Accident, Medical Expenses, Income protection, Sickness, and other related products.
                    If the document pertains to both Life and Non-Life insurance, classify it according to the main type of coverage.
                    Reply 'None' if not a full reinsurance contract or unknown.
                    """)
    )
    class_of_business: (
        Literal[
            "Accident and Health",
            "Property",
            "Casualty",
            "Specialty",
            "CreditMulti-Line",
            "Health",
            "Longevity",
            "Mortality",
            "Other Life",
        ]
        | None
    ) = dspy.OutputField(
        desc=dedent("""\
                    What is the class of business of the underlying insurance portfolio? 
                    For Non-Life, including Health like Non-Life, the available classifications are:
                    Accident and Health:
                      - Personal and Group accident insurance
                      - Medical Expenses
                      - Sickness income protection
                      - Workers compensation
                    Property:
                        - Property damage from Fire and Allied Perils and Natural Catastrophes, including Business interruption and other ancillary coverage
                    Casualty:
                      - Commercial liability, including General liability, Product liability, Professional liability, Medical malpractice, Directors and Officers liability and similar lines
                      - Motor Liability, including Personal injury and Motor own damage if part of the same portfolio
                      - Personal third party liability
                    Credit:
                      - Trade Credit insurance
                      - Mortgage insurance
                      - Surety insurance
                      - Other Credit insurance
                    Specialty:
                      - Marine insurance
                      - Aviation insurance
                      - Transport insurance
                      - Cyber insurance
                      - Terrorism insurance when not part of a broader coverage
                      - Other Specialty insurance
                    For Life, including Health like Life, the available classifications are:
                      - Health, including Long-Term Care and Critical Illness, Disability
                      - Longevity, including Pensions, Annuities, Tontines and similar products
                      - Mortality, including Term Life, Whole Life and similar products
                      - Other Life, including unit-linked investment products and other lines not covered in previous categories
                    For both Life and Non-Life, use Multi-Line in the following cases:
                      - Whole Account reinsurance when not limited to one of the categories above
                      - Coverage that explicitly includes many of the lines of insurance described above, excluding ancillary coverage
                    Reply 'None' if not a full reinsurance contract or unknown.
    """)
    )


amodule = dspy.asyncify(dspy.ChainOfThought(DocumentClassifier))
encoder = Tokenizer(model_name)


def preprocess_document(file: str) -> str:
    if file.endswith(".pdf"):
        return None
    with open(file, "r") as f:
        try:
            text = f.read()
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            return None
    if file.endswith(".html") or file.endswith(".htm"):
        text = html2text.html2text(text)
    tokens = encoder.encode(text)
    if len(tokens) > max_document_tokens:
        logging.warning(
            f"Document {file} is too long ({len(tokens)} tokens), truncating to {max_document_tokens} tokens"
        )
        text = (
            encoder.decode(tokens[: (max_document_tokens // 2)])
            + "\n\n[...some content omitted...]\n\n"
            + encoder.decode(tokens[-(max_document_tokens // 2) :])
        )
    return text


async def classify_file(file: str, metadata: str) -> dict | None:
    logging.info(f"Classifying {file}...")
    if not os.path.exists(file):
        logging.warning(f"File {file} does not exist")
        return None
    text = preprocess_document(file)
    if text is None:
        logging.warning(f"Skipping {file} (unsupported format)")
        return None
    try:
        response = await amodule(document=text, metadata=metadata)
    except Exception as e:
        logging.error(f"Error classifying {file}: {e}")
        return None
    return response


async def main():
    years = range(2001, 2025)  # or use individual years, e.g. [2020, 2021]
    for year in years:
        semaphore = asyncio.Semaphore(NUM_WORKERS)
        start_time = time.time()
        logging.info(f"Processing year {year}...")
        output = []
        index = os.path.join(ROOT_DIR, INDEX_DIR, f"index-{str(year)}.csv")
        index_df = pd.read_csv(index)

        async def process_row(row):
            async with semaphore:
                file = os.path.join(ROOT_DIR, SOURCE_DIR, row["downloadFilename"])
                metadata = row[
                    [
                        "cik",
                        "companyNameLong",
                        "ticker",
                        "description",
                        "formType",
                        "type",
                        "filingUrl",
                        "filedAt",
                    ]
                ].to_string()
                response = await classify_file(file, metadata)
                if response is not None:
                    return {"filename": row["downloadFilename"]} | response.toDict()
                else:
                    return None
                

        tasks = [process_row(row) for _, row in index_df.iterrows()]

        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                output.append(result)

        output_dir = os.path.join(ROOT_DIR, DEST_DIR)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, f"index-classification-{model_name}-{str(year)}.csv"
        )
        pd.DataFrame(output).to_csv(output_file, index=False)
        finish_time = time.time()
        logging.info(
            f"Finished processing year {year} in {finish_time - start_time:.1f} seconds"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
