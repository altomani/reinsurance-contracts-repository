# Reinsurance Contracts Repository

A large collection of publicly available reinsurance contracts extracted from SEC filings.

## Project Description
This repository contains:
- Thousands of reinsurance contracts, retrieved from the EDGAR database of SEC filings.
- Metadata about the contracts, and additional classification metadata generated with gpt-4o-mini and gemini-2.0-flash.
- The scripts used to search, download and classify the contracts

## Detailed description of the data

### download/
This folder contains the contracts in their original format, mostly HTML, or TXT for the early years, and a few scanned PDFs.

The files are the result of a search in the EDGAR database for all reinsurance-related files, that are Exhibit 10 attachments to a 10-K or 10-Q filing for the years from 2001 to 2024. A large majority of the results of this query are reinsurance contracts, but some are other kinds of agreement that jost happen to mention reinsurance.

### index-download/
This folder contains CSV files, one for each year, with metadata about the files (issuer, title, various SEC document identifiers). The index for year 2021 contains some entries without a corresponding file. These are the entries for which the original file is not available (404 error).

### index-classification/ index-classification-gemini/
Thees folders contain CSV files, one for each year, with additional columns for contract classification, written by gpt-4o-mini and gemini-2.0-flash. 

The PDF files have not been classified because most of them are scanned documents. Some documents have no classification by gemini because it failed to return an answer in the correct format.

### scripts/
The scripts to download and classify the contracts.

## Instructions to run the scripts

### Requirements
- Required packages: requests, sec-api, python-dotenv, pandas, aiohttp, asyncio, nest_asyncio, openai, html2text.
- API keys for https://sec-api.io/, https://platform.openai.com/ and https://aistudio.google.com/.

### Execution Instructions
1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:  
   pip install -r requirements.txt
4. Create a `.env` file in the repository root with the following variables (an `.env_example` with invalid values is included):
   - SEC_API_KEY: Your SEC API key.
   - USER_AGENT_NAME: Your name (for user agent for SEC downloads).
   - USER_AGENT_EMAIL: Your email (for user agent for SEC downloads).
   - OPENAI_API_KEY: Your OpenAI API key.
   - GEMINI_API_KEY: Your Gemini API key.
5. Move to the `scripts` folder and adjust the dates inside the three scripts.
6. To search and download filings, run:

    `python search-download-reinsurance-contracts.py`

7. To classify contracts with gpt-4o-mini, run:

    `python classify-contracts.py`

8. To classify contracts with gemini-2.0-flash, run:

    `python classify-contracts-gemini.py`

The scripts process filings year by year (e.g., from 2002 to 2003) and print progress to the console.


