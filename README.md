# Reinsurance Contracts Repository

A collection of reinsurance contracts extracted from SEC filings.

## Project Description
This repository retrieves, processes, and classifies reinsurance contracts obtained from SEC filings. It includes scripts for:
- Searching and downloading filings (e.g., 10-K, 10-Q forms)
- Filtering filings containing specific exhibit types
- Classifying contracts using OpenAI models

## Detailed Requirements
- Python 3.7 or above.
- Required libraries: requests, sec-api, python-dotenv, pandas, aiohttp, asyncio, nest_asyncio, openai, html2text.
- Internet connection for API access.

## Installation & Setup
1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:  
   pip install -r requirements.txt
4. Create a `.env` file in the repository root with the following variables:
   - SEC_API_KEY: Your SEC API key.
   - USER_AGENT_NAME: Your name (for user agent).
   - USER_AGENT_EMAIL: Your email (for user agent).
   - OPENAI_API_KEY: Your OpenAI API key.

## Execution Instructions
- To search and download filings, run:
  python search-download-reinsurance-contracts.py
- To classify contracts, run:
  python classify-contracts.py

The scripts process filings year by year (e.g., from 2002 to 2003) and print progress to the console.

## Output Files
- Downloaded filings are saved in the `download` directory.
- A CSV file with metadata (e.g., index-YYYY.csv) is saved in the `index-download` directory.
- The classification results are saved as CSV files (e.g., index-classification-YYYY.csv) in the `classified` directory.

## Additional Notes
- Ensure the `.env` file is correctly populated.
- Check terminal outputs for download and processing errors.
- The project is designed to handle pagination of search results and asynchronous downloads to efficiently manage large batches of documents.


