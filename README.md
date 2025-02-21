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

## Contents of the Folders

### download
This folder contains the contracts in their original format, mostly HTML, TXT for the early years, and a few scanned PDFs.

### index-download
This folder contains CSV files, one for each year, with metadata about the files (issuer, title, and other information from EDGAR).

### index-classification
This folder contains CSV files, one for each year, with additional columns for contract classification, done by gpt-4o-mini.

### index-classification-gemini
This folder contains CSV files, one for each year, with additional columns for contract classification, done by gemini-2.0-flash.

## Repository Contents Clarification
This repository contains documents extracted from the EDGAR SEC database that are reinsurance-related Exhibit 10 attachments to a SEC quarterly or yearly filing. Most of them are reinsurance contracts, but not all. They have been classified with the help of gpt-4o-mini and gemini-2.0-flash. The classification identifies which ones are reinsurance contracts and adds some other metadata, like type of treaty, line of business.

The PDF files have not been classified because most of them are scanned documents.

Some documents classified with gemini have no classification because gemini failed to return an answer in the correct format.

Some files from 2021 are missing because they are 404 on EDGAR.
