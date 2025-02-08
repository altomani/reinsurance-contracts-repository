# %% [markdown]
# # Install Dependencies
# Use pip or conda to install necessary packages like requests or edgar libraries.

# %%
# Install Dependencies
# %pip install requests sec-api python-dotenv aiohttp asyncio nest_asyncio pandas

# %% [markdown]
# # Configure Search Parameters
# Define parameters such as company name or keywords to narrow the EDGAR search.

# %%
# Configure Search Parameters

import os
import json
import nest_asyncio
import asyncio
import aiohttp
import pandas as pd
from urllib.parse import urlparse
from sec_api import FullTextSearchApi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the API key from environment variable
api_key = os.getenv('SEC_API_KEY')

# Initialize the FullTextSearchApi with the API key
full_text_search_api = FullTextSearchApi(api_key)

def load_config():
    """Load environment variables and initialize API."""
    load_dotenv()
    api_key = os.getenv('SEC_API_KEY')
    if not api_key:
        raise ValueError("SEC_API_KEY is missing in environment variables.")
    return FullTextSearchApi(api_key)

def build_search_params(year):
    """Build search parameters for a given year."""
    return {
        "query": "reinsurance (contract OR agreement OR treaty) EX- NOT EX-99.1 NOT EX-99.2 NOT EX-13.1",
        "formTypes": ["10-K", "10-Q", "S-1"],
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31"
    }

def perform_search(api, search_params):
    """Perform and return search results."""
    response = api.get_filings(search_params)
    if response:
        print(f"Year {search_params['startDate'][:4]}: Total filings found: {response['total']['value']}")
        return response
    else:
        print(f"Year {search_params['startDate'][:4]}: Error: Request failed")
        return {}

def filter_exhibit_filings(search_results):
    """Filter results to include only filings with 'EX-10' in type."""
    filings = search_results.get("filings", [])
    exhibit_filings = [filing for filing in filings if "EX-10" in filing.get("type", "")]
    print(f"Exhibit 10.xx: {len(exhibit_filings)} filings found.")
    return exhibit_filings

# Prepare for asynchronous downloads.
nest_asyncio.apply()
download_dir = 'download'
os.makedirs(download_dir, exist_ok=True)

async def download_filing(session, filing, semaphore):
    async with semaphore:
        url = filing.get('filingUrl')
        path = urlparse(url).path
        ext = os.path.splitext(path)[1] or '.html'
        filename = os.path.join(download_dir, filing.get('accessionNo') + ext)
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(filename, 'wb') as f:
                        f.write(content)
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {url}: HTTP {resp.status}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

async def download_all_filings(filings):
    semaphore = asyncio.Semaphore(1000)
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0 (Company info@company.com)"}) as session:
        tasks = [download_filing(session, filing, semaphore) for filing in filings]
        for i in range(0, len(tasks), 5):
            await asyncio.gather(*tasks[i:i+5])
            await asyncio.sleep(1)

def save_metadata_to_csv(filings, year):
    """Save metadata to CSV file for a specific year."""
    metadata = [{
        "accessionNo": filing.get("accessionNo"),
        "cik": filing.get("cik"),
        "companyNameLong": filing.get("companyNameLong"),
        "ticker": filing.get("ticker"),
        "description": filing.get("description"),
        "formType": filing.get("formType"),
        "type": filing.get("type"),
        "filingUrl": filing.get("filingUrl"),
        "filedAt": filing.get("filedAt")
    } for filing in filings]
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(download_dir, f"index-{year}.csv"), index=False)
    return df

def process_year(api, year):
    """Process searching and downloading filings for a specific year."""
    search_params = build_search_params(year)
    results = perform_search(api, search_params)
    if results:
        exhibit_filings = filter_exhibit_filings(results)
        if exhibit_filings:
            asyncio.run(download_all_filings(exhibit_filings))
            df = save_metadata_to_csv(exhibit_filings, year)
            return df
        else:
            print(f"Year {year}: No exhibit filings to download.")
            return pd.DataFrame()
    else:
        print(f"Year {year}: Search yielded no results.")
        return pd.DataFrame()

def main():
    api = load_config()
    all_metadata = []
    for year in range(2001, 2025):
        print(f"Processing year: {year}")
        df = process_year(api, year)
        all_metadata.append(df)
        print("-" * 40)
    full_index = pd.concat(all_metadata, ignore_index=True)
    full_index.to_csv(os.path.join(download_dir, "index-full.csv"), index=False)

if __name__ == '__main__':
    main()
