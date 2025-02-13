# Configure Search Parameters

import os
import nest_asyncio
import asyncio
import aiohttp
import pandas as pd
from urllib.parse import urlparse
from sec_api import FullTextSearchApi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Read the API key from environment variable
sec_api_key = os.getenv('SEC_API_KEY')
ua_name = os.getenv('USER_AGENT_NAME')
ua_email = os.getenv('USER_AGENT_EMAIL')

# Initialize the FullTextSearchApi with the API key
full_text_search_api = FullTextSearchApi(sec_api_key)

def load_config():
    """Load environment variables and initialize API."""
    api_key = os.getenv('SEC_API_KEY')
    if not api_key:
        raise ValueError("SEC_API_KEY is missing in environment variables.")
    return FullTextSearchApi(api_key)

def build_search_params(year):
    """Build search parameters for a given year."""
    return {
        "query": "reinsurance (contract OR agreement OR treaty) EX- NOT EX-99.1 NOT EX-99.2 NOT EX-13.1",
        "formTypes": ["10-K", "10-Q"], # 10-K: Annual report, 10-Q: Quarterly report, should also include S-1?
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31"
    }

def perform_search(api, search_params):
    """Perform search with pagination and return aggregated results."""
    all_filings = []
    page = 1
    while True:
        search_params.update({"page": page})
        response = api.get_filings(search_params)
        if response:
            filings = response.get("filings", [])
            all_filings.extend(filings)
            print(f"Year {search_params['startDate'][:4]}, Page {page}: {len(filings)} filings retrieved.")
            # if fewer than 100 items returned, assume last page
            if len(filings) < 100:
                break
            page += 1
        else:
            print(f"Year {search_params['startDate'][:4]}, Page {page}: Error: Request failed.")
            break
    print(f"Total filings after pagination: {len(all_filings)}")
    return {"filings": all_filings, "total": {"value": len(all_filings)}}

def filter_exhibit_filings(search_results):
    """Filter results to include only filings with 'EX-10' in type."""
    filings = search_results.get("filings", [])
    exhibit_filings = [filing for filing in filings if "EX-10" in filing.get("type", "")]
    print(f"Exhibit 10.xx: {len(exhibit_filings)} filings found.")
    return exhibit_filings

# Prepare for asynchronous downloads.
nest_asyncio.apply()
download_dir = 'download'
index_download_dir = 'index-download'
os.makedirs(download_dir, exist_ok=True)
os.makedirs(index_download_dir, exist_ok=True)

async def download_filing(session, filing, year, semaphore):
    async with semaphore:
        url = filing.get('filingUrl')
        path = urlparse(url).path
        original_filename = os.path.basename(path) or 'document.html'
        ext = os.path.splitext(path)[1] or '.html'
        # Compose final download file name using year, CIK, accession no., and original filename
        download_filename = f"{year}-{filing.get('cik', 'UNKNOWN')}-{filing.get('accessionNo', 'UNKNOWN')}-{original_filename}"

        filename = os.path.join(download_dir, download_filename)
        # Store for CSV output
        filing["downloadFilename"] = download_filename

        if os.path.exists(filename):
            print(f"Skipped download (already exists): {filename}")
            return
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

async def download_all_filings(filings, year):
    semaphore = asyncio.Semaphore(1000)
    async with aiohttp.ClientSession(headers={"User-Agent": f"Mozilla/5.0 ({ua_name} {ua_email})"}) as session:
        tasks = [download_filing(session, filing, year, semaphore) for filing in filings]
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
        "filedAt": filing.get("filedAt"),
        "downloadFilename": filing.get("downloadFilename", "")
    } for filing in filings]
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(index_download_dir, f"index-{year}.csv"), index=False)
    return df

def process_year(api, year):
    """Process searching and downloading filings for a specific year."""
    search_params = build_search_params(year)
    results = perform_search(api, search_params)
    if results:
        exhibit_filings = filter_exhibit_filings(results)
        if exhibit_filings:
            asyncio.run(download_all_filings(exhibit_filings, year))
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
    for year in range(2011, 2025):
        print(f"Processing year: {year}")
        df = process_year(api, year)
        print("-" * 40)

if __name__ == '__main__':
    main()
