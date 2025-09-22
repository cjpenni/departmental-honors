import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from boxsdk import Client, OAuth2

# ---- BOX CONFIG ----
# Replace with your Box developer token
BOX_DEVELOPER_TOKEN = 'ZUaqy8jt4mmuM72iJdpo2cezjDPUmhOY'
BOX_FOLDER_ID = '342315668958'  # e.g., '123456789'
LOCAL_OUTPUT_DIR = '/scratch/cjpenni/departmental-honors/data_pipeline/html_to_csv_output'
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
# ---------------------

def extract_and_remove_datetime(text):
    # Define a regex pattern for the specified date and time format
    datetime_pattern = r"(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4}),\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+(?P<meridian>[APap][Mm])\s+(?P<timezone>[A-Z]+)"
    match = re.search(datetime_pattern, text)
    if match:
        date = match.group('month') + " " + match.group('day') + ", " + match.group('year')
        time = match.group('hour') + ":" + match.group('minute') + ":" + match.group('second') + " " + match.group('meridian') + " " + match.group('timezone')
        return date, time
    else:
        return None, None

def read_html_file_fast(input_html):
    """Efficiently reads an HTML file and returns a parsed BeautifulSoup object."""
    with open(input_html, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'lxml')  # Use 'lxml' for speed

def process_html_to_csv(input_html, output_csv):
    soup = read_html_file_fast(input_html)

    # Extract all activity entries
    entries = soup.find_all('div', class_='outer-cell')

    data = []
    for entry in entries:
        # Extract source (Column 1)
        source_tag = entry.find('p', class_='mdl-typography--title')
        source = source_tag.get_text(strip=True) if source_tag else None

        # Extract link (Column 2)
        link_tag = entry.find('a')
        link = None
        if link_tag:
            match = re.search(r'q=(https://[^\&]+)', link_tag['href'])
            link = match.group(1) if match else link_tag['href']

        # Extract title (Column 3)
        title = None
        if link_tag:
            title = None if "https://" in link_tag.get_text(strip=True) else link_tag.get_text(strip=True)

        # Extract date and time (Column 4 & 5)
        text_content = entry.find('div', class_='content-cell').get_text(strip=True)
        date, time = extract_and_remove_datetime(text_content)

        # Append data to list
        data.append([source, link, title, date, time])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Search Site', 'Search Link', 'Search Title', 'Search Date', 'Search Time'])

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

def main():
    oauth2 = OAuth2(
        client_id=None,
        client_secret=None,
        access_token=BOX_DEVELOPER_TOKEN
    )
    client = Client(oauth2)
    folder = client.folder(folder_id=BOX_FOLDER_ID).get()
    items = folder.get_items()
    for item in items:
        if item.type == 'file' and item.name.lower().endswith('.html'):
            local_html_path = os.path.join(LOCAL_OUTPUT_DIR, item.name)
            with open(local_html_path, 'wb') as f:
                client.file(item.id).download_to(f)
            output_csv = os.path.join(
                LOCAL_OUTPUT_DIR,
                os.path.splitext(item.name)[0] + '.csv'
            )
            process_html_to_csv(local_html_path, output_csv)

if __name__ == "__main__":
    main()