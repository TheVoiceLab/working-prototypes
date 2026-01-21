import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASE_URL = "https://api.tvmaze.com/schedule"
COUNTRY = "US"
YEARS_BACK = 4
REQUEST_DELAY = 0.25   # seconds between calls (TVMaze is friendly but be polite)
MAX_RETRIES = 3

# ---------------------------------------------------------
# API CALL WITH RETRIES
# ---------------------------------------------------------

def fetch_schedule(date_str):
    url = f"{BASE_URL}?country={COUNTRY}&date={date_str}"

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"Warning: {date_str} returned status {r.status_code}")
        except Exception as e:
            print(f"Error fetching {date_str}: {e}")

        time.sleep(1)  # wait before retry

    return []  # return empty if failed

# ---------------------------------------------------------
# MAIN DOWNLOAD LOOP
# ---------------------------------------------------------

def download_tvmaze_history():
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=YEARS_BACK * 365)

    print(f"Downloading TV schedules from {start_date} to {end_date}...")

    all_rows = []

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Fetching {date_str}...")

        data = fetch_schedule(date_str)

        for entry in data:
            show = entry.get("show", {})
            episode = entry.get("episode", {})

            all_rows.append({
                "date": date_str,
                "airtime": entry.get("airtime"),
                "runtime": entry.get("runtime"),
                "show_id": show.get("id"),
                "show_name": show.get("name"),
                "network": (show.get("network") or {}).get("name"),
                "episode_id": episode.get("id"),
                "episode_name": episode.get("name"),
                "season": episode.get("season"),
                "number": episode.get("number"),
                "summary": episode.get("summary"),
            })

        time.sleep(REQUEST_DELAY)
        current += timedelta(days=1)

    df = pd.DataFrame(all_rows)
    df.to_csv("tvmaze_schedule_4years.csv", index=False)

    print("✅ Done! Saved to tvmaze_schedule_4years.csv")
    return df

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    df = download_tvmaze_history()
    print(df.head())