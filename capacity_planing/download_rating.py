import pandas as pd
import requests
import gzip
import shutil
import os
from datetime import datetime, timedelta
from rapidfuzz import fuzz, process

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

IMDB_URLS = {
    "basics": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "ratings": "https://datasets.imdbws.com/title.ratings.tsv.gz",
}

YEARS_BACK = 4
COUNTRY = "US"

# ---------------------------------------------------------
# DOWNLOAD IMDB DATASETS
# ---------------------------------------------------------

def download_imdb():
    os.makedirs("imdb_data", exist_ok=True)

    for name, url in IMDB_URLS.items():
        print(f"Downloading IMDb {name}...")
        gz_path = f"imdb_data/{name}.tsv.gz"
        tsv_path = f"imdb_data/{name}.tsv"

        r = requests.get(url, stream=True)
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        # Unzip
        with gzip.open(gz_path, "rb") as f_in:
            with open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("✅ IMDb datasets downloaded and extracted.")

# ---------------------------------------------------------
# LOAD IMDB DATA
# ---------------------------------------------------------

def load_imdb():
    basics = pd.read_csv(
        "imdb_data/basics.tsv",
        sep="\t",
        na_values="\\N",
        dtype=str
    )

    ratings = pd.read_csv(
        "imdb_data/ratings.tsv",
        sep="\t",
        na_values="\\N",
        dtype={"tconst": str, "averageRating": float, "numVotes": int}
    )

    # Filter to TV series only
    basics = basics[basics["titleType"].isin(["tvSeries", "tvMiniSeries"])]

    # Merge basics + ratings
    imdb = basics.merge(ratings, on="tconst", how="left")

    # Clean
    imdb["primaryTitle"] = imdb["primaryTitle"].str.lower()
    imdb["startYear"] = pd.to_numeric(imdb["startYear"], errors="coerce")

    return imdb

# ---------------------------------------------------------
# LOAD TVMAZE SCHEDULE (PREVIOUSLY DOWNLOADED)
# ---------------------------------------------------------

def load_tvmaze():
    df = pd.read_csv("tvmaze_schedule_4years.csv")
    df["show_name_clean"] = df["show_name"].str.lower()
    return df

# ---------------------------------------------------------
# FUZZY MATCHING
# ---------------------------------------------------------

def fuzzy_match_tvmaze_to_imdb(tvmaze_df, imdb_df):
    imdb_titles = imdb_df["primaryTitle"].tolist()

    matches = []
    for show in tvmaze_df["show_name_clean"].unique():
        match, score, idx = process.extractOne(
            show,
            imdb_titles,
            scorer=fuzz.token_sort_ratio
        )
        matches.append((show, match, score))

    match_df = pd.DataFrame(matches, columns=["tvmaze_name", "imdb_name", "score"])

    # Keep only strong matches
    match_df = match_df[match_df["score"] >= 80]

    # Merge IMDb metadata
    merged = tvmaze_df.merge(
        match_df,
        left_on="show_name_clean",
        right_on="tvmaze_name",
        how="left"
    )

    merged = merged.merge(
        imdb_df,
        left_on="imdb_name",
        right_on="primaryTitle",
        how="left"
    )

    return merged

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Step 1: Downloading IMDb datasets...")
    download_imdb()

    print("Step 2: Loading IMDb data...")
    imdb_df = load_imdb()

    print("Step 3: Loading TVMaze schedule...")
    tvmaze_df = load_tvmaze()

    print("Step 4: Fuzzy matching TVMaze → IMDb...")
    merged = fuzzy_match_tvmaze_to_imdb(tvmaze_df, imdb_df)

    merged.to_csv("tvmaze_with_imdb_popularity.csv", index=False)

    print("✅ Done! Saved merged dataset to tvmaze_with_imdb_popularity.csv")
    print(merged.head())