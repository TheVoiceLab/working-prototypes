import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# =========================================================
# OUTPUT SCHEMA (FIXED & SAFE)
# =========================================================
OUTPUT_COLUMNS = [
    # time + traffic
    "time", "traffic",

    # viewers
    "viewers", "sub_multiplier", "imdb_factor",

    # show / network
    "show_id", "show_name", "network",

    # episode
    "episode_id", "episode_name", "season", "number",

    # IMDb
    "tconst", "averageRating", "numVotes",
    "titleType", "primaryTitle",
    "startYear", "endYear",
    "runtimeMinutes", "genres",
]

# =========================================================
# 1. LOAD MERGED TVMAZE + IMDB DATA
# =========================================================
def load_tvmaze_imdb(path="tvmaze_with_imdb_popularity.csv"):
    df = pd.read_csv(path, low_memory=False)

    # normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # airtime → hour
    def to_hour(x):
        try:
            return int(str(x).split(":")[0])
        except:
            return np.random.choice([19, 20, 21, 22])

    df["hour"] = df["airtime"].apply(to_hour)

    # numeric cleanup
    for c in ["averageRating", "numVotes", "runtimeMinutes"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================================================
# 2. MARKET TREND MODEL
# =========================================================
def get_subscriber_multiplier(date):
    annual_decline = {
        2020: -0.064,
        2021: -0.119,
        2022: -0.063,
        2023: -0.096,
        2024: -0.099,
    }
    m = 1.0
    for y in range(2020, date.year):
        m *= (1 + annual_decline.get(y, -0.05))
    return m


def imdb_popularity_factor(rating, votes):
    if pd.isna(rating) or pd.isna(votes):
        return 1.0
    r = max(0.5, min(1.5, (rating - 5) / 5 * 0.5 + 1))
    v = 0.5 + np.log10(votes + 10) / 6
    return r * v


def base_hourly_curve(h):
    prime = np.exp(-0.5 * ((h - 21) / 2) ** 2) * 1.5
    off   = np.exp(-0.5 * ((h - 14) / 6) ** 2) * 0.4
    return prime + off

# =========================================================
# 3. HOURLY VIEWERSHIP SIMULATION
# =========================================================
def simulate_viewership(start_date, days, df):
    rows = []
    curve = base_hourly_curve(np.arange(24))
    max_viewers = 5_000_000

    for d in range(days):
        date = start_date + timedelta(days=d)
        sub_mult = get_subscriber_multiplier(date)

        day_df = df[df["date"].dt.date == date.date()]

        for h in range(24):
            airing = day_df[day_df["hour"] == h]

            if not airing.empty:
                best = airing.sort_values("numVotes", ascending=False).iloc[0]
                pop = imdb_popularity_factor(best["averageRating"], best["numVotes"])
            else:
                best = None
                pop = 1.0

            viewers = int((curve[h] / curve.max()) * max_viewers * sub_mult * pop)

            rows.append({
                "datetime": date + timedelta(hours=h),
                "viewers": viewers,
                "sub_multiplier": sub_mult,
                "imdb_factor": pop,

                "show_id": best.get("show_id") if best is not None else None,
                "show_name": best.get("show_name") if best is not None else None,
                "network": best.get("network") if best is not None else None,

                "episode_id": best.get("episode_id") if best is not None else None,
                "episode_name": best.get("episode_name") if best is not None else None,
                "season": best.get("season") if best is not None else None,
                "number": best.get("number") if best is not None else None,

                "tconst": best.get("tconst") if best is not None else None,
                "averageRating": best.get("averageRating") if best is not None else None,
                "numVotes": best.get("numVotes") if best is not None else None,
                "titleType": best.get("titleType") if best is not None else None,
                "primaryTitle": best.get("primaryTitle") if best is not None else None,
                "startYear": best.get("startYear") if best is not None else None,
                "endYear": best.get("endYear") if best is not None else None,
                "runtimeMinutes": best.get("runtimeMinutes") if best is not None else None,
                "genres": best.get("genres") if best is not None else None,
            })

    return pd.DataFrame(rows)

# =========================================================
# 4. MINUTE-LEVEL TRAFFIC (SAFE CSV)
# =========================================================
def simulate_remote_usage(df_h):
    rows = []

    for _, r in df_h.iterrows():
        base = r["viewers"] * 0.0015
        spike = np.random.uniform(3, 6) * r["imdb_factor"]

        for m in range(60):
            mult = spike if (m < 5 and pd.notna(r["show_name"])) else 1.0
            traffic = int(base * mult * np.random.uniform(0.8, 1.2))

            rows.append({
                "time": r["datetime"] + timedelta(minutes=m),
                "traffic": traffic,
                "viewers": r["viewers"],
                "sub_multiplier": r["sub_multiplier"],
                "imdb_factor": r["imdb_factor"],

                "show_id": r["show_id"],
                "show_name": r["show_name"],
                "network": r["network"],
                "episode_id": r["episode_id"],
                "episode_name": r["episode_name"],
                "season": r["season"],
                "number": r["number"],

                "tconst": r["tconst"],
                "averageRating": r["averageRating"],
                "numVotes": r["numVotes"],
                "titleType": r["titleType"],
                "primaryTitle": r["primaryTitle"],
                "startYear": r["startYear"],
                "endYear": r["endYear"],
                "runtimeMinutes": r["runtimeMinutes"],
                "genres": r["genres"],
            })

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

# =========================================================
# 5. BATCH GENERATION (CSV SAFE)
# =========================================================
def batch_generate(start, end, df, out):
    cur = start
    first = True

    if os.path.exists(out):
        os.remove(out)

    while cur < end:
        nxt = min(cur + timedelta(days=30), end)
        days = (nxt - cur).days

        df_h = simulate_viewership(cur, days, df)
        df_m = simulate_remote_usage(df_h)

        df_m.to_csv(
            out,
            mode="w" if first else "a",
            header=first,
            index=False
        )

        first = False
        cur = nxt
        print(f"Saved through {cur.date()}")

# =========================================================
# 6. EXECUTION
# =========================================================
if __name__ == "__main__":
    merged = load_tvmaze_imdb()

    start = datetime(2021, 12, 1)
    end   = datetime(2025, 12, 21)
    out   = "remote_traffic_4year_full.csv"

    batch_generate(start, end, merged, out)

    # preview (SAFE)
    df_prev = pd.read_csv(out, low_memory=False).tail(2880)
    df_prev["time"] = pd.to_datetime(df_prev["time"])

    plt.figure(figsize=(14, 6))
    plt.plot(df_prev["time"], df_prev["traffic"])
    plt.title("Last 48 Hours Traffic")
    plt.grid(True)
    plt.show()
