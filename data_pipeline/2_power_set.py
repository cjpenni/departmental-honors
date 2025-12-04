import pandas as pd
import os
from glob import glob

# ========== CONFIGURATION ==========
CSV_FOLDER = '/scratch/cjpenni/departmental-honors/data_pipeline/html_to_csv_output'
POWERSET_OUTPUT = '/scratch/cjpenni/departmental-honors/data_pipeline/powerset_all_year_col.csv'
POWERSET_200_OUTPUT = '/scratch/cjpenni/departmental-honors/data_pipeline/powerset_col200++.csv'
FILTERED_BY_YEAR_FOLDER = '/scratch/cjpenni/departmental-honors/data_pipeline/filtered_by_year'
POWERSET_BY_YEAR_OUTPUT = '/scratch/cjpenni/departmental-honors/data_pipeline/powerset_by_year/allActivity_2024.csv'

os.makedirs(FILTERED_BY_YEAR_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(POWERSET_BY_YEAR_OUTPUT), exist_ok=True)

# ========== READ ALL FILES ==========
csv_files = glob(os.path.join(CSV_FOLDER, "*.csv"))
df_names = []
search_title_dfs = []

for file in csv_files:
    output_filename = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    search_title_cols = [col for col in df.columns if "Search Title" in col]
    search_date_cols = [col for col in df.columns if "Search Date" in col]
    if search_title_cols and search_date_cols:
        search_title_df = df[search_title_cols + search_date_cols].copy()
        for date_col in search_date_cols:
            search_title_df[date_col] = pd.to_datetime(search_title_df[date_col], errors='coerce').dt.year
        for col in search_title_cols:
            search_title_df[col] = search_title_df[col].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        search_title_df.dropna(subset=search_title_cols, inplace=True)
        search_title_df.drop_duplicates(inplace=True)
        search_title_df.columns = [f"{output_filename}_{col}" for col in search_title_df.columns]
        search_title_dfs.append(search_title_df)
    df_names.append(output_filename)

if search_title_dfs:
    combined_search_titles_df = pd.concat(search_title_dfs, axis=1)
    combined_search_titles_df.to_csv(POWERSET_OUTPUT, index=False, encoding='utf-8')
    print(f"Data saved to {POWERSET_OUTPUT}")

# ========== DROPPING COLUMNS THAT HAS LESS THAN 200 DATA POINTS ==========
if search_title_dfs:
    column_lengths = combined_search_titles_df.count()
    columns_to_keep = column_lengths[column_lengths >= 200].index
    combined_search_titles_df = combined_search_titles_df[columns_to_keep]
    print(combined_search_titles_df.head())
    combined_search_titles_df.to_csv(POWERSET_200_OUTPUT, index=False, encoding='utf-8')
    print(f"Data saved to {POWERSET_200_OUTPUT}")

# ========== FILTERING SEARCH TITLES FOR GIVEN ACTIVITY BY YEAR ==========
def process_dataframe(df, df_name):
    search_title_cols = [col for col in df.columns if "Search Title" in col]
    search_date_cols = [col for col in df.columns if "Search Date" in col or "Date" in col]
    if not search_title_cols or not search_date_cols:
        print(f"Skipping {df_name}: No valid 'Search Title' or 'Search Date' column found.")
        return
    search_data = df[search_title_cols + search_date_cols].copy()
    date_col = search_date_cols[0]
    search_data[date_col] = pd.to_datetime(search_data[date_col], errors='coerce').dt.year
    search_data.dropna(subset=search_title_cols, inplace=True)
    search_data.drop_duplicates(inplace=True)
    unique_years = search_data[date_col].dropna().unique()
    for year in unique_years:
        year_data = search_data[search_data[date_col] == year][search_title_cols]
        output_filename = f"{df_name}_search_titles_{year}.csv"
        output_path = os.path.join(FILTERED_BY_YEAR_FOLDER, output_filename)
        year_data.to_csv(output_path, index=False)
        print(f"Saved: {output_filename} ({len(year_data)} rows)")

# Load and process each DataFrame
for file in csv_files:
    output_filename = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    if "Search Title" in df.columns:
        df.drop_duplicates(subset=["Search Title"], inplace=True)
        df.dropna(subset=["Search Title"], inplace=True)
    df.columns = [f"{output_filename}_{col}" for col in df.columns]
    globals()[output_filename] = df

for df_name in df_names:
    if df_name in globals():
        process_dataframe(globals()[df_name], df_name)
    else:
        print(f"Skipping {df_name}: DataFrame not found in memory.")

# ========== COMBINE ALL FILTERED CSVs FOR A YEAR ==========
YEAR = "2024"
year_folder = FILTERED_BY_YEAR_FOLDER  # All CSVs are in the same folder
csv_files_year = [f for f in glob(os.path.join(year_folder, f"*_{YEAR}.csv"))]
if csv_files_year:
    df_combined = pd.concat((pd.read_csv(file) for file in csv_files_year), ignore_index=True)
    df_sorted = df_combined.apply(lambda col: col.dropna().tolist() + [None] * col.isna().sum(), axis=0)
    df_sorted.to_csv(POWERSET_BY_YEAR_OUTPUT, index=False)
    print(f"Combined and saved all activity for {YEAR} to {POWERSET_BY_YEAR_OUTPUT}")
    non_nan_counts = df_combined.count()