import pandas as pd

LEGIT_DATA = "data/raw/real_legitimate_v1.csv"
MALWARE_DATA = "data/raw/real_malware_v1.csv"

STATIC_HEADERS_FILE = "headers/static_headers.csv"
DYNAMIC_HEADERS_FILE = "headers/dynamic_headers.csv"
REMOVED_HEADERS_FILE = "headers/removed_headers.csv"

PATH_TO_SAVE_STATIC = "data/refined/static_data.csv"
PATH_TO_SAVE_DYNAMIC = "data/refined/dynamic_data.csv"

def refine_data():
    legit_data = pd.read_csv(LEGIT_DATA)
    malware_data = pd.read_csv(MALWARE_DATA)

    # Some of the families are incorectly identified, so I will just set all of them to Benign
    legit_data['MalFamily'] = 'Benign'

    print("-----------LEGIT DATASET-----------")
    print(legit_data.head())
    print("-----------MALWARE DATASET-----------")
    print(malware_data.head())

    combined = pd.concat([legit_data, malware_data], ignore_index=True)
    print("-----------COMBINED DATASET-----------")
    print(combined.head())

    # Remove empty rows and columns from the dataset (All rows contain at least some values, so this will do nothing)
    combined = combined.dropna(how='all', axis=0) # Rows
    combined = combined.dropna(how='all', axis=1) # Columns

    # Drop rows where MalFamily is NaN
    combined = combined.dropna(subset=['MalFamily'])

    # See which columns have missing values
    nan_columns = combined.isna().sum()
    nan_columns = nan_columns[nan_columns > 0]  # Only show columns with NA values
    print("Columns with NaN values and their counts:")
    print(nan_columns)

    print(f"Number of rows with NaN: {combined.isna().any(axis=1).sum()}")
    print(f"Number of columns with NaN: {combined.isna().any(axis=0).sum()}")

    # Here we get:
    # Columns with NaN values and their counts:
    # Activities                641
    # NrIntServices             641
    # NrIntServicesActions      641
    # NrIntActivities           641
    # NrIntActivitiesActions    641
    # NrIntReceivers            641
    # NrIntReceiversActions     641
    # TotalIntentFilters        642
    # NrServices                641
    # dtype: int64
    # Number of rows with NaN: 642

    # Meaning that there are 642 rows that have NaN in the same colums, so the best thing we can do is to just drop those rows. 
    # 642 rows makes up about 0.8% of all rows.
    combined = combined.dropna(axis=0, how='any') # Drop rows with at least one NaN value


    # Remove columns that have the same value for all rows
    original_cols = combined.columns.tolist()
    combined = combined.loc[:, combined.nunique() > 1]
    deleted_cols = [col for col in original_cols if col not in combined.columns]
    print(f"Removed {len(deleted_cols)} columns: {deleted_cols}")


    static_cols = pd.read_csv(STATIC_HEADERS_FILE).columns.tolist()
    removed_cols = pd.read_csv(REMOVED_HEADERS_FILE).columns.tolist()

    static_cols = [c for c in static_cols if c not in deleted_cols]

    # Remove columns we definatelly won't use
    combined = combined.drop(columns = removed_cols)
    print(combined.shape)

    # Save static data
    static_dataset = combined[static_cols]
    static_dataset.to_csv(PATH_TO_SAVE_STATIC, index=False)
    print(f"Static data saved to: {PATH_TO_SAVE_STATIC}")

    # Save dynamic data     
    dynamic_dataset = combined
    dynamic_dataset.to_csv(PATH_TO_SAVE_DYNAMIC, index=False)
    print(f"Dynamic data saved to: {PATH_TO_SAVE_DYNAMIC}")