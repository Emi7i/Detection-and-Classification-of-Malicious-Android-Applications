import numpy as np
import pandas as pd
LEGIT_DATA = "data/real_legitimate_v1.csv"
MALWARE_DATA = "data/real_malware_v1.csv"

def main():
    legit_data = pd.read_csv(LEGIT_DATA)
    malware_data = pd.read_csv(MALWARE_DATA)

    print("-----------LEGIT DATASET-----------")
    print(legit_data.head())
    print("-----------MALWARE DATASET-----------")
    print(malware_data.head())

if __name__ == "__main__":
    main()