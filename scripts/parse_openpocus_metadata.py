"""
Parse 'Database for Github.xlsx' from kumarandre/OpenPOCUS.
Prints shape, columns, head, dtypes, and value_counts for label columns.

Usage:
  python scripts/parse_openpocus_metadata.py
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Run: pip install pandas openpyxl")

XLSX_PATH = Path("data/raw/openpocus_repo/Database for Github.xlsx")

LABEL_KEYWORDS = ['label', 'diagnosis', 'class', 'normal', 'finding',
                  'pathol', 'disease', 'category', 'type', 'status',
                  'result', 'interpret', 'grade', 'score', 'severity']


def main():
    if not XLSX_PATH.exists():
        sys.exit(f"Not found: {XLSX_PATH}")

    xl = pd.ExcelFile(str(XLSX_PATH))
    print(f"Sheets: {xl.sheet_names}")

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        print(f"\n{'='*60}")
        print(f"Sheet: {sheet}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nFirst 10 rows:\n{df.head(10).to_string()}")

        print("\n--- Label/diagnosis column value counts ---")
        found_any = False
        for col in df.columns:
            if any(k in col.lower() for k in LABEL_KEYWORDS):
                print(f"\n=== {col} ===")
                print(df[col].value_counts(dropna=False))
                found_any = True
        if not found_any:
            print("(no label columns detected — printing all string columns)")
            for col in df.columns:
                if df[col].dtype == object:
                    print(f"\n=== {col} ===")
                    print(df[col].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
