"""One-off generator for tests/fixtures/mini_ecommerce.csv.

Not part of the test suite itself (no test_ prefix, not collected by pytest) — kept
alongside the fixture as a record of how its values were derived, since several tests
in test_backend.py hand-verify exact numbers (sums, IQR outlier counts, missing %,
duplicate counts) against this file. Re-run only if the fixture's engineered scenarios
need to change; if so, update the corresponding hand-computed assertions too.

The fixture is deliberately *dirtier* than the real Global E-Commerce Sales CSV, which
has no nulls and no duplicate rows. Cleaning and profiling code still has to handle
those cases, so the scenarios below are engineered in on purpose:

  - row 7, 8   : missing total_revenue -> exercises the missing-% profiler
  - row 11     : "fashion" lowercase   -> exercises casing standardization
  - row 12     : revenue 5000, far outside the others -> exercises IQR outlier detection
  - row 13     : exact duplicate of row 1 -> exercises duplicate-row detection
  - "PayPal"   : present throughout, and must survive casing standardization uncorrupted
                 (it is why payment_method is excluded from categorical_casing_columns)

Unlike the previous dataset's fixture there is no Row.ID column to exclude from
duplicate detection — this dataset has no surrogate key, so row 13 is a true full-row
duplicate.
"""

import csv
from pathlib import Path

FIELDNAMES = [
    "Transaction Date", "Customer ID", "Region", "Product", "Category", "Price",
    "Quantity", "Discount (%)", "Total Revenue", "Payment Method",
]

# (date, customer, region, category, price, qty, discount_pct, total_revenue, payment)
ROWS = [
    ("2022-01-07", "CUST_00001", "Asia",          "Electronics", 100.00, 3, 0.0,  300.00, "Credit Card"),
    ("2022-01-10", "CUST_00002", "Asia",          "Electronics", 150.00, 2, 0.0,  300.00, "PayPal"),
    ("2022-01-15", "CUST_00003", "Asia",          "Books",       200.00, 4, 0.0,  800.00, "Cash"),
    ("2022-01-20", "CUST_00004", "Asia",          "Books",       120.00, 1, 30.0,  84.00, "Crypto"),
    ("2022-02-01", "CUST_00005", "Europe",        "Electronics",  90.00, 2, 0.0,  180.00, "Bank Transfer"),
    ("2022-02-05", "CUST_00006", "Europe",        "Electronics", 110.00, 3, 0.0,  330.00, "PayPal"),
    # Rows 7 and 8 have a missing Total Revenue (blank in the CSV).
    ("2022-02-10", "CUST_00007", "Europe",        "Books",        95.00, 2, 0.0,      "", "Credit Card"),
    ("2022-02-14", "CUST_00008", "Europe",        "Books",       105.00, 2, 0.0,      "", "Cash"),
    ("2022-03-01", "CUST_00009", "Asia",          "Fashion",     130.00, 3, 0.0,  390.00, "PayPal"),
    ("2022-03-05", "CUST_00010", "Europe",        "Fashion",      98.00, 2, 0.0,  196.00, "Crypto"),
    # Row 11 carries a deliberate casing anomaly in Category ("fashion").
    ("2022-03-10", "CUST_00011", "Asia",          "fashion",     140.00, 3, 0.0,  420.00, "Bank Transfer"),
    # Row 12 is the deliberate high outlier for IQR detection.
    ("2022-03-15", "CUST_00012", "Asia",          "Electronics", 500.00, 10, 0.0, 5000.00, "Credit Card"),
    # Row 13 is an exact duplicate of row 1 (every column identical).
    ("2022-01-07", "CUST_00001", "Asia",          "Electronics", 100.00, 3, 0.0,  300.00, "Credit Card"),
    ("2022-04-01", "CUST_00014", "North America", "Electronics", 115.00, 2, 0.0,  230.00, "PayPal"),
    ("2022-04-05", "CUST_00015", "North America", "Books",        99.00, 2, 0.0,  198.00, "Cash"),
    ("2022-04-10", "CUST_00016", "North America", "Fashion",     125.00, 3, 0.0,  375.00, "Credit Card"),
]


def build_rows() -> list[dict]:
    """Expand the compact ROWS tuples into full raw-schema dict rows."""
    out = []
    for (date, customer, region, category, price, qty, discount, revenue, payment) in ROWS:
        out.append({
            "Transaction Date": date,
            "Customer ID": customer,
            "Region": region,
            # Product is a synthetic identifier in the real dataset too; a single
            # value keeps the fixture's hand-computed numbers easy to follow.
            "Product": "Product_0001",
            "Category": category,
            "Price": price,
            "Quantity": qty,
            "Discount (%)": discount,
            "Total Revenue": revenue,
            "Payment Method": payment,
        })
    return out


def main() -> None:
    """Write the fixture CSV next to this script."""
    out_path = Path(__file__).resolve().parent / "mini_ecommerce.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
