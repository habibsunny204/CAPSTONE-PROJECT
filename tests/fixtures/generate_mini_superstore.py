"""One-off generator for tests/fixtures/mini_superstore.csv.

Not part of the test suite itself (no test_ prefix, not collected by pytest) — kept
alongside the fixture as a record of how its values were derived, since several tests
in test_backend.py hand-verify exact numbers (sums, IQR outlier counts, missing %,
duplicate counts) against this file. Re-run only if the fixture's engineered scenarios
need to change; if so, update the corresponding hand-computed assertions too.
"""

import csv
from pathlib import Path

FIELDNAMES = [
    "Category", "City", "Country", "Customer.ID", "Customer.Name", "Discount",
    "Market", "记录数", "Order.Date", "Order.ID", "Order.Priority", "Product.ID",
    "Product.Name", "Profit", "Quantity", "Region", "Row.ID", "Sales", "Segment",
    "Ship.Date", "Ship.Mode", "Shipping.Cost", "State", "Sub.Category", "Year",
    "Market2", "weeknum",
]

# (row_id, customer_id, customer_name, region, state, city, segment, sales, profit,
#  discount, quantity, order_date, ship_date, week_num)
ROWS = [
    (1, "CU-0001", "Alice Anderson", "West", "California", "Los Angeles", "Consumer", 100, 10, 0.0, 3, "2013-01-07", "2013-01-09", 2),
    (2, "CU-0002", "Bob Brown", "West", "California", "Los Angeles", "Consumer", 150, 15, 0.0, 2, "2013-01-10", "2013-01-12", 2),
    (3, "CU-0003", "Carol Clark", "West", "California", "Los Angeles", "Corporate", 200, 20, 0.0, 4, "2013-01-15", "2013-01-17", 3),
    (4, "CU-0004", "David Davis", "West", "California", "Los Angeles", "Corporate", 120, -8, 0.3, 1, "2013-01-20", "2013-01-22", 4),
    (5, "CU-0005", "Ellen Evans", "East", "New York", "New York City", "Consumer", 90, 9, 0.0, 2, "2013-02-01", "2013-02-03", 5),
    (6, "CU-0006", "Frank Foster", "East", "New York", "New York City", "Consumer", 110, 11, 0.0, 3, "2013-02-05", "2013-02-07", 6),
    (7, "CU-0007", "Grace Green", "East", "New York", "New York City", "Corporate", 95, "", 0.0, 2, "2013-02-10", "2013-02-12", 6),
    (8, "CU-0008", "Henry Hill", "East", "New York", "New York City", "Corporate", 105, "", 0.0, 2, "2013-02-14", "2013-02-16", 7),
    (9, "CU-0009", "Ivy Irwin", "West", "California", "Los Angeles", "Home Office", 130, 13, 0.0, 3, "2013-03-01", "2013-03-03", 9),
    (10, "CU-0010", "Jack Johnson", "East", "New York", "New York City", "Home Office", 98, 10, 0.0, 2, "2013-03-05", "2013-03-07", 10),
    (11, "CU-0011", "Karen King", "West", "California", "Los Angeles", "consumer", 140, 14, 0.0, 3, "2013-03-10", "2013-03-12", 10),
    (12, "CU-0012", "Leo Lewis", "West", "California", "Los Angeles", "Consumer", 5000, 500, 0.0, 10, "2013-03-15", "2013-03-17", 11),
    # Deliberate exact duplicate of row 1 (all columns identical except Row.ID/Order.ID).
    (13, "CU-0001", "Alice Anderson", "West", "California", "Los Angeles", "Consumer", 100, 10, 0.0, 3, "2013-01-07", "2013-01-09", 2),
    (14, "CU-0014", "Mona Miller", "West", "California", "Los Angeles", "Corporate", 115, 12, 0.0, 2, "2013-04-01", "2013-04-03", 14),
    (15, "CU-0015", "Nathan Nolan", "East", "New York", "New York City", "Consumer", 99, 9, 0.0, 2, "2013-04-05", "2013-04-07", 14),
    (16, "CU-0016", "Olivia Owen", "West", "California", "Los Angeles", "Home Office", 125, 13, 0.0, 3, "2013-04-10", "2013-04-12", 15),
]


def build_rows() -> list[dict]:
    """Expand the compact ROWS tuples into full raw-schema dict rows."""
    out = []
    for (row_id, cust_id, cust_name, region, state, city, segment, sales, profit,
         discount, qty, order_date, ship_date, week_num) in ROWS:
        # Row 13 intentionally reuses row 1's Order.ID too, so it is a true full-row
        # duplicate (excluding Row.ID) rather than merely having matching metrics.
        order_id = "US-2013-0001" if row_id == 13 else f"US-2013-{row_id:04d}"
        out.append({
            "Category": "Office Supplies",
            "City": city,
            "Country": "United States",
            "Customer.ID": cust_id,
            "Customer.Name": cust_name,
            "Discount": discount,
            "Market": "US",
            "记录数": 1,
            "Order.Date": f"{order_date} 00:00:00.000",
            "Order.ID": order_id,
            "Order.Priority": "Medium",
            "Product.ID": "OFF-PA-0001",
            "Product.Name": "Xerox Paper",
            "Profit": profit,
            "Quantity": qty,
            "Region": region,
            "Row.ID": row_id,
            "Sales": sales,
            "Segment": segment,
            "Ship.Date": f"{ship_date} 00:00:00.000",
            "Ship.Mode": "Standard Class",
            "Shipping.Cost": 5.0,
            "State": state,
            "Sub.Category": "Paper",
            "Year": 2013,
            "Market2": "North America",
            "weeknum": week_num,
        })
    return out


def main() -> None:
    """Write the fixture CSV next to this script."""
    out_path = Path(__file__).resolve().parent / "mini_superstore.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
