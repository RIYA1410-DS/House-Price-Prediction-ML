
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "eda_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    df = pd.read_csv(DATA_PATH)

    # Exclude the identifier from analysis
    df = df.drop(columns=["Property_ID"])

    print("\n========== CATEGORY DISTRIBUTIONS ==========")
    print("\nLocation counts:")
    print(df["Location"].value_counts())

    print("\nProperty type counts:")
    print(df["Property_Type"].value_counts())

    print("\n========== CORRELATION WITH PRICE ==========")
    numeric_cols = df.select_dtypes(include="number")
    print(
        numeric_cols.corr()["Price"]
        .sort_values(ascending=False)
    )

    # Plot 1: Price distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Price"], bins=20, edgecolor="black")
    plt.title("Distribution of House Prices")
    plt.xlabel("Price (₹)")
    plt.ylabel("Number of Properties")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png")
    plt.close()

    # Plot 2: Area vs Price
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Area"], df["Price"], alpha=0.7)
    plt.title("Property Area vs Price")
    plt.xlabel("Area (sq. ft.)")
    plt.ylabel("Price (₹)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "area_vs_price.png")
    plt.close()

    # Plot 3: Average price by location
    location_prices = (
        df.groupby("Location")["Price"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    location_prices.plot(kind="bar")
    plt.title("Average House Price by Location")
    plt.xlabel("Location")
    plt.ylabel("Average Price (₹)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_by_location.png")
    plt.close()

    print("\nEDA completed successfully.")
    print(f"Charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
