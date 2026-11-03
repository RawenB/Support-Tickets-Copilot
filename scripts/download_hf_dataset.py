from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "huggingface"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset_name = "Tobi-Bueck/customer-support-tickets"
    print(f"Loading dataset: {dataset_name}")

    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()

    csv_path = OUT_DIR / "customer-support-tickets.csv"
    json_path = OUT_DIR / "customer-support-tickets.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"Rows saved: {len(df)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
