"""Render the deterministic synthetic ATXY page-oracle development datasets."""

from pathlib import Path

from molor.experimental.atxy_page_oracle_dataset import main


if __name__ == "__main__":
    raise SystemExit(main(default_output_directory=Path(__file__).resolve().parent / "dataset"))
