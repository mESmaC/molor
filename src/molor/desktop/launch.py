"""Console entry point that keeps PySide6 an optional dependency."""

from __future__ import annotations


def main() -> int:
    try:
        from molor.desktop.app import run
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise SystemExit(
                'PySide6 is not installed. Run: pip install -e ".[hf,desktop]"'
            ) from exc
        raise
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
