"""Verify a reader's Heidelberg download against the shipped BCC split manifest."""

import argparse
import sys
from pathlib import Path

import pandas as pd

_DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "splits" / "heidelberg_bcc.csv"
_REQUIRED_COLUMNS = ("file", "case", "set", "label")
_VALID_SPLITS = ("Train", "Validation", "Test")
_SAMPLE_CAP = 20


def _crossing_cases(df):
    """Cases whose `case` appears under more than one `set`."""
    per_case = df.groupby("case")["set"].nunique()
    return sorted(per_case[per_case > 1].index.tolist(), key=str)


def _missing_files(df, data_root):
    """Manifest `file` entries that are not a real file under data_root."""
    base = data_root.resolve()
    missing = []
    for f in df["file"]:
        p = (data_root / f).resolve()
        if not (p.is_file() and p.is_relative_to(base)):
            missing.append(f)
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", required=True, type=Path,
        help="directory containing the extracted Heidelberg `data/` folder",
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    try:
        df = pd.read_csv(args.manifest, dtype={"case": str})
        missing_columns = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            raise KeyError(f"missing required column(s): {', '.join(missing_columns)}")
        required = df[list(_REQUIRED_COLUMNS)]
        is_blank = required.isnull() | required.apply(lambda c: c.astype(str).str.strip().eq(""))
        if is_blank.any().any():
            raise ValueError("null or blank value(s) in required column(s)")
        bad_splits = sorted(set(df["set"].astype(str)) - set(_VALID_SPLITS))
        if bad_splits:
            raise ValueError(f"invalid set value(s): {', '.join(bad_splits)}")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError, ValueError) as exc:
        print(f"FAIL: cannot read manifest {args.manifest}: {exc}")
        return 1

    crossing = _crossing_cases(df)
    if crossing:
        print(f"FAIL: {len(crossing)} case(s) cross splits (manifest must be patient-disjoint):")
        for case in crossing[:_SAMPLE_CAP]:
            print(f"  case {case}")
        return 1

    missing = _missing_files(df, args.data_root)
    if missing:
        print(f"FAIL: {len(missing)} manifest file(s) missing under {args.data_root}:")
        for f in missing[:_SAMPLE_CAP]:
            print(f"  {f}")
        return 1

    for split in _VALID_SPLITS:
        rows = df[df["set"] == split]
        n_bcc = int((rows["label"] == 1).sum())
        print(f"{split}: {len(rows)} tiles ({n_bcc} BCC, {len(rows) - n_bcc} non-BCC)")
    print(f"OK: {len(df)} tiles, {df['case'].nunique()} cases, patient-disjoint, all files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
