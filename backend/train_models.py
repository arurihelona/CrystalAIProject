from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from app.services.structure_features import FEATURE_NAMES, featurize_cif_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "materials_final.csv"
DEFAULT_CIF_DIR = PROJECT_ROOT / "data" / "cif"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "material_property_regressor.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train real material property regressors from local CIF data.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=70000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_training_frame(args.csv_path, args.cif_dir, args.max_samples)

    if dataset.empty:
        raise RuntimeError("No training samples were built. Check that the CSV and CIF folders are present.")

    feature_columns = FEATURE_NAMES
    target_columns = ["band_gap", "formation_energy"]

    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=args.random_state)

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
    )
    model.fit(train_df[feature_columns], train_df[target_columns])

    predictions = model.predict(test_df[feature_columns])
    band_gap_mae = mean_absolute_error(test_df["band_gap"], predictions[:, 0])
    formation_energy_mae = mean_absolute_error(test_df["formation_energy"], predictions[:, 1])
    band_gap_r2 = r2_score(test_df["band_gap"], predictions[:, 0])
    formation_energy_r2 = r2_score(test_df["formation_energy"], predictions[:, 1])

    artifact = {
        "model": model,
        "feature_names": feature_columns,
        "target_names": target_columns,
        "metrics": {
            "band_gap_mae": float(band_gap_mae),
            "formation_energy_mae": float(formation_energy_mae),
            "band_gap_r2": float(band_gap_r2),
            "formation_energy_r2": float(formation_energy_r2),
        },
        "target_ranges": {
            "band_gap": [float(dataset["band_gap"].min()), float(dataset["band_gap"].max())],
            "formation_energy": [
                float(dataset["formation_energy"].min()),
                float(dataset["formation_energy"].max()),
            ],
        },
        "sample_count": int(len(dataset)),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_path)

    print(f"Saved artifact to: {args.output_path}")
    print(f"Samples used: {len(dataset)}")
    print(f"Band gap MAE: {band_gap_mae:.4f} eV | R2: {band_gap_r2:.4f}")
    print(f"Formation energy MAE: {formation_energy_mae:.4f} eV/atom | R2: {formation_energy_r2:.4f}")


def build_training_frame(csv_path: Path, cif_dir: Path, max_samples: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["material_id", "band_gap", "formation_energy"]).copy()
    df = df.sample(frac=1.0, random_state=42)

    rows: list[dict[str, float | str]] = []

    for _, row in df.iterrows():
        material_id = str(row["material_id"]).strip()
        cif_path = cif_dir / f"{material_id}.cif"
        if not cif_path.exists():
            continue

        try:
            features = featurize_cif_path(cif_path)
        except Exception:  # noqa: BLE001
            continue

        rows.append(
            {
                "material_id": material_id,
                **dict(zip(FEATURE_NAMES, features, strict=True)),
                "band_gap": float(row["band_gap"]),
                "formation_energy": float(row["formation_energy"]),
            }
        )

        if len(rows) >= max_samples:
            break

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
