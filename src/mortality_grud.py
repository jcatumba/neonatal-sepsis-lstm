"""Fixed, development-only comparison of GRU-D and mortality references."""

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mortality_autoencoder_study import AutoencoderCandidate, candidate_scores, ranking_metrics
from .mortality_comparison import biomarker_features, make_logistic_classifier, patient_folds
from .mortality_panel_study import KEYS, OUTPUT_ROOT, PANELS, ROOT, load_panel_input
from .mortality_panel_study import timing_sensitivity, validate_panel_input
from .utils import save_json, set_seed


SEED = 42
EPOCHS = 20
HIDDEN = 16
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def recurrent_inputs(training: np.ndarray, held: np.ndarray) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Scale observed values using the fitting fold; build causal, bin-index gaps."""
    if not np.isfinite(training).any(axis=(0, 1)).all():
        raise ValueError("Every marker needs an observation in the fitting fold.")
    mean = np.nanmean(training, axis=(0, 1))
    spread = np.nanstd(training, axis=(0, 1))
    spread = np.where(spread > 0, spread, 1)

    def encode(raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = np.isfinite(raw)
        values = np.where(mask, (raw - mean) / spread, 0)
        gaps = np.zeros_like(values)
        for step in range(1, raw.shape[1]):
            gaps[:, step] = 1 + np.where(mask[:, step - 1], 0, gaps[:, step - 1])
        return torch.from_numpy(values).float(), torch.from_numpy(mask).float(), torch.from_numpy(gaps).float()

    return encode(training), encode(held)


class MortalityGRU(nn.Module):
    """GRU with optional GRU-D input and hidden-state decay on normalized inputs."""

    def __init__(self, markers: int, decay: bool):
        super().__init__()
        self.cell = nn.GRUCell(2 * markers, HIDDEN)
        self.classifier = nn.Linear(HIDDEN, 1)
        self.hidden_decay = nn.Linear(markers, HIDDEN) if decay else None
        self.input_rate = nn.Parameter(torch.full((markers,), 0.1)) if decay else None
        self.input_offset = nn.Parameter(torch.zeros(markers)) if decay else None
        if self.hidden_decay is not None:
            nn.init.constant_(self.hidden_decay.weight, 0.1 / markers)
            nn.init.zeros_(self.hidden_decay.bias)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        hidden = values.new_zeros((len(values), HIDDEN))
        last = torch.zeros_like(values[:, 0])
        input_factor = hidden_factor = None
        if self.hidden_decay is not None and self.input_rate is not None and self.input_offset is not None:
            input_factor = torch.exp(-torch.relu(gaps * self.input_rate + self.input_offset))
            hidden_factor = torch.exp(-torch.relu(self.hidden_decay(gaps)))
        for step in range(values.shape[1]):
            present = mask[:, step].bool()
            # Zero is the fitting mean in standardized units; decay contracts last toward it.
            substitute = last if input_factor is None else input_factor[:, step] * last
            current = torch.where(present, values[:, step], substitute)
            if hidden_factor is not None:
                hidden = hidden_factor[:, step] * hidden
            hidden = self.cell(torch.cat((current, mask[:, step]), dim=1), hidden)
            last = torch.where(present, values[:, step], last)
        return self.classifier(hidden).squeeze(-1)


def supervised_scores(training: np.ndarray, labels: np.ndarray, held: np.ndarray, *,
                      decay: bool, seed: int, epochs: int) -> np.ndarray:
    set_seed(seed)
    fitting, validation = recurrent_inputs(training, held)
    model = MortalityGRU(training.shape[-1], decay)
    loader = DataLoader(
        TensorDataset(*fitting, torch.as_tensor(labels, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for values, mask, gaps, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(values, mask, gaps), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised mortality loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(*validation).numpy()


def summarize(records: list[dict]) -> dict:
    summary = {}
    for model in records[0]["metrics"]:
        summary[model] = {}
        for metric in ("average_precision", "auroc"):
            values = np.array([record["metrics"][model][metric] for record in records])
            summary[model][metric] = {
                "fold_mean": float(values.mean()), "fold_sd": float(values.std(ddof=1)),
            }
    return summary


def compare_grud(arrays: dict[str, np.ndarray], *, epochs: int = EPOCHS, n_splits: int = 5) -> dict:
    validate_panel_input(arrays)
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    observed, labels, groups = (arrays[key] for key in ("observed", "labels", "groups"))
    early, eligible = timing_sensitivity(observed)
    folds = list(patient_folds(labels, groups, n_splits, SEED))
    results = {}
    for window, data, keep in (
        ("main_24h", observed, np.ones(len(labels), dtype=bool)),
        ("sensitivity_21_6h", early, eligible),
    ):
        panel = data[:, :, PANELS["glucose_ph"]]
        features = biomarker_features(panel)
        records = []
        coverage = np.zeros(len(labels), dtype=int)
        for fold, (train, held) in enumerate(folds, 1):
            train, held = train[keep[train]], held[keep[held]]
            if (np.intersect1d(groups[train], groups[held]).size
                    or set(labels[train]) != {0, 1} or set(labels[held]) != {0, 1}):
                raise ValueError("Folds must be patient-disjoint with both outcomes.")
            print(f"{window}: fold {fold}/{n_splits}", flush=True)
            seed = SEED + fold
            reference = make_logistic_classifier()
            reference.fit(features[train], labels[train])
            scores = {
                "autoencoder": candidate_scores(
                    panel[train], labels[train], panel[held], AutoencoderCandidate("glucose_ph"),
                    seed=seed, device="cpu",
                ),
                "logistic": reference.predict_proba(features[held])[:, 1],
            }
            for name, decay in (("gru_no_decay", False), ("gru_d", True)):
                scores[name] = supervised_scores(
                    panel[train], labels[train], panel[held], decay=decay, seed=seed, epochs=epochs,
                )
            metrics = {name: ranking_metrics(labels[held], score) for name, score in scores.items()}
            coverage[held] += 1
            records.append({
                "fold": fold, "fit_admissions": len(train), "held_admissions": len(held),
                "held_deaths": int(labels[held].sum()), "metrics": metrics,
            })
        if not np.array_equal(coverage, keep.astype(int)):
            raise RuntimeError("Each eligible admission must be evaluated exactly once.")
        contrasts = {}
        for reference_name in ("autoencoder", "logistic", "gru_no_decay"):
            contrasts[reference_name] = {}
            for metric in ("average_precision", "auroc"):
                delta = np.array([record["metrics"]["gru_d"][metric]
                                  - record["metrics"][reference_name][metric] for record in records])
                contrasts[reference_name][metric] = {
                    "mean_delta": float(delta.mean()), "folds_above_reference": int((delta > 0).sum()),
                }
        results[window] = {
            "admissions": int(keep.sum()), "patients": int(np.unique(groups[keep]).size),
            "deaths": int(labels[keep].sum()), "excluded_admissions": int((~keep).sum()),
            "excluded_deaths": int(labels[~keep].sum()), "folds": records,
            "summary": summarize(records), "gru_d_paired_contrasts": contrasts,
        }
    return {
        "test_accessed": False, "results": results,
        "protocol": {
            "endpoint": "same-hospitalization mortality at the 24h landmark",
            "panel": ["lactate", "total_bilirubin", "creatinine", "glucose", "pH"],
            "fold_seed": SEED, "folds": n_splits, "training_seed": "42 plus 1-based fold",
            "supervised_gru": {
                "epochs": epochs, "hidden_size": HIDDEN, "batch_size": BATCH_SIZE,
                "adam_lr": LEARNING_RATE, "gradient_clip": 1.0, "loss": "unweighted binary cross-entropy",
                "scaling": "observed fitting-fold mean and standard deviation only",
                "gaps": "past-only, per-marker, in 2.4-hour bin steps, not result timestamps",
                "control": "same GRUCell, initial shared weights, mask and training; past carry without decay",
            },
            "autoencoder": "unchanged glucose+pH: observed MSE/MAE, ten epochs, fold-local MinMax",
            "logistic": "fixed L2 C=1, 25 same-panel per-marker summaries, fold-local pipeline",
            "sensitivity": "mask last lab bin before all fitting; common earlier-baseline subset",
            "metrics": "fold mean AP primary; AUROC secondary; paired differences descriptive",
            "selection": "fixed configurations, no tuning, threshold, checkpoint selection or final fit",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=OUTPUT_ROOT / "panel-export/development.npz")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "grud-study")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs must be under ignored outputs/.")
    if destination.exists():
        raise FileExistsError("Choose a new GRU-D study output directory.")
    arrays = load_panel_input(args.input_file)
    torch.set_num_threads(2)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    log = destination / "training.log"
    with log.open("x") as stream, redirect_stdout(stream):
        log.chmod(0o600)
        report = compare_grud(arrays)
    digest = hashlib.sha256()
    for key in sorted(KEYS):
        digest.update(arrays[key].tobytes())
    report["development_arrays_sha256"] = digest.hexdigest()
    report["versions"] = {name: version(name) for name in ("numpy", "torch", "scikit-learn")}
    target = destination / "study.json"
    save_json(target, report)
    target.chmod(0o600)
    print(f"Development-only GRU-D comparison saved to {target}.")


if __name__ == "__main__":
    main()
