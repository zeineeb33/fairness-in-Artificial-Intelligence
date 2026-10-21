"""Reproduce the notebook's fairness study on held-out patients and compare mitigations."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fairaudit import __version__
from fairaudit.audit import audit
from fairaudit.casestudy.config import StudyConfig
from fairaudit.casestudy.data import (
    ATTRIBUTE_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_metadata,
    patient_split,
)
from fairaudit.casestudy.results import (
    MODEL_CARD_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    RESULTS_FILE,
    ComparisonRow,
    ExperimentResult,
    GroupBaseRate,
    SplitSummary,
    StudyResults,
)
from fairaudit.mitigation import GroupThresholdOptimizer, reweighing_weights

logger = logging.getLogger(__name__)

Scores = NDArray[np.float64]


@dataclass
class StudyRun:
    results: StudyResults
    predictions: pd.DataFrame  # test-set labels, sensitive attributes and every model's output


def _pipeline(
    features: Sequence[str], model: object, pca_components: int | None = None
) -> Pipeline:
    numeric = [f for f in NUMERIC_FEATURES if f in features]
    categorical = [f for f in CATEGORICAL_FEATURES if f in features]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    steps: list[tuple[str, object]] = [
        (
            "features",
            ColumnTransformer([("num", StandardScaler(), numeric), ("cat", encoder, categorical)]),
        )
    ]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))
    steps.append(("model", model))
    return Pipeline(steps)


def _logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _fit_scores(
    pipe: Pipeline,
    train: pd.DataFrame,
    *others: pd.DataFrame,
    sample_weight: NDArray[np.float64] | None = None,
) -> list[Scores]:
    fit_params = {} if sample_weight is None else {"model__sample_weight": sample_weight}
    pipe.fit(train, train["label"], **fit_params)
    return [pipe.predict_proba(df)[:, 1] for df in others]


def run_study(config: StudyConfig) -> StudyRun:
    started = time.perf_counter()
    df, cleaning = load_metadata(config.data_path, config.max_age)
    splits = patient_split(df, config.test_size, config.val_size, config.seed)
    train, val, test = splits["train"], splits["val"], splits["test"]
    target = config.target_attribute
    logger.info(
        "Loaded %d images / %d patients; split train=%d val=%d test=%d",
        cleaning.n_rows,
        cleaning.n_patients,
        len(train),
        len(val),
        len(test),
    )

    y_test = test["label"].to_numpy()
    sensitive: dict[str, NDArray[np.generic]] = {
        a: test[a].to_numpy() for a in config.audit_attributes
    }
    predictions = test[["label", *config.audit_attributes]].copy()
    experiments: list[ExperimentResult] = []

    def record(
        name: str,
        title: str,
        description: str,
        stage: str,
        scores: Scores,
        preds: NDArray[np.int8] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        report = audit(
            y_test,
            preds,
            sensitive=sensitive,
            y_score=scores,
            threshold=config.threshold,
            policy=config.policy,
        )
        if preds is None:  # post-processing experiments reuse the baseline scores
            predictions[f"score_{name}"] = np.round(scores, 4)
        predictions[f"pred_{name}"] = (
            preds if preds is not None else (scores >= config.threshold).astype(np.int8)
        )
        experiments.append(
            ExperimentResult(
                name=name,
                title=title,
                description=description,
                stage=stage,
                report=report,
                details=details or {},
            )
        )
        attr = report.attribute(target)
        logger.info(
            "%-16s accuracy=%.4f DI(%s)=%s EOdds=%s",
            name,
            report.overall.accuracy or float("nan"),
            target,
            attr.disparate_impact_ratio,
            attr.equalized_odds_difference,
        )

    # 1. Notebook replication: shallow decision tree on all metadata features.
    tree = _pipeline(
        FEATURES, DecisionTreeClassifier(max_depth=config.tree_max_depth, random_state=config.seed)
    )
    (tree_test,) = _fit_scores(tree, train, test)
    record(
        "decision_tree",
        f"Decision tree (depth {config.tree_max_depth})",
        "Baseline replicating the notebook: all metadata features, sensitive attributes included.",
        "none",
        tree_test,
    )

    # 2. Baseline used by every mitigation below.
    (base_val, base_test) = _fit_scores(_pipeline(FEATURES, _logreg()), train, val, test)
    record(
        "logreg",
        "Logistic regression",
        "Baseline for the mitigations: all metadata features, sensitive attributes included.",
        "none",
        base_test,
    )

    # 3. Pre-processing: Kamiran-Calders reweighing on the target attribute.
    weights = reweighing_weights(train["label"], train[target])
    (rw_test,) = _fit_scores(_pipeline(FEATURES, _logreg()), train, test, sample_weight=weights)
    cell_weights = {
        f"{g}|y={y}": round(float(w), 4)
        for g, y, w in zip(train[target], train["label"], weights, strict=True)
    }
    record(
        "reweighing",
        "Reweighing (Kamiran-Calders)",
        f"Training samples weighted so that the label is independent of '{target}'.",
        "pre-processing",
        rw_test,
        details={"weights": dict(sorted(cell_weights.items()))},
    )

    # 4. Fairness through unawareness: drop every feature encoding the target attribute.
    removed = ATTRIBUTE_FEATURES[target]
    kept = [f for f in FEATURES if f not in removed]
    (un_test,) = _fit_scores(_pipeline(kept, _logreg()), train, test)
    record(
        "unawareness",
        "Sensitive attribute removed",
        f"Features {list(removed)} dropped; correlated proxies may remain.",
        "pre-processing",
        un_test,
        details={"removed_features": list(removed)},
    )

    # 5. PCA representation, as in the notebook (dimensionality reduction, not a fairness method).
    pca_pipe = _pipeline(FEATURES, _logreg(), pca_components=config.pca_components)
    (pca_test,) = _fit_scores(pca_pipe, train, test)
    explained = float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum())
    record(
        "pca",
        f"PCA representation ({config.pca_components} components)",
        "Logistic regression on a low-dimensional PCA projection of all features.",
        "pre-processing",
        pca_test,
        details={"explained_variance_ratio": round(explained, 4)},
    )

    # 6-7. Post-processing: per-group thresholds on the baseline scores, tuned on validation.
    for constraint, slug, label, rate in (
        ("demographic_parity", "thresholds_dp", "demographic parity", "selection rates"),
        ("equal_opportunity", "thresholds_eo", "equal opportunity", "true-positive rates"),
    ):
        optimizer = GroupThresholdOptimizer(constraint).fit(  # type: ignore[arg-type]
            base_val, val["label"], val[target]
        )
        record(
            slug,
            f"Group thresholds ({label})",
            f"Baseline scores with one threshold per '{target}' group, chosen on the validation "
            f"patients to equalise {rate} ({label}) while maximising accuracy.",
            "post-processing",
            base_test,
            preds=optimizer.predict(base_test, test[target]),
            details={"thresholds": {k: round(v, 4) for k, v in optimizer.thresholds_.items()}},
        )

    results = StudyResults(
        fairaudit_version=__version__,
        config=config,
        dataset=cleaning,
        splits={
            name: SplitSummary(
                n_rows=len(part),
                n_patients=int(part["patient_id"].nunique()),
                positive_rate=round(float(part["label"].mean()), 6),
            )
            for name, part in splits.items()
        },
        label_base_rates={a: _base_rates(df, a) for a in config.audit_attributes},
        experiments=experiments,
        comparison=[
            ComparisonRow(
                experiment=e.name,
                attribute=a.attribute,
                accuracy=e.report.overall.accuracy,
                balanced_accuracy=e.report.overall.balanced_accuracy,
                roc_auc=e.report.overall.roc_auc,
                disparate_impact_ratio=a.disparate_impact_ratio,
                demographic_parity_difference=a.demographic_parity_difference,
                equal_opportunity_difference=a.equal_opportunity_difference,
                equalized_odds_difference=a.equalized_odds_difference,
            )
            for e in experiments
            for a in e.report.attributes
        ],
    )
    logger.info("Study finished in %.1fs", time.perf_counter() - started)
    return StudyRun(results=results, predictions=predictions)


def _base_rates(df: pd.DataFrame, attribute: str) -> list[GroupBaseRate]:
    grouped = df.groupby(attribute)["label"].agg(["size", "mean"])
    return [
        GroupBaseRate(
            group=str(group),
            n=int(row["size"]),
            share=round(row["size"] / len(df), 6),
            base_rate=round(float(row["mean"]), 6),
        )
        for group, row in grouped.iterrows()
    ]


def write_outputs(run: StudyRun, output_dir: Path) -> list[Path]:
    """Write results JSON, Markdown report, model card and test predictions."""
    from fairaudit.render import render_model_card, render_study_markdown

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        RESULTS_FILE: run.results.model_dump_json(indent=2),
        REPORT_FILE: render_study_markdown(run.results),
        MODEL_CARD_FILE: render_model_card(run.results),
    }
    paths = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    predictions_path = output_dir / PREDICTIONS_FILE
    run.predictions.to_csv(predictions_path, index=False)
    paths.append(predictions_path)
    return paths
