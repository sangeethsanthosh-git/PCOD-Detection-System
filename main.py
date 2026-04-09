"""Main entrypoint for fast, leakage-safe PCOS prediction using data1.csv."""

from __future__ import annotations

from pathlib import Path
import shutil

import joblib

from src.explainability import compute_global_shap_importance, select_explainer_model_name
from src.evaluate import (
    evaluate_models,
    save_confusion_matrix_plot,
    save_feature_importance_from_table,
    save_feature_importance_plot,
    save_model_metrics_csv,
    save_roc_curve_plot,
    select_best_model,
)
from src.feature_engineering import engineer_features, remove_highly_correlated_features
from src.feature_selection import select_top_features
from src.load_data import load_dataset
from src.preprocess import (
    TabularPreprocessor,
    apply_limited_smote,
    remove_duplicates,
    split_features_target,
    stratified_split,
)
from src.stacking import train_stacking_ensemble
from src.train_models import train_and_tune_models
from src.utils import ensure_directories


RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 20
CORRELATION_THRESHOLD = 0.85


def main() -> None:
    """Run complete fast pipeline from loading to model export."""
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset" / "data1.csv"
    models_dir = base_dir / "models"
    django_models_dir = base_dir / "pcos_project" / "models"
    results_dir = base_dir / "results"
    ensure_directories([models_dir, django_models_dir, results_dir])

    # 1) Load dataset.
    dataframe, load_info = load_dataset(dataset_path)
    print(f"Loaded: {load_info['dataset_path']}")
    print(f"Raw shape: {load_info['raw_shape']} -> final shape: {load_info['final_shape']}")
    print(f"Dropped identifier columns: {load_info['dropped_identifier_columns']}")

    # 2) Clean data.
    dataframe = remove_duplicates(dataframe)
    print(f"Shape after duplicate removal: {dataframe.shape}")

    # 3) Feature engineering.
    dataframe, engineered_flags = engineer_features(dataframe)
    print(f"Engineered features: {engineered_flags}")

    # 4) Train/test split.
    X, y = split_features_target(dataframe)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = stratified_split(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    # 5) Preprocess (train-only fit) and feature selection prep.
    preprocessor = TabularPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    scaler = preprocessor.state.scaler
    y_train = y_train_raw.astype(int).copy()
    y_test = y_test_raw.astype(int).copy()

    X_train, X_test, dropped_corr = remove_highly_correlated_features(
        X_train=X_train,
        X_test=X_test,
        threshold=CORRELATION_THRESHOLD,
    )
    print(f"Dropped correlated features (> {CORRELATION_THRESHOLD}): {len(dropped_corr)}")

    # 6) Class balancing with limited SMOTE.
    X_train_bal, y_train_bal, smote_info = apply_limited_smote(
        X_train=X_train,
        y_train=y_train,
        sampling_strategy=0.3,
        random_state=RANDOM_STATE,
    )
    print(f"SMOTE info: {smote_info}")
    print(f"Class distribution before SMOTE:\n{y_train.value_counts().sort_index()}")
    print(f"Class distribution after SMOTE:\n{y_train_bal.value_counts().sort_index()}")

    # 7) Feature selection.
    X_train_sel, X_test_sel, selected_features, feature_importance_df = select_top_features(
        X_train=X_train_bal,
        y_train=y_train_bal,
        X_test=X_test,
        top_k=TOP_K_FEATURES,
        random_state=RANDOM_STATE,
    )
    X_train_reference = X_train[selected_features].copy()
    feature_importance_df.to_csv(results_dir / "feature_scores.csv", index=False)
    print(f"Selected top {len(selected_features)} features.")

    # 8) Train base models.
    tuned_models, cv_summary = train_and_tune_models(
        X_train=X_train_sel,
        y_train=y_train_bal,
        random_state=RANDOM_STATE,
    )
    cv_summary.to_csv(results_dir / "cv_summary.csv", index=False)

    # 9) Train stacking ensemble.
    stacking_model = train_stacking_ensemble(
        tuned_models=tuned_models,
        X_train=X_train_sel,
        y_train=y_train_bal,
        random_state=RANDOM_STATE,
    )
    tuned_models["StackingEnsemble"] = stacking_model

    # 10) Evaluate, select best by F1, and save artifacts.
    results = evaluate_models(tuned_models, X_test_sel, y_test)
    metrics_df = save_model_metrics_csv(results, results_dir / "model_metrics.csv")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))

    best_model_name, best_metrics = select_best_model(results)
    best_model = tuned_models[best_model_name]
    print(f"\nSelected best model by F1: {best_model_name}")

    explainer_model_name = select_explainer_model_name(results)
    explainer_model = tuned_models[explainer_model_name]
    print(f"Selected SHAP explainer model: {explainer_model_name}")

    shap_importance_df = None
    try:
        shap_importance_df = compute_global_shap_importance(
            model=explainer_model,
            X=X_train_reference,
            y=y_train,
            max_samples=300,
            random_state=RANDOM_STATE,
        )
        shap_importance_df.to_csv(results_dir / "shap_feature_importance.csv", index=False)
    except ImportError:
        print("SHAP is not installed; falling back to model-native feature importance output.")

    joblib.dump(best_model, models_dir / "pcos_model.pkl")
    joblib.dump(stacking_model, models_dir / "stacking_model.pkl")
    joblib.dump(explainer_model, models_dir / "explainer_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(
        {
            "selected_features": selected_features,
            "dropped_correlated_features": dropped_corr,
            "best_model_name": best_model_name,
            "stacking_model_name": "StackingEnsemble",
            "explainer_model_name": explainer_model_name,
            "dataset_path": str(dataset_path),
        },
        models_dir / "model_metadata.pkl",
    )

    confusion_path = results_dir / "confusion_matrix.png"
    roc_path = results_dir / "roc_curve.png"
    importance_path = results_dir / "feature_importance.png"

    save_confusion_matrix_plot(best_metrics["confusion_matrix"], confusion_path, best_model_name)
    if best_metrics["roc"] is not None:
        save_roc_curve_plot(
            best_metrics["roc"]["fpr"],
            best_metrics["roc"]["tpr"],
            best_metrics["roc"]["auc"],
            roc_path,
            best_model_name,
        )

    importance_saved = False
    if shap_importance_df is not None:
        importance_saved = save_feature_importance_from_table(
            importance_df=shap_importance_df,
            output_path=importance_path,
            top_n=20,
            title=f"SHAP Feature Importance - {explainer_model_name}",
        )
    if not importance_saved:
        importance_saved = save_feature_importance_plot(
            model=best_model,
            feature_names=selected_features,
            output_path=importance_path,
            top_n=20,
        )
    if not importance_saved and "RandomForest" in tuned_models:
        save_feature_importance_plot(
            model=tuned_models["RandomForest"],
            feature_names=selected_features,
            output_path=importance_path,
            top_n=20,
        )

    for artifact_name in ("pcos_model.pkl", "stacking_model.pkl", "explainer_model.pkl", "scaler.pkl", "model_metadata.pkl"):
        shutil.copy2(models_dir / artifact_name, django_models_dir / artifact_name)

    print(f"Saved model: {models_dir / 'pcos_model.pkl'}")
    print(f"Saved stacking model: {models_dir / 'stacking_model.pkl'}")
    print(f"Saved explainer model: {models_dir / 'explainer_model.pkl'}")
    print(f"Saved scaler: {models_dir / 'scaler.pkl'}")
    print(f"Synced Django artifacts to: {django_models_dir}")
    print(f"Saved plots in: {results_dir}")


if __name__ == "__main__":
    main()
