import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algo.dbscan_v3 import compute_dbscan_with_dbscan_in_big_clusters


QUARTERS = {
    "Q1_JAN_MAR": [1, 2, 3],
    "Q2_APR_JUN": [4, 5, 6],
    "Q3_JUL_SEP": [7, 8, 9],
    "Q4_OCT_DEC": [10, 11, 12],
}

YEAR_GROUPS = (
    (2008, 2011),
    (2012, 2015),
    (2016, 2019),
)


def _run_dbscan_v3_on_subset(
    df_subset: pd.DataFrame,
    dbscan_params: Dict,
) -> pd.DataFrame:
    """Run the dbscan_v3 pipeline on a subset and return subset + cluster columns."""
    work = df_subset.dropna(subset=["lat", "long"]).copy()
    if work.empty:
        return work

    _, _, _, labels = compute_dbscan_with_dbscan_in_big_clusters(work, **dbscan_params)
    work["cluster_local"] = labels
    return work


def _period_stats(df_clustered: pd.DataFrame, period: str) -> Dict:
    if df_clustered.empty:
        return {
            "period": period,
            "n_points": 0,
            "n_clusters": 0,
            "noise_ratio": 0.0,
        }

    labels = df_clustered["cluster_local"].to_numpy()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float((labels == -1).mean())

    return {
        "period": period,
        "n_points": int(len(df_clustered)),
        "n_clusters": int(n_clusters),
        "noise_ratio": noise_ratio,
    }


def temporal_dbscan_analysis(
    df: pd.DataFrame,
    year_col: str = "date_taken_year",
    month_col: str = "date_taken_month",
    dbscan_params: Dict = None,
    min_points: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Clustering dbscan_v3 par annee.
    2) Clustering dbscan_v3 par trimestre toutes annees confondues.
    Retourne:
      - un dataframe avec labels de clusters par periode
      - un dataframe de stats par periode
    """
    if dbscan_params is None:
        dbscan_params = {
            "eps": 25,
            "min_samples": 25,
            "eps2": 15,
            "min_samples2": 25,
            "max_cluster_size": 1000,
        }

    required_cols = {"lat", "long", year_col, month_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    results: List[pd.DataFrame] = []
    stats: List[Dict] = []

    for start_year, end_year in YEAR_GROUPS:
        group_label = f"{start_year}_{end_year}"
        group_df = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)].copy()
        if len(group_df) < min_points:
            continue

        # Clustering par groupe d'annees
        clustered_year = _run_dbscan_v3_on_subset(group_df, dbscan_params)
        clustered_year["analysis_level"] = "year_group"
        clustered_year["period"] = group_label
        clustered_year["cluster_id"] = clustered_year["cluster_local"].apply(
            lambda c: f"{group_label}_C{c}" if c != -1 else f"{group_label}_NOISE"
        )
        results.append(clustered_year)
        stats.append(_period_stats(clustered_year, group_label))

    # Clustering saisonnier global (trimestres sur toutes les annees)
    for quarter_name, months in QUARTERS.items():
        quarter_df = df[df[month_col].isin(months)].copy()
        if len(quarter_df) < min_points:
            continue

        period = quarter_name
        clustered_quarter = _run_dbscan_v3_on_subset(quarter_df, dbscan_params)
        clustered_quarter["analysis_level"] = "quarter_global"
        clustered_quarter["period"] = period
        clustered_quarter["cluster_id"] = clustered_quarter["cluster_local"].apply(
            lambda c: f"{period}_C{c}" if c != -1 else f"{period}_NOISE"
        )
        results.append(clustered_quarter)
        stats.append(_period_stats(clustered_quarter, period))

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    all_clusters = pd.concat(results, ignore_index=True)
    stats_df = pd.DataFrame(stats).sort_values("period").reset_index(drop=True)
    return all_clusters, stats_df


def plot_temporal_cluster_comparison(
    clustered_df: pd.DataFrame,
    selected_year_groups: Tuple[str, ...] = ("2008_2011", "2012_2015", "2016_2019"),
) -> None:
    """Affiche uniquement les cartes de clusters demandees."""
    if clustered_df.empty:
        print("Aucune donnee a afficher.")
        return

    # 1) Cartes des clusters par groupe d'annees
    fig, axes = plt.subplots(
        1,
        len(selected_year_groups),
        figsize=(5 * len(selected_year_groups), 4),
        squeeze=False,
    )
    for idx, year_group in enumerate(selected_year_groups):
        ax = axes[0][idx]
        subset = clustered_df[
            (clustered_df["analysis_level"] == "year_group")
            & (clustered_df["period"] == year_group)
            & (clustered_df["cluster_local"] != -1)
        ]
        if subset.empty:
            ax.set_title(f"{year_group} (vide)")
            ax.axis("off")
            continue
        ax.scatter(
            subset["long"],
            subset["lat"],
            c=subset["cluster_local"],
            s=6,
            cmap="tab20",
            alpha=0.7,
        )
        ax.set_title(f"Clusters {year_group} (sans bruit)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.suptitle("Comparaison spatiale des clusters par groupe d'annees", y=1.02)
    plt.tight_layout()
    plt.show()

    # 2) Cartes des clusters par trimestre, toutes annees confondues
    quarter_clustered = clustered_df[
        (clustered_df["analysis_level"] == "quarter_global")
        & (clustered_df["cluster_local"] != -1)
    ].copy()
    if quarter_clustered.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    q_list = ["Q1_JAN_MAR", "Q2_APR_JUN", "Q3_JUL_SEP", "Q4_OCT_DEC"]
    quarter_titles = {
        "Q1_JAN_MAR": "Q1 (Janvier -> Mars)",
        "Q2_APR_JUN": "Q2 (Avril -> Juin)",
        "Q3_JUL_SEP": "Q3 (Juillet -> Septembre)",
        "Q4_OCT_DEC": "Q4 (Octobre -> Decembre)",
    }
    for idx, quarter in enumerate(q_list):
        ax = axes[idx // 2][idx % 2]
        subset = quarter_clustered[quarter_clustered["period"] == quarter]
        if subset.empty:
            ax.set_title(f"{quarter_titles[quarter]} (vide)")
            ax.axis("off")
            continue
        ax.scatter(
            subset["long"],
            subset["lat"],
            c=subset["cluster_local"],
            s=6,
            cmap="tab20",
            alpha=0.7,
        )
        ax.set_title(f"{quarter_titles[quarter]} - Toutes annees")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle("Clusters saisonniers (toutes annees confondues, sans bruit)", y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../../data/cleaned/cleaned_lyon_data.csv")
    df = pd.read_csv(file_path)

    clustered_df, stats_df = temporal_dbscan_analysis(df)

    # Affichage des cartes uniquement
    plot_temporal_cluster_comparison(
        clustered_df,
        selected_year_groups=("2008_2011", "2012_2015", "2016_2019"),
    )

    # Exports optionnels
    output_dir = os.path.join(current_dir, "../../data/processed")
    os.makedirs(output_dir, exist_ok=True)
    clustered_df.to_csv(os.path.join(output_dir, "temporal_clusters.csv"), index=False)
    stats_df.to_csv(os.path.join(output_dir, "temporal_cluster_stats.csv"), index=False)
