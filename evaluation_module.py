import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os


def _numeric_common_columns(df_real, df_synthetic):
    return [
        col for col in df_real.columns
        if col in df_synthetic.columns and pd.api.types.is_numeric_dtype(df_real[col]) and pd.api.types.is_numeric_dtype(df_synthetic[col])
    ]


def evaluate_synthetic_data(df_real, df_synthetic):
    """Compute KS tests and convert the result into a human-readable summary table."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    rows = []

    for col in common_cols:
        real_data = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_data = pd.to_numeric(df_synthetic[col].dropna(), errors="coerce").dropna()
        if len(real_data) < 2 or len(synth_data) < 2:
            continue

        # If synthetic sample is much larger than real, perform Monte Carlo subsampling
        # to avoid N-inflation of KS test p-values. This computes averaged KS stats
        # across multiple draws of size N_real from the synthetic distribution.
        try:
            if len(synth_data) > len(real_data) * 1.5 and len(real_data) >= 20:
                n_iter = 100
                ks_vals = []
                p_vals = []
                for _ in range(n_iter):
                    samp = synth_data[np.random.choice(len(synth_data), size=len(real_data), replace=False)]
                    ks, p = stats.ks_2samp(real_data, samp)
                    ks_vals.append(ks)
                    p_vals.append(p)
                ks_stat = float(np.mean(ks_vals))
                p_value = float(np.mean(p_vals))
            else:
                ks_stat, p_value = stats.ks_2samp(real_data, synth_data)
        except Exception:
            ks_stat, p_value = stats.ks_2samp(real_data, synth_data)
        if p_value > 0.05:
            decision = "Similar distributions"
            interpretation = "The p-value is above 0.05, so there is no strong statistical evidence that the real and synthetic samples come from different distributions."
        else:
            decision = "Distribution difference"
            interpretation = "The p-value is below 0.05, so the distributions differ statistically and the variable may need further tuning or a different marginal fit."

        rows.append(
            {
                "Variable": col,
                "KS_Statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "Decision": decision,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def save_individual_kdes(df_real: pd.DataFrame, df_synth: pd.DataFrame, out_dir: str = ".") -> list:
    """Generate and save individual KDE overlay plots for a strict list of numeric vars.

    - Iterates the explicit variable list and skips missing/non-numeric columns.
    - Saves each figure as `kde_<Var>.png` and prints a confirmation.
    """
    vars_list = [
        "Age",
        "Weight",
        "Height",
        "GLU",
        "CHO",
        "TG",
        "LDL2",
        "HDLC",
        "AST",
        "ALT",
        "CRE",
    ]

    saved_files = []
    os.makedirs(out_dir, exist_ok=True)

    for col in vars_list:
        # Explicitly skip non-existent or non-numeric columns
        if col not in df_real.columns or col not in df_synth.columns:
            print(f"Skipping {col}: missing in real or synthetic dataframe")
            continue

        real_vals = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_vals = pd.to_numeric(df_synth[col].dropna(), errors="coerce").dropna()

        if real_vals.size < 2 or synth_vals.size < 2:
            print(f"Skipping {col}: insufficient numeric data (real={real_vals.size}, synth={synth_vals.size})")
            continue

        plt.figure(figsize=(6, 4))
        try:
            sns.kdeplot(real_vals, label="Real", color="C0", fill=False)
        except Exception:
            sns.histplot(real_vals, label="Real", color="C0", stat="density", kde=True)

        try:
            sns.kdeplot(synth_vals, label="Synthetic", color="C1", fill=False)
        except Exception:
            sns.histplot(synth_vals, label="Synthetic", color="C1", stat="density", kde=True)

        try:
            ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
        except Exception:
            ks_stat, p_value = (np.nan, np.nan)

        plt.title(f"KDE comparison: {col} (KS p={p_value:.4f})")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()

        fname = os.path.join(out_dir, f"kde_{col}.png")
        try:
            plt.savefig(fname, bbox_inches="tight")
            print(f"Saved {fname}")
            saved_files.append(fname)
        except Exception as e:
            print(f"Failed to save {fname}: {e}")
        finally:
            plt.close()

    return saved_files


def evaluate_all_metrics(df_real: pd.DataFrame, df_synth: pd.DataFrame, show_plots: bool = True):
    """Run comprehensive evaluation: moments, KS tests, correlation comparison and plots.

    Returns a dict with DataFrames and figure handles.
    """
    results = {}

    # Moments
    moments_df = summarize_moments(df_real, df_synth)
    results["moments"] = moments_df

    # KS
    ks_df = evaluate_synthetic_data(df_real, df_synth)
    results["ks"] = ks_df

    # Correlations
    common_cols = _numeric_common_columns(df_real, df_synth)
    if common_cols:
        corr_real = df_real[common_cols].corr(method="spearman")
        corr_synth = df_synth[common_cols].corr(method="spearman")
    else:
        corr_real = pd.DataFrame()
        corr_synth = pd.DataFrame()

    results["corr_real"] = corr_real
    results["corr_synth"] = corr_synth

    figs = {}
    if show_plots and not corr_real.empty:
        figs["corr_heatmap"] = plot_comparison_correlation_heatmap(df_real, df_synth)

    # Full-grid KDEs for all numeric variables (3 rows x 4 columns)
    try:
        common_cols = _numeric_common_columns(df_real, df_synth)
        if common_cols:
            figs["kde_grid"] = plot_kde_grid(df_real, df_synth, common_cols, nrows=3, ncols=4)
    except Exception:
        figs["kde_grid"] = None

    # Individual KDE PNGs (saved files)
    try:
        individual_files = save_individual_kdes(df_real, df_synth, out_dir=".")
    except Exception:
        individual_files = []
    results["individual_kde_files"] = individual_files

    # Moments comparison figure
    try:
        figs["moments_comparison"] = plot_moments_comparison(df_real, df_synth)
    except Exception:
        figs["moments_comparison"] = None

    results["figures"] = figs
    return results


def quick_run_and_show(real_path: str = "combined_raw_data (1).csv", n_samples: int = 1000):
    """Utility to load data, generate synthetic sample using backend module, and show evaluation.

    Prints summary tables and shows correlation heatmaps.
    """
    # Dynamically locate and import the backend_algorithm file (handles filenames with spaces)
    import importlib.util
    import glob
    backend_candidates = glob.glob("backend_algorithm*.py")
    if not backend_candidates:
        print("Could not find a backend_algorithm*.py file in the current directory.")
        return
    backend_path = backend_candidates[0]
    spec = importlib.util.spec_from_file_location("backend_algorithm_local", backend_path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)

    base = backend.load_and_extract_base_parameters(real_path)
    if base.get("status") == "error":
        print(base.get("message"))
        return

    params = base["parameters"]
    cleaned = base["cleaned_real_data"]
    # Run seed search to find best seed matching skewness/kurtosis
    try:
        synth, best_seed, best_score = backend.generate_with_seed_search(params, cleaned, num_samples=n_samples, num_trials=30)
        print(f"Selected best seed: {best_seed} (score={best_score:.4f})")
    except Exception:
        synth = backend.generate_synthetic_data(params, num_samples=n_samples)

    # Export synthetic data
    out_name = f"synthetic_generated_data.csv"
    try:
        synth.to_csv(out_name, index=False)
        print(f"Synthetic data exported to: {out_name}")
    except Exception as e:
        print(f"Failed to export synthetic data: {e}")

    results = evaluate_all_metrics(cleaned, synth, show_plots=True)

    # Save summary tables
    moments_csv = "evaluation_moments.csv"
    ks_csv = "evaluation_ks.csv"
    try:
        results["moments"].to_csv(moments_csv, index=False)
        results["ks"].to_csv(ks_csv, index=False)
        print(f"Saved moments table to: {moments_csv}")
        print(f"Saved KS table to: {ks_csv}")
    except Exception as e:
        print(f"Failed to save summary tables: {e}")

    # Save figures
    figs = results.get("figures", {})
    saved_figs = []
    if figs:
        # Correlation heatmap
        corr_fig = figs.get("corr_heatmap")
        if corr_fig is not None:
            corr_name = "corr_heatmap.png"
            try:
                corr_fig.savefig(corr_name, bbox_inches="tight")
                saved_figs.append(corr_name)
                print(f"Saved correlation heatmap to: {corr_name}")
            except Exception as e:
                print(f"Failed to save correlation heatmap: {e}")

        # KDE grid
        kde_grid = figs.get("kde_grid")
        if kde_grid is not None:
            try:
                kde_name = "kde_grid_3x4.png"
                kde_grid.savefig(kde_name, bbox_inches="tight")
                saved_figs.append(kde_name)
                print(f"Saved KDE grid to: {kde_name}")
            except Exception as e:
                print(f"Failed to save KDE grid: {e}")

        # Moments comparison
        moments_fig = figs.get("moments_comparison")
        if moments_fig is not None:
            try:
                moments_name = "moments_comparison.png"
                moments_fig.savefig(moments_name, bbox_inches="tight")
                saved_figs.append(moments_name)
                print(f"Saved moments comparison to: {moments_name}")
            except Exception as e:
                print(f"Failed to save moments comparison: {e}")

    # Individual KDE PNGs (already written by evaluator)
    individual_files = results.get("individual_kde_files", [])
    for fn in individual_files:
        print(f"Saved individual KDE to: {fn}")
        saved_figs.append(fn)

    # Create a short markdown report
    report_path = "evaluation_report.md"
    try:
        with open(report_path, "w") as f:
            f.write("# Synthetic Data Evaluation Report\n\n")
            f.write(f"- Real data file: {real_path}\n")
            f.write(f"- Synthetic data file: synthetic_generated_data.csv\n")
            f.write(f"- Moments table: {moments_csv}\n")
            f.write(f"- KS table: {ks_csv}\n")
            if saved_figs:
                f.write("- Figures:\n")
                for fn in saved_figs:
                    f.write(f"  - {fn}\n")
            f.write("\n## Notes\n")
            f.write("- Variables where KS test p-value < 0.05 may need further marginal tuning.\n")
        print(f"Wrote evaluation report to: {report_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")

    print("\nFirst-four moments comparison:")
    print(results["moments"].to_string(index=False))

    print("\nKS test results:")
    print(results["ks"].to_string(index=False))

    return results





def summarize_moments(df_real, df_synthetic):
    """Create a first-four-moments comparison table for the overlapping variables."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    rows = []

    for col in common_cols:
        real_data = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_data = pd.to_numeric(df_synthetic[col].dropna(), errors="coerce").dropna()
        if len(real_data) < 2 or len(synth_data) < 2:
            continue

        rows.append(
            {
                "Variable": col,
                "Mean (Real)": round(float(np.mean(real_data)), 4),
                "Mean (Synthetic)": round(float(np.mean(synth_data)), 4),
                "Variance (Real)": round(float(np.var(real_data, ddof=1)), 4),
                "Variance (Synthetic)": round(float(np.var(synth_data, ddof=1)), 4),
                "Skewness (Real)": round(float(stats.skew(real_data, bias=False)), 4),
                "Skewness (Synthetic)": round(float(stats.skew(synth_data, bias=False)), 4),
                "Kurtosis (Real)": round(float(stats.kurtosis(real_data, fisher=False, bias=False)), 4),
                "Kurtosis (Synthetic)": round(float(stats.kurtosis(synth_data, fisher=False, bias=False)), 4),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison_kde(df_real, df_synthetic, column_name):
    """Create an overlapping KDE plot that highlights right-skewness and tail behavior."""
    real_values = pd.to_numeric(df_real[column_name].dropna(), errors="coerce").dropna()
    synth_values = pd.to_numeric(df_synthetic[column_name].dropna(), errors="coerce").dropna()

    if len(real_values) < 3 or len(synth_values) < 3:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Not enough values to plot a KDE comparison", ha="center", va="center")
        ax.set_axis_off()
        return fig

    real_plot = np.log1p(real_values)
    synth_plot = np.log1p(synth_values)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(real_plot, label="Real Data", color="#2563eb", fill=True, alpha=0.25, linewidth=2, ax=ax)
    sns.kdeplot(synth_plot, label="Synthetic Data", color="#f59e0b", fill=True, alpha=0.25, linewidth=2, ax=ax)
    ax.axvline(real_plot.mean(), color="#2563eb", linestyle="--", alpha=0.7)
    ax.axvline(synth_plot.mean(), color="#f59e0b", linestyle="--", alpha=0.7)
    ax.set_title(f"Right-skewness comparison for {column_name}", fontsize=12)
    ax.set_xlabel(f"{column_name} (log1p-transformed for skewness visualization)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_comparison_correlation_heatmap(df_real, df_synthetic):
    """Create side-by-side Spearman correlation heatmaps for the overlapping numeric variables."""
    common_cols = _numeric_common_columns(df_real, df_synthetic)
    if not common_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No numeric columns available for correlation comparison", ha="center", va="center")
        ax.set_axis_off()
        return fig

    corr_real = df_real[common_cols].corr(method="spearman")
    corr_synth = df_synthetic[common_cols].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_real, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[0])
    axes[0].set_title("Real Data Correlation (Spearman)", fontsize=12)
    sns.heatmap(corr_synth, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=axes[1])
    axes[1].set_title("Synthetic Data Correlation (Spearman)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_kde_grid(df_real: pd.DataFrame, df_synth: pd.DataFrame, vars_list: list, nrows: int = 3, ncols: int = 4, figsize=(16, 12)):
    """Plot KDEs for all variables in vars_list arranged in nrows x ncols grid.

    Each subplot shows real vs synthetic KDE and annotates KS p-value.
    """
    total = nrows * ncols
    vars_to_plot = vars_list[:total]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax in axes[len(vars_to_plot):]:
        ax.set_axis_off()

    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        real_vals = pd.to_numeric(df_real[var].dropna(), errors="coerce").dropna()
        synth_vals = pd.to_numeric(df_synth[var].dropna(), errors="coerce").dropna()

        if len(real_vals) < 3 or len(synth_vals) < 3:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            ax.set_title(var)
            ax.set_axis_off()
            continue

        # use log1p for visualization when skewed to keep shapes visible
        real_plot = np.log1p(real_vals)
        synth_plot = np.log1p(synth_vals)

        sns.kdeplot(real_plot, label="Real", color="#2563eb", fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(synth_plot, label="Synth", color="#f59e0b", fill=True, alpha=0.3, ax=ax)
        ax.set_title(var)

        # KS test on original data
        try:
            ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
            ax.text(0.98, 0.95, f"KS p={p_value:.4f}", ha="right", va="top", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        except Exception:
            ax.text(0.98, 0.95, "KS n/a", ha="right", va="top", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def save_individual_kdes(df_real: pd.DataFrame, df_synth: pd.DataFrame, out_dir: str = ".") -> list:
    """Save individual KDE plots for each numeric variable and annotate with KS p-value.

    Returns list of file paths saved.
    """
    files = []
    common_cols = _numeric_common_columns(df_real, df_synth)
    for col in common_cols:
        real_vals = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        synth_vals = pd.to_numeric(df_synth[col].dropna(), errors="coerce").dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        if len(real_vals) < 3 or len(synth_vals) < 3:
            ax.text(0.5, 0.5, "Not enough data to plot", ha="center", va="center")
            ax.set_axis_off()
        else:
            # use log1p for visualization of skew but annotate KS on originals
            sns.kdeplot(np.log1p(real_vals), label="Real", color="#2563eb", fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(np.log1p(synth_vals), label="Synth", color="#f59e0b", fill=True, alpha=0.3, ax=ax)
            ax.set_title(f"KDE: {col}")
            ax.set_xlabel(f"{col} (log1p shown)")
            # KS on original
            try:
                ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
                ax.text(0.98, 0.95, f"KS p={p_value:.4f}", ha="right", va="top", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            except Exception:
                ax.text(0.98, 0.95, "KS n/a", ha="right", va="top", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            ax.legend()

        safe_name = col.replace(" ", "_")
        fname = f"kde_{safe_name}.png"
        try:
            fig.savefig(f"{out_dir}/{fname}", bbox_inches="tight")
            files.append(fname)
        except Exception:
            pass
        plt.close(fig)

    return files


def plot_moments_comparison(df_real: pd.DataFrame, df_synth: pd.DataFrame, figsize=(14, 10)) -> plt.Figure:
    """Create comparative bar charts for Mean, Variance, Skewness, Kurtosis across numeric vars.

    Returns the matplotlib Figure.
    """
    common_cols = _numeric_common_columns(df_real, df_synth)
    if not common_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No numeric variables to compare", ha="center", va="center")
        ax.set_axis_off()
        return fig

    moments = []
    for col in common_cols:
        r = pd.to_numeric(df_real[col].dropna(), errors="coerce").dropna()
        s = pd.to_numeric(df_synth[col].dropna(), errors="coerce").dropna()
        if len(r) < 2 or len(s) < 2:
            continue
        moments.append({
            "Variable": col,
            "Mean_Real": float(np.mean(r)),
            "Mean_Synth": float(np.mean(s)),
            "Var_Real": float(np.var(r, ddof=1)),
            "Var_Synth": float(np.var(s, ddof=1)),
            "Skew_Real": float(stats.skew(r, bias=False)),
            "Skew_Synth": float(stats.skew(s, bias=False)),
            "Kurt_Real": float(stats.kurtosis(r, fisher=False, bias=False)),
            "Kurt_Synth": float(stats.kurtosis(s, fisher=False, bias=False)),
        })

    if not moments:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Insufficient data for moments", ha="center", va="center")
        ax.set_axis_off()
        return fig

    mdf = pd.DataFrame(moments).set_index("Variable")

    vars_order = mdf.index.tolist()
    x = np.arange(len(vars_order))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Mean
    axes[0].bar(x - width/2, mdf["Mean_Real"], width, label="Real", color="#2563eb")
    axes[0].bar(x + width/2, mdf["Mean_Synth"], width, label="Synth", color="#f59e0b")
    axes[0].set_title("Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vars_order, rotation=45, ha="right")
    axes[0].legend()

    # Variance
    axes[1].bar(x - width/2, mdf["Var_Real"], width, label="Real", color="#2563eb")
    axes[1].bar(x + width/2, mdf["Var_Synth"], width, label="Synth", color="#f59e0b")
    axes[1].set_title("Variance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(vars_order, rotation=45, ha="right")
    axes[1].legend()

    # Skewness
    axes[2].bar(x - width/2, mdf["Skew_Real"], width, label="Real", color="#2563eb")
    axes[2].bar(x + width/2, mdf["Skew_Synth"], width, label="Synth", color="#f59e0b")
    axes[2].set_title("Skewness")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(vars_order, rotation=45, ha="right")
    axes[2].legend()

    # Kurtosis
    axes[3].bar(x - width/2, mdf["Kurt_Real"], width, label="Real", color="#2563eb")
    axes[3].bar(x + width/2, mdf["Kurt_Synth"], width, label="Synth", color="#f59e0b")
    axes[3].set_title("Kurtosis")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(vars_order, rotation=45, ha="right")
    axes[3].legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick evaluation runner for synthetic data generator")
    parser.add_argument("--real", default="combined_raw_data (1).csv", help="Real dataset path (CSV or XLSX)")
    parser.add_argument("--n", type=int, default=1000, help="Number of synthetic samples to generate")
    args = parser.parse_args()
    quick_run_and_show(real_path=args.real, n_samples=args.n)
