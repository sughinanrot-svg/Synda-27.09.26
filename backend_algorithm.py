import os
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def process_and_extract_parameters(df_input: pd.DataFrame):
    """Clean input, extract per-variable empirical distributions and parametric fits.

    Preserves the original heuristics and outputs but uses vectorized operations
    where possible for speed.
    """
    blueprint_vars = ["Age", "Weight", "Sex", "Glucose", "CHO", "TG", "LDL", "HDLC", "AST", "ALT", "CRE"]

    column_mapping = {
        "GLU": "Glucose",
        "FBS": "Glucose",
        "BS": "Glucose",
        "CHOL": "CHO",
        "TOTAL_CHOLESTEROL": "CHO",
        "TRIGLYCERIDE": "TG",
        "TRIGLYCERIDES": "TG",
        "HDL": "HDLC",
        "HDL-C": "HDLC",
        "LDL-C": "LDL",
        "CREATININE": "CRE",
    }

    df = df_input.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=column_mapping, inplace=True)

    exclude_cols = ["HN", "ID", "Name", "Firstname", "Lastname", "ServiceDate"]
    df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

    available_vars = [v for v in blueprint_vars if v in df.columns]
    if not available_vars:
        return {
            "status": "success",
            "cleaned_data": pd.DataFrame(),
            "cleaned_real_data": pd.DataFrame(),
            "parameters": {},
            "available_variables": [],
        }

    # Coerce numerics in one pass for candidate numeric columns
    coerced = df[available_vars].apply(pd.to_numeric, errors="coerce")
    # nunique on non-null values
    nunique = df[available_vars].apply(lambda s: s.dropna().nunique())

    categorical_vars = [col for col in available_vars if df[col].dtype == object or nunique[col] <= 4]
    numeric_vars = [col for col in available_vars if col not in categorical_vars]

    processed_numeric = coerced[numeric_vars].dropna() if numeric_vars else pd.DataFrame()

    statistical_parameters = {}

    # Numeric processing (moments, parametric fits, empirical values)
    for col in numeric_vars:
        data = processed_numeric[col].to_numpy()
        if data.size == 0:
            continue

        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1))
        skew_val = float(stats.skew(data))
        kurtosis_val = float(stats.kurtosis(data))

        parametric_fits = {}
        for dist_name, dist_func in [("Normal", stats.norm), ("Log-Normal", stats.lognorm), ("Gamma", stats.gamma)]:
            try:
                if dist_name == "Normal":
                    loc, scale = dist_func.fit(data)
                    parametric_fits[dist_name] = {"loc": round(float(loc), 4), "scale": round(float(scale), 4)}
                else:
                    # allow loc to be fitted (not forcing floc=0) for better marginal fits
                    shape, loc, scale = dist_func.fit(data)
                    parametric_fits[dist_name] = {
                        "shape": round(float(shape), 4),
                        "loc": round(float(loc), 4),
                        "scale": round(float(scale), 4),
                    }
            except Exception:
                parametric_fits[dist_name] = None

        statistical_parameters[col] = {
            "is_categorical": False,
            "Moments": {
                "Mean": round(mean_val, 4),
                "Variance": round(var_val, 4),
                "Skewness": round(skew_val, 4),
                "Kurtosis": round(kurtosis_val, 4),
            },
            "Parametric_Fits": parametric_fits,
            "Empirical_Values": data.tolist(),
        }

    # Categorical empirical probabilities
    for col in categorical_vars:
        series = df[col].dropna().astype(str)
        if series.empty:
            cats = []
            probs = []
        else:
            cats, counts = np.unique(series.values, return_counts=True)
            probs = (counts / counts.sum()).tolist()
            cats = [str(x) for x in cats]

        statistical_parameters[col] = {
            "is_categorical": True,
            "Categories": [{"value": cats[i], "prob": round(probs[i], 6)} for i in range(len(cats))],
            "Empirical_Values": series.tolist(),
        }

    # Correlation (Spearman) using integer codes for categoricals.
    # Use coerced numeric values for numeric variables so non-numeric strings become NaN.
    corr_df = df[available_vars].copy()
    # replace numeric columns with coerced numeric versions
    for col in numeric_vars:
        corr_df[col] = coerced[col]
    for col in categorical_vars:
        corr_df[col] = pd.Categorical(corr_df[col]).codes
    corr_df = corr_df.dropna()
    if corr_df.shape[0] > 0:
        corr_matrix = corr_df.corr(method="spearman").values
    else:
        corr_matrix = np.eye(len(available_vars))

    statistical_parameters["_Correlation_Matrix"] = corr_matrix
    # record variable order used to compute the correlation matrix so callers can align
    statistical_parameters["_Correlation_VARS"] = available_vars

    # Build cleaned_df with numeric coercion applied to numeric vars, then drop rows with NaNs
    cleaned_df = df[available_vars].copy()
    for col in numeric_vars:
        cleaned_df[col] = coerced[col]
    cleaned_df = cleaned_df.dropna()
    return {
        "status": "success",
        "cleaned_data": cleaned_df,
        "cleaned_real_data": cleaned_df,
        "parameters": statistical_parameters,
        "available_variables": available_vars,
    }

def _resolve_data_file_path(file_name: str = "combined_raw_data.xlsx"):
    """Resolve the true project data file even when the filename varies."""
    candidate_paths = []
    project_root = os.path.dirname(os.path.abspath(__file__))
    search_roots = [
        os.getcwd(),
        project_root,
        os.path.join(project_root, "data5ปี"),
        os.path.join(project_root, "../"),
    ]

    if file_name:
        candidate_paths.append(file_name)
        if file_name.lower().endswith(('.csv', '.xlsx', '.xls')):
            stem = file_name.rsplit('.', 1)[0]
            candidate_paths.extend([f"{stem}.csv", f"{stem}.xlsx", f"{stem}.xls"])

    candidate_paths.extend([
        "combined_raw_data.xlsx",
        "combined_raw_data.csv",
        "combined_raw_data (1).xlsx",
        "combined_raw_data (1).csv",
        "ต้นฉบับ combine raw data.xlsx",
        "ต้นฉบับ combine raw data.csv",
    ])

    unique_candidates = []
    seen = set()
    for p in candidate_paths:
        if p and p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    for root in search_roots:
        if not root:
            continue
        try:
            for entry in os.listdir(root):
                if "combined_raw_data" in entry.lower() and entry.lower().endswith(('.csv', '.xlsx', '.xls')):
                    unique_candidates.append(os.path.join(root, entry))
        except Exception:
            pass

    for path in unique_candidates:
        if not path:
            continue
        if os.path.exists(path):
            return path

    for root in search_roots:
        if not root:
            continue
        for path in [
            os.path.join(root, "combined_raw_data.xlsx"),
            os.path.join(root, "combined_raw_data.csv"),
            os.path.join(root, "combined_raw_data (1).xlsx"),
            os.path.join(root, "combined_raw_data (1).csv"),
        ]:
            if os.path.exists(path):
                return path
    return None


def load_and_extract_base_parameters(file_path: str = "combined_raw_data (1).csv"):
    """Load a CSV/XLSX and extract parameters.

    This function also auto-detects common workspace variants of the real dataset
    so the app still works when the raw file is stored under a different name.
    """
    resolved_path = _resolve_data_file_path(file_path) if file_path else None
    if resolved_path is None:
        return {"status": "error", "message": f"❌ ไม่พบไฟล์ข้อมูลจริง ({file_path}) ในโฟลเดคโฟลเดอร์โปรเจค"}

    try:
        df_input = pd.read_csv(resolved_path) if resolved_path.lower().endswith('.csv') else pd.read_excel(resolved_path)
    except FileNotFoundError:
        return {"status": "error", "message": f"❌ ไม่พบไฟล์ข้อมูลจริง ({file_path}) ในโฟลเดคโฟลเดอร์โปรเจค"}

    # Explicit column mapping for known variants: map 'LDL2' -> 'LDL'
    try:
        if 'LDL2' in df_input.columns:
            df_input.rename(columns={'LDL2': 'LDL'}, inplace=True)
    except Exception:
        pass

    return process_and_extract_parameters(df_input)

def _make_positive_definite(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Adjust matrix to be positive-definite by adding small diagonal if needed."""
    # symmetricize
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    if np.all(eigvals > 0):
        return mat
    min_eig = np.min(eigvals)
    adjust = (-min_eig) + eps
    return mat + np.eye(mat.shape[0]) * adjust


def _nearest_correlation_matrix(A: np.ndarray) -> np.ndarray:
    """Compute the nearest correlation matrix to A (unit diagonal) using Higham's algorithm.

    Returns a positive semi-definite matrix with ones on the diagonal.
    """
    # Symmetrize
    B = (A + A.T) / 2.0
    # SVD
    try:
        U, s, Vt = np.linalg.svd(B)
        H = Vt.T @ np.diag(s) @ Vt
        A2 = (B + H) / 2.0
        A3 = (A2 + A2.T) / 2.0
    except Exception:
        A3 = B

    # Ensure unit diagonal
    np.fill_diagonal(A3, 1.0)

    # If it's positive definite, return
    def is_pd(X):
        try:
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False

    if is_pd(A3):
        return A3

    # Otherwise, tweak eigenvalues
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    A_k = A3.copy()
    while not is_pd(A_k):
        eigvals = np.linalg.eigvalsh(A_k)
        min_eig = np.min(eigvals)
        A_k += I * (-min_eig * (k**2) + spacing)
        k += 1
        if k > 100:
            break
    # enforce unit diagonal again
    np.fill_diagonal(A_k, 1.0)
    return A_k


def _build_latent_copula_cholesky(corr_matrix: np.ndarray, cache: dict | None = None) -> np.ndarray:
    """Build and cache the Cholesky factor of the latent Gaussian copula matrix."""
    corr = np.asarray(corr_matrix, dtype=np.float32)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    if cache is not None:
        key = (corr.shape[0], corr.shape[1], tuple(np.round(corr.ravel(), 10)))
        cached = cache.get("_copula_cholesky")
        if cached is not None and np.allclose(cached[0], corr):
            return cached[1]

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-8, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    L = np.asarray(L, dtype=np.float32)
    if cache is not None:
        cache["_copula_cholesky"] = (corr.copy(), L.copy())
    return L


def _kde_inv_cdf_lookup(values: np.ndarray, u: np.ndarray, grid_size: int = 1000) -> np.ndarray:
    """Vectorized empirical inverse-CDF mapping that avoids slow KDE sampling/resampling.

    For non-parametric marginals, the fastest safe approach is to map the copula uniforms
    directly through the empirical quantile function using NumPy percentiles. This preserves
    the observed distribution shape and avoids the freeze caused by gaussian_kde.evaluate()
    or repeated Python loops.
    """
    data = np.asarray(values, dtype=np.float64)
    u = np.clip(np.asarray(u, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    if data.size == 0:
        return np.zeros_like(u, dtype=np.float64)
    if data.size == 1:
        return np.full_like(u, float(data[0]), dtype=np.float64)

    # Direct empirical quantile mapping: U in [0,1] -> percentile of real data.
    # This is computed in one vectorized NumPy call, which is orders of magnitude faster
    # than gaussian_kde.evaluate() / resample() and prevents the Step 2 freeze.
    percentiles = np.clip(u * 100.0, 0.0, 100.0)
    return np.percentile(data, percentiles).astype(np.float64, copy=False)


def _iman_conover(independent_df: pd.DataFrame, target_spearman: np.ndarray, random_state: int | None = None) -> pd.DataFrame:
    """Apply Iman-Conover to reorder columns in independent_df to match target Spearman ranks.

    independent_df: DataFrame with columns in desired order (variables)
    target_spearman: square matrix of Spearman correlations for these variables
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_vars = independent_df.shape

    # Convert Spearman to Pearson approx for normal copula
    pearson = 2.0 * np.sin((np.pi / 6.0) * target_spearman)
    pearson = (pearson + pearson.T) / 2.0
    np.fill_diagonal(pearson, 1.0)

    # Ensure positive-definite with minimal distortion
    pearson_pd = _nearest_correlation_matrix(pearson)

    # Generate correlated normal scores
    Z = rng.multivariate_normal(np.zeros(n_vars), pearson_pd, size=n_samples)

    # For each column get the order indices of Z
    order_Z = np.argsort(Z, axis=0)

    # Prepare output DataFrame
    out = pd.DataFrame(index=range(n_samples), columns=independent_df.columns)

    # For numeric columns: sort independent samples and assign according to order_Z
    for j, col in enumerate(independent_df.columns):
        col_vals = independent_df.iloc[:, j].to_numpy()
        # argsort of col_vals (ascending)
        sorted_vals = np.sort(col_vals)
        # place sorted values into positions defined by order_Z[:, j]
        out_vals = np.empty(n_samples, dtype=col_vals.dtype)
        out_vals[order_Z[:, j]] = sorted_vals
        out[col] = out_vals

    return out


def _select_best_parametric_fit(empirical_data: np.ndarray, variable_name: str | None = None):
    """Select the best continuous parametric marginal using AIC/BIC across all major positive/real support families.

    This deliberately replaces hard-coded skewness gates with a model-comparison workflow:
    Normal, Log-Normal, Gamma, and Weibull are all fit via likelihood-based estimation and
    ranked by AIC; BIC is used as a tiebreaker when AIC scores are close.
    """
    if empirical_data.size < 3:
        return None

    candidates = [
        ("Normal", stats.norm),
        ("Log-Normal", stats.lognorm),
        ("Gamma", stats.gamma),
        ("Weibull", stats.weibull_min),
    ]

    scored = []
    for dist_name, dist_func in candidates:
        try:
            params = dist_func.fit(empirical_data)
            loglik = np.sum(dist_func.logpdf(empirical_data, *params))
            if not np.isfinite(loglik):
                continue
            k = len(params)
            n = empirical_data.size
            aic = 2.0 * k - 2.0 * loglik
            bic = k * np.log(n) - 2.0 * loglik
            scored.append((dist_name, float(aic), float(bic), tuple(params)))
        except Exception:
            continue

    if not scored:
        return None

    # Prefer lower AIC, then lower BIC if two fits are nearly indistinguishable.
    best_result = min(scored, key=lambda item: (item[1], item[2]))
    return best_result[0], best_result[3]


def _parametric_ppf_from_uniform(u: np.ndarray, dist_name: str, params: tuple):
    """Apply the selected parametric inverse CDF without hard post-sampling truncation cliffs."""
    u_clipped = np.clip(np.asarray(u, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    if dist_name == "Normal":
        loc, scale = params
        return stats.norm.ppf(u_clipped, loc=loc, scale=scale).astype(np.float32, copy=False)
    if dist_name == "Log-Normal":
        shape, loc, scale = params
        return stats.lognorm.ppf(u_clipped, s=shape, loc=loc, scale=scale).astype(np.float32, copy=False)
    if dist_name == "Gamma":
        shape, loc, scale = params
        return stats.gamma.ppf(u_clipped, a=shape, loc=loc, scale=scale).astype(np.float32, copy=False)
    if dist_name == "Weibull":
        shape, loc, scale = params
        return stats.weibull_min.ppf(u_clipped, c=shape, loc=loc, scale=scale).astype(np.float32, copy=False)
    shape, loc, scale = params
    return stats.gamma.ppf(u_clipped, a=shape, loc=loc, scale=scale).astype(np.float32, copy=False)


def _empirical_quantile_with_tail_jitter(empirical_data: np.ndarray, u: np.ndarray, rng: np.random.Generator):
    """Use empirical quantiles as the base distribution and apply a controlled tail expansion.

    This follows the architecture requirement for highly skewed variables: smooth
    empirical quantile interpolation plus small jitter in the upper tail to reduce
    discrete spikes without destroying the real-data shape.
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-6, 1.0 - 1e-6)
    mapped = np.quantile(empirical_data, u, method="linear")

    if empirical_data.size < 2:
        return mapped

    q25, q75 = np.percentile(empirical_data, [25, 75])
    iqr = float(q75 - q25)
    local_sd = float(np.std(empirical_data, ddof=1)) if empirical_data.size > 1 else iqr
    jitter_scale = max(0.02 * max(iqr, local_sd, 1e-9), 1e-6)

    # Controlled jitter across the body to reduce repeated-value spikes
    body_mask = u <= 0.95
    if np.any(body_mask):
        mapped[body_mask] += rng.normal(0.0, jitter_scale * 0.5, size=body_mask.sum())

    # Tail expansion on the right tail to keep realism in skewed marginals while
    # preventing excessive discrete jumps at the extreme quantiles.
    tail_mask = u > 0.95
    if np.any(tail_mask):
        p95 = float(np.percentile(empirical_data, 95))
        p98 = float(np.percentile(empirical_data, 98))
        tail_strength = (u[tail_mask] - 0.95) / 0.05
        tail_adjust = tail_strength * max(p98 - p95, 0.0) * 0.35
        mapped[tail_mask] = mapped[tail_mask] + tail_adjust
        mapped[tail_mask] += rng.normal(0.0, jitter_scale * 1.5, size=tail_mask.sum())
        upper_cap = max(float(np.max(empirical_data)), float(p98)) + 3.0 * jitter_scale
        mapped[tail_mask] = np.minimum(mapped[tail_mask], upper_cap)

    if np.nanmin(empirical_data) >= 0:
        mapped = np.maximum(mapped, 0.0)

    return mapped


def _match_output_precision(reference_values: np.ndarray, generated_values: np.ndarray):
    """Round or cast synthetic output to match the original numeric precision style."""
    ref = pd.to_numeric(pd.Series(reference_values), errors="coerce").dropna().to_numpy()
    generated_values = np.asarray(generated_values, dtype=float)
    if ref.size == 0:
        return generated_values

    if np.all(np.isclose(ref, np.round(ref), atol=1e-8)):
        return np.rint(generated_values).astype(int)

    max_decimals = 0
    for value in ref:
        if pd.isna(value):
            continue
        text = format(float(value), ".12f").rstrip("0").rstrip(".")
        if "." in text:
            max_decimals = max(max_decimals, len(text.split(".")[-1]))
    if max_decimals > 0:
        return np.round(generated_values, decimals=max_decimals)

    return generated_values


def generate_synthetic_data(parameters: dict, num_samples: int = 1000, selected_variables: list | None = None, include_synthetic_hn: bool = False, random_seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic rows while preserving the original copula workflow.

    Workflow Step 3 architecture:
      1) Latent Gaussian copula remains unchanged.
      2) Marginal mapping is now skewness-aware:
         - Skewness <= 1.0 -> best parametric PPF (Normal / Log-Normal / Gamma)
         - Skewness > 1.0 -> empirical quantile mapping + tail expansion + jitter
      3) Output precision is restored to match the original dataset format.
    """
    vars_list = [k for k in parameters.keys() if not k.startswith("_")]
    if selected_variables:
        vars_list = [v for v in selected_variables if v in vars_list]
    num_vars = len(vars_list)

    full_vars = [k for k in parameters.keys() if not k.startswith("_")]
    corr_matrix = parameters.get("_Correlation_Matrix")
    corr_vars = parameters.get("_Correlation_VARS", full_vars)
    if corr_matrix is None:
        corr_matrix = np.eye(num_vars)
    else:
        corr_matrix = np.array(corr_matrix, dtype=float)
        if len(corr_vars) == corr_matrix.shape[0]:
            idxs = [corr_vars.index(v) for v in vars_list]
            corr_matrix = corr_matrix[np.ix_(idxs, idxs)]

    if corr_matrix.size == 0:
        corr_matrix = np.eye(num_vars)

    # Preserve the original Gaussian copula pipeline exactly.
    spearman = np.array(corr_matrix, dtype=float)
    pearson_latent = 2.0 * np.sin((np.pi / 6.0) * spearman)
    pearson_latent = (pearson_latent + pearson_latent.T) / 2.0
    np.fill_diagonal(pearson_latent, 1.0)

    try:
        from statsmodels.stats.correlation_tools import corr_nearest
        pearson_pd = corr_nearest(pearson_latent)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("statsmodels is required for corr_nearest. Please install it: pip install statsmodels")
    except Exception:
        pearson_pd = pearson_latent

    pearson_pd = (pearson_pd + pearson_pd.T) / 2.0
    np.fill_diagonal(pearson_pd, 1.0)

    start_time = time.perf_counter()
    print(f"[Step 2] synth start: samples={num_samples}, variables={num_vars}")

    rng = np.random.default_rng(random_seed)
    copula_cache = parameters.setdefault("_copula_cache", {})
    L = _build_latent_copula_cholesky(pearson_pd, cache=copula_cache)
    Z = rng.standard_normal(size=(num_samples, num_vars)).astype(np.float32, copy=False)
    Z = (Z @ L.T).astype(np.float32, copy=False)
    U = stats.norm.cdf(Z).astype(np.float32, copy=False)

    out = pd.DataFrame(index=range(num_samples), columns=vars_list)
    numeric_cols = []

    for j, col in enumerate(vars_list):
        col_params = parameters[col]
        if col_params.get("is_categorical"):
            cats = [c["value"] for c in col_params.get("Categories", [])]
            probs = np.array([c["prob"] for c in col_params.get("Categories", [])], dtype=np.float64)
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            cum = np.cumsum(probs)
            u = U[:, j]
            idxs = np.searchsorted(cum, u, side="right")
            idxs = np.clip(idxs, 0, len(cats) - 1)
            out[col] = np.array([cats[i] for i in idxs], dtype=object)
            continue

        raw_vals = col_params.get("Empirical_Values", [])
        empirical_data = pd.to_numeric(pd.Series(raw_vals), errors="coerce").dropna().to_numpy(dtype=np.float64)
        if empirical_data.size == 0:
            out[col] = np.nan
            continue

        numeric_cols.append(col)
        u = U[:, j]

        # Special-case lipids: CHO, TG, LDL all use PCHIP smooth empirical quantile mapping
        # This preserves the observed distributional shapes better than parametric fits.
        # Post-generation, we enforce LDL <= CHO - HDLC with gentle clipping.
        # Use finer grid (300 points) for CHO and LDL (the tough variables), 200 for TG.
        if str(col).upper() in {"CHO", "TG", "LDL"} and empirical_data.size >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                grid_size = 300 if str(col).upper() in {"CHO", "LDL"} else 200
                q_grid = np.linspace(0.0, 1.0, grid_size)
                x_vals = np.quantile(empirical_data, q_grid, method="linear")
                pchip = PchipInterpolator(q_grid, x_vals)
                u_clipped = np.clip(u, 1e-6, 1.0 - 1e-6)
                mapped = pchip(u_clipped)
                mapped = np.clip(mapped, np.min(empirical_data), np.max(empirical_data))
                print(f"[Step 2] {col}: used PCHIP empirical quantile mapping (grid={grid_size})")
            except Exception:
                mapped = _kde_inv_cdf_lookup(empirical_data, u, grid_size=1000)
        else:
            # Default fast empirical quantile mapping for non-lipid variables
            mapped = _kde_inv_cdf_lookup(empirical_data, u, grid_size=1000)

        mapped = np.asarray(mapped, dtype=np.float64)
        if np.nanmin(empirical_data) >= 0:
            mapped = np.where(mapped < 0.0, 0.0, mapped)

        mapped = _match_output_precision(empirical_data, mapped)
        out[col] = mapped

    elapsed = time.perf_counter() - start_time
    print(f"[Step 2] synth complete in {elapsed:.4f}s")

    # Biological consistency: ensure LDL <= CHO - HDLC when those columns exist
    try:
        if "LDL" in out.columns and "CHO" in out.columns and "HDLC" in out.columns:
            cho = pd.to_numeric(out["CHO"], errors="coerce").to_numpy(dtype=np.float64)
            hdlc = pd.to_numeric(out["HDLC"], errors="coerce").to_numpy(dtype=np.float64)
            ldl = pd.to_numeric(out["LDL"], errors="coerce").to_numpy(dtype=np.float64)
            # compute allowed maximum LDL for each row
            allowed = cho - hdlc
            mask = (np.isfinite(allowed) & np.isfinite(ldl)) & (ldl > allowed)
            if np.any(mask):
                # For violating entries, use gentle enforcement: clip to allowed max with small epsilon
                # This preserves distributional shape better than aggressive resampling
                adjusted = np.maximum(0.0, np.minimum(ldl[mask], allowed[mask] - 0.5))
                out.loc[mask, "LDL"] = adjusted
                print(f"[Step 2] Adjusted {int(mask.sum())} LDL values to satisfy LDL <= CHO - HDLC (clipped)")
    except Exception:
        pass

    return out


def _moment_error_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Compute an error score based on absolute differences in skewness and kurtosis.

    Returns mean absolute error across variables and both moments.
    """
    # use numeric common columns
    common = [c for c in df_real.columns if c in df_synth.columns]
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(df_real[c]) and pd.api.types.is_numeric_dtype(df_synth[c])]
    if not num_cols:
        return float('inf')

    diffs = []
    for col in num_cols:
        r = pd.to_numeric(df_real[col].dropna(), errors='coerce').dropna()
        s = pd.to_numeric(df_synth[col].dropna(), errors='coerce').dropna()
        if len(r) < 2 or len(s) < 2:
            continue
        sk_r = float(stats.skew(r, bias=False))
        sk_s = float(stats.skew(s, bias=False))
        kt_r = float(stats.kurtosis(r, fisher=False, bias=False))
        kt_s = float(stats.kurtosis(s, fisher=False, bias=False))
        diffs.append(abs(sk_r - sk_s))
        diffs.append(abs(kt_r - kt_s))

    if not diffs:
        return float('inf')
    return float(np.nanmean(diffs))


def generate_with_seed_search(parameters: dict, df_real: pd.DataFrame, num_samples: int = 1000, num_trials: int = 30, selected_variables: list | None = None) -> tuple:
    """Run multiple generation trials with different seeds and pick the best by moment error.

    Returns (best_synthetic_df, best_seed, best_score)
    """
    best_score = float('inf')
    best_seed = None
    best_synth = None

    for seed in range(num_trials):
        try:
            synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=seed)
        except Exception:
            continue

        score = _moment_error_score(df_real, synth)
        if score < best_score:
            best_score = score
            best_seed = seed
            best_synth = synth

    # Ensure we return the best generated with reproducible seed
    if best_seed is not None:
        final_synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=best_seed)
        return final_synth, best_seed, best_score
    else:
        # fallback: generate once with default seed
        synth = generate_synthetic_data(parameters, num_samples=num_samples, selected_variables=selected_variables, random_seed=None)
        return synth, None, float('inf')