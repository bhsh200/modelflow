import math

import pandas as pd


def _num(x) -> float | None:
    """Convert to a JSON-safe float (NaN/inf become None)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def profile(df: pd.DataFrame) -> tuple[dict[str, str], dict]:
    """Return ({column: dtype}, {column: stats}) for a DataFrame."""
    columns = {str(c): str(df[c].dtype) for c in df.columns}
    stats: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        info: dict = {"missing": int(s.isna().sum())}

        if is_numeric(s):
            clean = s.dropna()
            info.update(
                kind="numeric",
                mean=_num(clean.mean()) if len(clean) else None,
                std=_num(clean.std()) if len(clean) > 1 else None,
                min=_num(clean.min()) if len(clean) else None,
                max=_num(clean.max()) if len(clean) else None,
                # decile cut points: the baseline bins for drift (PSI) later
                deciles=[_num(q) for q in clean.quantile([i / 10 for i in range(1, 10)])]
                if len(clean)
                else [],
            )
        else:
            freqs = s.astype("string").fillna("<missing>").value_counts(normalize=True)
            info.update(
                kind="categorical",
                n_unique=int(s.nunique()),
                top={str(k): float(v) for k, v in freqs.head(50).items()},
            )

        stats[str(col)] = info

    return columns, stats