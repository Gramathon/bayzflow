# fx_utils/forecast.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pyro import poutine


@torch.no_grad()
def forecast_paths_primary_only(
    model,
    guide,
    dataset,
    last_seq_row_norm: np.ndarray,
    last_close: float,
    ret_mean: float,
    ret_std: float,
    steps: int = 24,
    samples: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Path-consistent Bayesian recurrent forecast.

    For each Monte Carlo path:
      - Sample ONE set of weights from the guide (trace+replay).
      - At each horizon, sample obs ~ StudentT(...) from the model.
      - Update that path's price and feature vector using its OWN sampled return.
    """

    device = next(model.parameters()).device

    def recompute_indicators(sim_closes: np.ndarray) -> dict:
        s = pd.Series(sim_closes)
        r = s.pct_change().values[-1] if len(s) > 1 else 0.0

        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else 50.0

        ma = s.rolling(20, min_periods=1).mean().iloc[-1]
        sd = s.rolling(20, min_periods=1).std().iloc[-1]
        hband = float(ma + 2 * sd)
        lband = float(ma - 2 * sd)
        bbp = float((s.iloc[-1] - lband) / (hband - lband)) if (hband - lband) > 0 else 0.5
        bw = float((hband - lband) / ma) if ma != 0 else 0.0

        flags = {
            "BB_above_upper": int(s.iloc[-1] > hband),
            "BB_below_lower": int(s.iloc[-1] < lband),
            "BB_cross_up": 0,
            "BB_cross_down": 0,
            "BB_cross_any": 0,
        }
        return {
            "Return": r,
            "RSI": rsi,
            "BB_mavg": float(ma),
            "BB_bbp": bbp,
            "BB_bw": bw,
            **flags,
        }

    def make_replayed_model(model, guide, x_proto: torch.Tensor):
        trace = poutine.trace(guide).get_trace(x_proto, None)

        def model_replayed(x):
            return poutine.replay(model, trace=trace)(x, None)

        return model_replayed

    F = last_seq_row_norm.shape[0]
    x_proto = torch.from_numpy(last_seq_row_norm).float().to(device).view(1, 1, F)

    replayed_models = [make_replayed_model(model, guide, x_proto) for _ in range(samples)]

    prices = np.zeros((samples, steps), dtype=np.float64)
    rets = np.zeros((samples, steps), dtype=np.float64)

    rows = np.tile(last_seq_row_norm[None, :], (samples, 1))
    base_cw = dataset.df["Close"].values[-dataset.seq_len :].astype(np.float64)
    closes_windows = [base_cw.copy() for _ in range(samples)]
    px = np.full(samples, last_close, dtype=np.float64)

    idx_return = dataset.feature_cols.index("Return")
    idx_close = dataset.feature_cols.index("Close")
    idx_open = dataset.feature_cols.index("Open")
    idx_high = dataset.feature_cols.index("High")
    idx_low = dataset.feature_cols.index("Low")
    idx_time_sin = dataset.feature_cols.index("Time_sin")
    idx_time_cos = dataset.feature_cols.index("Time_cos")
    idxs_ind = {
        k: dataset.feature_cols.index(k)
        for k in [
            "RSI",
            "BB_mavg",
            "BB_bbp",
            "BB_bw",
            "BB_above_upper",
            "BB_below_lower",
            "BB_cross_up",
            "BB_cross_down",
            "BB_cross_any",
        ]
    }

    fm = dataset.feat_mean
    fs = dataset.feat_std

    try:
        delta = pd.Timedelta(dataset.interval)
    except Exception:
        delta = pd.Timedelta("15min")
    t0 = dataset.df.index[-1]

    for s_idx, m in enumerate(replayed_models):
        t = t0 + delta
        for h in range(steps):
            x = torch.from_numpy(rows[s_idx]).float().to(device).view(1, 1, F)
            trace = poutine.trace(m).get_trace(x)
            y_norm = trace.nodes["obs"]["value"].detach().cpu().numpy().reshape(-1)[0]

            r = y_norm * ret_std + ret_mean
            px[s_idx] *= 1.0 + r
            prices[s_idx, h] = px[s_idx]
            rets[s_idx, h] = r

            cw = list(closes_windows[s_idx])
            cw.append(px[s_idx])
            cw = cw[-dataset.seq_len :]
            closes_windows[s_idx] = cw

            ind = recompute_indicators(np.asarray(cw, dtype=np.float64))
            row = rows[s_idx]

            row[idx_open] = (px[s_idx] - fm["Open"]) / fs["Open"]
            row[idx_high] = (px[s_idx] - fm["High"]) / fs["High"]
            row[idx_low] = (px[s_idx] - fm["Low"]) / fs["Low"]
            row[idx_close] = (px[s_idx] - fm["Close"]) / fs["Close"]

            seconds = t.hour * 3600 + t.minute * 60 + t.second
            row[idx_time_sin] = (np.sin(2 * np.pi * seconds / 86400.0) - fm["Time_sin"]) / fs["Time_sin"]
            row[idx_time_cos] = (np.cos(2 * np.pi * seconds / 86400.0) - fm["Time_cos"]) / fs["Time_cos"]

            row[idx_return] = (r - fm["Return"]) / fs["Return"]
            row[idxs_ind["RSI"]] = (ind["RSI"] - fm["RSI"]) / fs["RSI"]
            row[idxs_ind["BB_mavg"]] = (ind["BB_mavg"] - fm["BB_mavg"]) / fs["BB_mavg"]
            row[idxs_ind["BB_bbp"]] = (ind["BB_bbp"] - fm["BB_bbp"]) / fs["BB_bbp"]
            row[idxs_ind["BB_bw"]] = (ind["BB_bw"] - fm["BB_bw"]) / fs["BB_bw"]
            row[idxs_ind["BB_above_upper"]] = ind["BB_above_upper"]
            row[idxs_ind["BB_below_lower"]] = ind["BB_below_lower"]
            row[idxs_ind["BB_cross_up"]] = ind["BB_cross_up"]
            row[idxs_ind["BB_cross_down"]] = ind["BB_cross_down"]
            row[idxs_ind["BB_cross_any"]] = ind["BB_cross_any"]

            rows[s_idx] = row
            t = t + delta

    return prices, rets
