#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from examples.fx.fx_utils.forecast import forecast_paths_primary_only
from core import Bayzflow


def main():
    bf = Bayzflow(exp=None, config_path="bayzflow.yaml")  # default_experiment fx

    # Build model + dataset + wrapper
    model, dataset, wrapper = bf.load_experiment()

    # Train
    history = bf.fit()

    # Posterior predictive on test set
    y_samp_ret = bf.predict()  # [S, T_test]

    p_up_test = (y_samp_ret > 0).mean(axis=0)
    price_paths = dataset.price0 * np.cumprod(1.0 + y_samp_ret, axis=1)

    px_q05 = np.quantile(price_paths, 0.05, axis=0)
    px_q50 = np.quantile(price_paths, 0.50, axis=0)
    px_q95 = np.quantile(price_paths, 0.95, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(dataset.df.index, dataset.df["Close"].values, alpha=0.4, label="Actual Close")
    plt.plot(dataset.tidx_test, px_q50, label="Predictive median (test)")
    plt.fill_between(dataset.tidx_test, px_q05, px_q95, alpha=0.2, label="90% credible band")
    plt.title(f"Out-of-sample price distribution for {dataset.ticker}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"OOS P(up) at last test point: {float(p_up_test[-1]):.3f}")

    # Multi-step forecast
    px_paths, r_paths = bf.forecast()
    px_q05_f = np.quantile(px_paths, 0.05, axis=0)
    px_q50_f = np.quantile(px_paths, 0.50, axis=0)
    px_q95_f = np.quantile(px_paths, 0.95, axis=0)
    p_up_future = (r_paths > 0).mean(axis=0)

    from pandas import Timedelta, date_range

    delta = Timedelta(dataset.interval)
    f_times = date_range(start=dataset.df.index[-1] + delta,
                         periods=px_paths.shape[1],
                         freq=delta)

    plt.figure(figsize=(12, 6))
    plt.plot(dataset.df.index, dataset.df["Close"].values, alpha=0.4, label="Actual Close")
    plt.plot(f_times, px_q50_f, label="Forecast median")
    plt.fill_between(f_times, px_q05_f, px_q95_f, alpha=0.2, label="90% credible band")
    plt.title(f"{px_paths.shape[1]}-step Forecast Distribution for {dataset.ticker}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Future P(up) per step:", np.round(p_up_future, 3))


if __name__ == "__main__":
    main()
