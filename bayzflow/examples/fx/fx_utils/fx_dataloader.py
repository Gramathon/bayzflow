# fx_utils/fx_dataloader.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import ta


class FXDataset:
    """
    FX time-series dataset with engineered features, scaling, and
    train/val/test splits in sequence space for an LSTM.

    Exposes:
      - df, df_scaled
      - feature_cols, continuous_cols, binary_cols
      - feat_mean, feat_std
      - X_train, y_train, X_val, y_val, X_test, y_test
      - price0, tidx_test
      - ret_mean_train, ret_std_train
      - train_loader(batch_size)
      - meta_for_checkpoint()
    """

    def __init__(
        self,
        ticker: str,
        period: str,
        interval: str,
        seq_len: int,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.seq_len = seq_len
        self.train_frac = train_frac
        self.val_frac = val_frac

        # Fetch + features
        self.df = self._fetch_and_features(ticker, period, interval)

        # Define feature sets
        self.continuous_cols = [
            "Return",
            "RSI",
            "Open",
            "High",
            "Low",
            "Close",
            "Time_sin",
            "Time_cos",
            "BB_mavg",
            "BB_bbp",
            "BB_bw",
            "Ret2_5",
            "Ret2_20",
            "ATR_ratio",
            "RangePct",
            "DOW_sin",
            "DOW_cos",
            "MOY_sin",
            "MOY_cos",
        ]
        self.binary_cols = [
            "BB_above_upper",
            "BB_below_lower",
            "BB_cross_up",
            "BB_cross_down",
            "BB_cross_any",
            "Sess_Tokyo",
            "Sess_London",
            "Sess_NY",
        ]
        self.feature_cols = self.continuous_cols + self.binary_cols

        # Split in raw time domain
        n = len(self.df)
        n_train_raw = int(n * train_frac)
        n_val_raw = int(n * val_frac)
        n_test_raw = n - n_train_raw - n_val_raw

        # Fit scalers on TRAIN-only slice
        self.feat_mean = self.df.iloc[:n_train_raw][self.continuous_cols].mean()
        self.feat_std = (
            self.df.iloc[:n_train_raw][self.continuous_cols].std().replace(0, 1.0)
        )

        # Scale
        self.df_scaled = self.df.copy()
        self.df_scaled[self.continuous_cols] = (
            self.df_scaled[self.continuous_cols] - self.feat_mean
        ) / self.feat_std

        data = self.df_scaled[self.feature_cols].values
        self.ret_mean_train = self.feat_mean["Return"]
        self.ret_std_train = self.feat_std["Return"]

        if len(data) <= seq_len:
            raise RuntimeError(f"Not enough rows ({len(data)}) for seq_len={seq_len}")

        # Build rolling windows in sequence space
        X_np = np.stack(
            [data[i : i + seq_len] for i in range(len(data) - seq_len)]
        )  # [T-SEQ, L, F]
        y_np = data[seq_len:, self.feature_cols.index("Return")]  # [T-SEQ]

        N = len(X_np)
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)

        self.X_train_np, self.y_train_np = X_np[:n_train], y_np[:n_train]
        self.X_val_np, self.y_val_np = (
            X_np[n_train : n_train + n_val],
            y_np[n_train : n_train + n_val],
        )
        self.X_test_np, self.y_test_np = X_np[n_train + n_val :], y_np[n_train + n_val :]

        # Convert to tensors
        self.X_train = torch.from_numpy(self.X_train_np).float().to(self.device)
        self.y_train = torch.from_numpy(self.y_train_np).float().to(self.device)
        self.X_val = torch.from_numpy(self.X_val_np).float().to(self.device)
        self.y_val = torch.from_numpy(self.y_val_np).float().to(self.device)
        self.X_test = torch.from_numpy(self.X_test_np).float().to(self.device)
        self.y_test = torch.from_numpy(self.y_test_np).float().to(self.device)

        # Test indexing for plotting price paths
        T_test = len(self.y_test_np)
        seq_start_test = n_train + n_val
        self.price0 = self.df["Close"].values[(seq_start_test + seq_len) - 1]

        start_idx = seq_start_test + seq_len
        end_idx = start_idx + T_test
        self.tidx_test = self.df.index[start_idx:end_idx]
        assert len(self.tidx_test) == T_test, (len(self.tidx_test), T_test)

    # ---------------- feature engineering ----------------
    @staticmethod
    def _fetch_and_features(symbol: str, period: str, interval: str) -> pd.DataFrame:
        df = yf.Ticker(symbol).history(
            period=period, interval=interval, prepost=False, auto_adjust=False
        )
        if df.empty:
            raise RuntimeError(
                f"No data for {symbol} with period={period}, interval={interval}"
            )

        df = df.tz_convert("UTC").copy()
        df["Return"] = df["Close"].pct_change()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Time-of-day
        seconds = (
            df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        ).astype(float)
        df["Time_sin"] = np.sin(2 * np.pi * seconds / 86400.0)
        df["Time_cos"] = np.cos(2 * np.pi * seconds / 86400.0)

        # Bollinger
        bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_mavg"] = bb.bollinger_mavg()
        df["BB_hband"] = bb.bollinger_hband()
        df["BB_lband"] = bb.bollinger_lband()
        df["BB_bbp"] = bb.bollinger_pband()
        df["BB_bw"] = bb.bollinger_wband()

        df["BB_above_upper"] = (df["Close"] > df["BB_hband"]).astype(int)
        df["BB_below_lower"] = (df["Close"] < df["BB_lband"]).astype(int)
        df["BB_cross_up"] = (
            (df["Close"].shift(1) <= df["BB_hband"].shift(1))
            & (df["Close"] > df["BB_hband"])
        ).astype(int)
        df["BB_cross_down"] = (
            (df["Close"].shift(1) >= df["BB_lband"].shift(1))
            & (df["Close"] < df["BB_lband"])
        ).astype(int)
        df["BB_cross_any"] = (df["BB_cross_up"] | df["BB_cross_down"]).astype(int)

        df["Ret2_5"] = df["Return"].rolling(5).apply(
            lambda x: np.sqrt((x**2).sum()), raw=True
        )
        df["Ret2_20"] = df["Return"].rolling(20).apply(
            lambda x: np.sqrt((x**2).sum()), raw=True
        )

        tr = (df["High"] - df["Low"]).abs()
        tr1 = (df["High"] - df["Close"].shift()).abs()
        tr2 = (df["Low"] - df["Close"].shift()).abs()
        df["TR"] = pd.concat([tr, tr1, tr2], axis=1).max(axis=1)
        df["ATR14"] = df["TR"].rolling(14).mean()
        df["ATR_ratio"] = df["ATR14"] / df["Close"]

        df["RangePct"] = (df["High"] - df["Low"]) / df["Close"]

        h = df.index.hour
        df["Sess_Tokyo"] = ((h >= 0) & (h < 8)).astype(int)
        df["Sess_London"] = ((h >= 8) & (h < 16)).astype(int)
        df["Sess_NY"] = ((h >= 13) & (h < 21)).astype(int)

        df["DOW_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["DOW_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df["MOY_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
        df["MOY_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)

        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[price_cols] = df[price_cols].ffill()
        df = df.dropna()
        df = df.tz_localize(None)
        print(df)
        return df

    # ---------------- loader + meta ----------------
    def train_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        ds = TensorDataset(self.X_train, self.y_train)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def meta_for_checkpoint(self) -> dict:
        return {
            "feature_cols": self.feature_cols,
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
            "feat_mean": self.feat_mean.to_dict(),
            "feat_std": self.feat_std.to_dict(),
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "seq_len": self.seq_len,
            "ret_mean_train": float(self.ret_mean_train),
            "ret_std_train": float(self.ret_std_train),
        }
