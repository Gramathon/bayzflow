# core/bayzflow.py

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Tuple

import torch
import pyro
import yaml
from examples.fx.fx_utils.forecast import forecast_paths_primary_only
from .wrapper import BayesianWrapper


class Bayzflow:
    """
    High-level orchestrator that loads an experiment config,
    builds the dataset, model, guide, and wrapper, and exposes
    fit(), predict(), forecast().

    This version keeps ALL original behaviour,
    but adds clean YAML device support:

        device: "auto"
        device: "cpu"
        device: "cuda"
        device: "cuda:1"
    """

    def __init__(self, exp: str | None = None, config_path="bayzflow.yaml", device=None):
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            full = yaml.safe_load(f)["bayzflow"]

        self.full_cfg = full

        # choose experiment
        if exp is None:
            exp = full["default_experiment"]
        self.exp_name = exp
        self.exp_cfg = full["experiments"][exp]

        # ------------------------------
        # Device resolution priority:
        #   1) CLI override
        #   2) YAML experiment.device
        #   3) auto fallback
        # ------------------------------
        if device is not None:
            # explicit CLI override
            self.device = torch.device(device)
        else:
            yaml_dev = self.exp_cfg.get("device", "auto")
            self.device = self._resolve_device(yaml_dev)

        print(f"[Bayzflow] Using device: {self.device}")

        self.dataset = None
        self.model = None
        self.wrapper: BayesianWrapper | None = None

    # --------------------------------------------------------
    # Device resolution helper
    # --------------------------------------------------------
    def _resolve_device(self, dev_field: Any) -> torch.device:
        """
        YAML values accepted:
            auto
            cpu
            cuda
            cuda:0
            cuda:1
            cuda:N
        """
        if not isinstance(dev_field, str):
            dev_field = str(dev_field)

        dev = dev_field.lower()

        # Auto-select GPU if available
        if dev == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if dev == "cpu":
            return torch.device("cpu")

        if dev == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                print("[Bayzflow] CUDA requested but not available, falling back to CPU.")
                return torch.device("cpu")

        if dev.startswith("cuda:"):
            if torch.cuda.is_available():
                return torch.device(dev)
            else:
                print(f"[Bayzflow] {dev} requested but CUDA unavailable → CPU.")
                return torch.device("cpu")

        raise ValueError(f"Invalid device string in YAML: {dev_field}")

    # ---------------- utils ----------------
    def _import(self, dotted: str):
        """
        Import "a.b.c.ClassName" and return ClassName.
        """
        module_name, cls_name = dotted.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name)

    # ---------------- build experiment ----------------
    # core/bayzflow.py (inside class Bayzflow)

    def load_experiment(self):
        """
        Build dataset, model and wrapper from the selected experiment config.
        Returns (model, dataset, wrapper).
        """
        data_cfg = self.exp_cfg["data"]
        model_cfg = self.exp_cfg["model"]
        guide_cfg = self.exp_cfg.get("guide", {})

        print(f"[Bayzflow] building experiment: {self.exp_name}")
        print(f"[Bayzflow] using device: {self.device}")

        # ----------------- dataset -----------------
        DatasetClass = self._import(data_cfg["module"])
        ds_kwargs = {k: v for k, v in data_cfg.items() if k != "module"}

        # let YAML override, otherwise use Bayzflow's device
        ds_kwargs.setdefault("device", self.device)

        print(f"[Bayzflow] Dataset class: {DatasetClass.__name__}")
        print(f"[Bayzflow] Dataset kwargs: {ds_kwargs}")

        self.dataset = DatasetClass(**ds_kwargs)

        input_size = len(self.dataset.feature_cols)
        print(f"[Bayzflow] dataset.feature_cols -> input_size = {input_size}")

        # ----------------- model -------------------
        ModelClass = self._import(model_cfg["module"])
        m_kwargs = {k: v for k, v in model_cfg.items() if k not in ["module", "type"]}

        # again, YAML can supply device; otherwise use Bayzflow's
        m_kwargs.setdefault("device", self.device)

        print(f"[Bayzflow] Model class: {ModelClass.__name__}")
        print(f"[Bayzflow] Model kwargs (before ctor): {m_kwargs}")

        # 🔥 this is where hidden_size is passed in
        self.model = ModelClass(
            input_size=input_size,
            **m_kwargs,
        ).to(self.device)

        # ----------------- wrapper -----------------
        guide_kind = guide_cfg.get("kind", "auto_lowrank")
        guide_kwargs = guide_cfg.get("kwargs", {})

        print(f"[Bayzflow] Guide kind: {guide_kind}, kwargs: {guide_kwargs}")

        self.wrapper = BayesianWrapper(
            model=self.model,
            pyro_model=self.model,  # model.forward is the Pyro model
            device=self.device,
            default_guide_kind=guide_kind,
            default_guide_kwargs=guide_kwargs,
        )

        return self.model, self.dataset, self.wrapper


    # ---------------- training ----------------
    def fit(self):
        """
        Train using config-defined params and early stopping.
        """
        if self.dataset is None or self.model is None or self.wrapper is None:
            self.load_experiment()

        train_cfg = self.exp_cfg["training"]

        train_loader = self.dataset.train_loader(batch_size=train_cfg["batch_size"])

        if train_cfg.get("svi_loss", "trace_elbo").lower() == "trace_elbo":
            svi_loss = pyro.infer.Trace_ELBO(
                retain_graph=train_cfg.get("retain_graph", False)
            )
        else:
            raise ValueError("Unsupported svi_loss in config.")

        ckpt_cfg = self.exp_cfg["checkpoints"]
        ckpt_path = Path(ckpt_cfg["outdir"]) / ckpt_cfg["best_name"]
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        history = self.wrapper.fit_with_early_stopping(
            train_loader=train_loader,
            x_val=self.dataset.X_val,
            y_val=self.dataset.y_val,
            num_epochs=train_cfg["epochs"],
            lr=train_cfg["lr"],
            patience=train_cfg["patience"],
            svi_loss=svi_loss,
            checkpoint_path=str(ckpt_path),
            extra_meta=self.dataset.meta_for_checkpoint(),
            progress=True,
        )
        return history

    # ---------------- posterior predictive ----------------
    def predict(self, num_samples: int | None = None):
        if self.dataset is None or self.wrapper is None:
            self.load_experiment()

        if num_samples is None:
            num_samples = self.exp_cfg["posterior"]["num_samples"]

        with torch.no_grad():
            samples = self.wrapper.sample_posterior(
                self.dataset.X_test,
                y=None,
                num_samples=num_samples,
                return_sites=["obs"],
            )

        y_samp_norm = samples["obs"].detach().cpu().numpy()
        y_samp_ret = (
            y_samp_norm * self.dataset.ret_std_train + self.dataset.ret_mean_train
        )

        return y_samp_ret

    # ---------------- multi-step forecast ----------------
    def forecast(self, steps: int | None = None, samples: int | None = None):
        """
        Multi-step replay forecast for FX experiments.
        Delegates to fx_utils.forecast.forecast_paths_primary_only.
        """
        if self.exp_cfg["kind"] != "fx":
            raise ValueError("forecast() currently only implemented for kind='fx'.")

        if self.dataset is None or self.wrapper is None or self.model is None:
            self.load_experiment()

        from examples import forecast_paths_primary_only

        post_cfg = self.exp_cfg["posterior"]
        if steps is None:
            steps = post_cfg["forecast_steps"]
        if samples is None:
            samples = post_cfg["num_samples"]

        last_row_norm = self.dataset.df_scaled[self.dataset.feature_cols].values[-1]
        last_close = self.dataset.df["Close"].values[-1]

        px_paths, r_paths = forecast_paths_primary_only(
            model=self.model,
            guide=self.wrapper.guide,
            dataset=self.dataset,
            last_seq_row_norm=last_row_norm,
            last_close=last_close,
            ret_mean=self.dataset.ret_mean_train,
            ret_std=self.dataset.ret_std_train,
            steps=steps,
            samples=samples,
        )

        return px_paths, r_paths
