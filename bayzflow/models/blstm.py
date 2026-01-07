# models/blstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
from typing import Optional


class BayesianLSTM(pnn.PyroModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        prior_scale_lstm: float = 1.0,
        prior_scale_head: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        print(
            "[BayesianLSTM.__init__] input_size=", input_size,
            "hidden_size=", hidden_size,
            "prior_scale_lstm=", prior_scale_lstm,
            "prior_scale_head=", prior_scale_head,
            "device=", device,
        )

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.hidden_size = hidden_size

        def prior_lstm(shape):
            return dist.Normal(
                torch.zeros(shape, device=self.device),
                torch.full(shape, float(prior_scale_lstm), device=self.device),
            ).to_event(len(shape))

        def prior_head(shape):
            return dist.Normal(
                torch.zeros(shape, device=self.device),
                torch.full(shape, float(prior_scale_head), device=self.device),
            ).to_event(len(shape))

        # ✅ This MUST include hidden_size
        self.lstm = pnn.PyroModule[nn.LSTM](
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        ).to(device)
        self.lstm.to(self.device)

        self.lstm.weight_ih_l0 = pnn.PyroSample(prior_lstm(self.lstm.weight_ih_l0.shape))
        self.lstm.weight_hh_l0 = pnn.PyroSample(prior_lstm(self.lstm.weight_hh_l0.shape))
        self.lstm.bias_ih_l0   = pnn.PyroSample(prior_lstm(self.lstm.bias_ih_l0.shape))
        self.lstm.bias_hh_l0   = pnn.PyroSample(prior_lstm(self.lstm.bias_hh_l0.shape))

        self.head_mean = pnn.PyroModule[nn.Linear](hidden_size, 1).to(device)
        self.head_scale = pnn.PyroModule[nn.Linear](hidden_size, 1).to(device)

        
        self.head_mean.weight = pnn.PyroSample(prior_head(self.head_mean.weight.shape))
        self.head_mean.bias   = pnn.PyroSample(prior_head(self.head_mean.bias.shape))

        self.head_scale.weight = pnn.PyroSample(prior_head(self.head_scale.weight.shape))
        self.head_scale.bias   = pnn.PyroSample(prior_head(self.head_scale.bias.shape))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        device = self.device
        x = x.to(device)
        if y is not None:
            y = y.to(device)

        batch_size, seq_len, _ = x.shape

        lstm_out, _ = self.lstm(x)      # [B, T, H]
        h_last = lstm_out[:, -1, :]     # [B, H]

        loc = self.head_mean(h_last).squeeze(-1)
        scale_raw = self.head_scale(h_last).squeeze(-1)
        sigma = 0.1 + 0.9 * F.softplus(scale_raw)

        df = torch.tensor(5.0, device=device)

        with pyro.plate("data", batch_size):
            obs = pyro.sample(
                "obs",
                dist.StudentT(df, loc=loc, scale=sigma),
                obs=y,
            )

        return loc
