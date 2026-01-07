#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from bayzflow import BayzFlow
from bayzflow.models import BayesianLSTM
from bayzflow.examples.fx.fx_utils import FXDataset


def main():

    bf = BayzFlow.exp("bayzflow/bayzflow.yaml", exp= "fx_blstm_jpyx")  

    

if __name__ == "__main__":
    main()
