from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorly as tl
from gluonts.dataset.util import to_pandas


from tens_utils import (
    get_gluonts_dataset,
    mad,
    rmse,
    get_param_sweep,
    trend_cycle_decompose,
)

from forecaster import (
    DCTForecaster, DFTForecaster, HoltWintersForecaster, CPForecaster, TuckerForecaster
)
from experiment import SingleForecasterExperiment, TensorSeasonExperiment


if __name__ == "__main__":
    exp = TensorSeasonExperiment(
        dataset_name="traffic_75",
        folds=(24, 7),
        nr_in_cycles=70,
        nr_examples=5,
    )
    traffic = exp._get_dataset()

    tg = traffic[0]['target']

    hw = HoltWintersForecaster(folds=(24, 7))

    res = hw(pd.Series(tg), nr_in_cycles=70)

    print(res.in_errors)