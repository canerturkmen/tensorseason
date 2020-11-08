from typing import List, Dict, Type, Tuple, Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorly as tl
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from scipy import interpolate
from scipy.stats import linregress
from scipy.fftpack import rfft, irfft, dct, idct
from tensorly.decomposition import parafac, tucker

from forecaster import SeasonalForecaster
from tens_utils import (
    get_gluonts_dataset,
    multifold,
    repeat,
    dct_dft_errors,
    tensor_errors,
    trend_cycle_decompose,
    naive_seasonal_decompose,
    analyze_and_plot,
    plot_comparison,
    mad,
    rmse, get_param_sweep, dct_reconstruct, dft_reconstruct,
)


class SingleForecasterExperiment:

    def __init__(
        self,
        forecaster_class: Type[SeasonalForecaster],
        param_sweep: List[int],
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad)
    ):
        self.fcaster_getter = lambda x: forecaster_class(
            nr_params=x, folds=folds, error_callbacks=error_callbacks
        )
        self.param_sweep = param_sweep
        self.error_callbacks = error_callbacks
        self.forecaster_class = forecaster_class

    def __call__(self, vals: pd.Series, nr_in_cycles: int) -> Tuple[Dict, Dict, List, List]:

        in_errors, out_errors, results = [], [], []
        total_params = []

        for n in self.param_sweep:
            fcaster = self.fcaster_getter(n)
            result = fcaster(vals, nr_in_cycles=nr_in_cycles)

            results.append(result)
            in_errors.append(result.in_errors)
            out_errors.append(result.out_errors)
            total_params.append(result.nr_total_params)

        in_errors_dict, out_errors_dict = [
            dict(
                zip(
                    [f.__name__ for f in self.error_callbacks],
                    zip(*errors)
                )
            ) for errors in [in_errors, out_errors]
        ]

        return in_errors_dict, out_errors_dict, total_params, results
