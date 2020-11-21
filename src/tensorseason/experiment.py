import json
from pathlib import Path
from random import sample
from typing import List, Dict, Type, Tuple, Callable
from uuid import uuid4

import pandas as pd
import numpy as np
from tqdm import tqdm

from .forecaster import (
    CPForecaster,
    DCTForecaster,
    DFTForecaster,
    FourierBasisRegressionForecaster,
    HoltWintersForecaster,
    SeasonalForecaster,
    TuckerForecaster,
)
from .utils import (
    get_param_sweep, mad, rmse, trend_cycle_decompose
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


class TensorSeasonExperiment:
    """

    Parameters
    ----------
    dataset_name: str
        The dataset name to run the experiments on. There must be a directory under
        `datasets/` with a matching name, which contains JSON files in GluonTS data set
        format (with "start" and "target" keys).
    folds: Tuple[int]
        Number of `folds` in the multiple seasonal pattern, with the fastest index first.
        For example, (24, 7).
    nr_in_cycles: int
        Number of cycles (a cycle has np.prod(folds) length) to consider in sample.
    nr_examples: int
        `nr_examples` many time series will be sampled (without replacement) from the data set
        in order to perform experiments. If -1, all time series will be used.
    dft_sweep_length: int
        number of parameters in DFT and DCT
    tensor_sweep_length: int
        number of parameters (ranks) for tensor based methods
    n_jobs: int
        If greater than 1, joblib will be used to parallelize the experiment on `n_jobs`
        workers.
    """
    def __init__(
        self,
        dataset_name: str,
        folds: Tuple[int],
        nr_in_cycles: int,
        nr_examples: int = 10,
        dft_sweep_length: int = 100,
        tensor_sweep_length: int = 8,
        data_freq: str = "1h",
        n_jobs: int = 1,
    ) -> None:
        self.dataset_name = dataset_name
        self.nr_in_cycles = nr_in_cycles
        self.folds = folds
        self.nr_examples = nr_examples
        self.dft_sweep_length = dft_sweep_length
        self.tensor_sweep_length = tensor_sweep_length
        self.data_freq = data_freq
        self.n_jobs = n_jobs

        data_path = Path.iterdir(Path("datasets/") / self.dataset_name)
        self.data_path_list = [
            d for d in data_path if ".DS_Store" not in str(d)
        ]
        self.data_indices = sample(range(len(self.data_path_list)), nr_examples)

    def _get_dataset(self) -> List[Dict]:
        dataset = []
        for i in self.data_indices:
            with open(self.data_path_list[i]) as fp:
                dataset.append(json.load(fp))
        return dataset

    def _get_experiments(self) -> List[SingleForecasterExperiment]:
        dft_sweep = get_param_sweep(
            int(np.prod(self.folds)), "log", self.dft_sweep_length
        )
        tensor_sweep = list(range(1, self.tensor_sweep_length))
        fbm_sweep = list(range(1, 40))

        experiments = [
            SingleForecasterExperiment(DCTForecaster, dft_sweep, folds=self.folds),
            SingleForecasterExperiment(DFTForecaster, dft_sweep, folds=self.folds),
            SingleForecasterExperiment(CPForecaster, tensor_sweep, folds=self.folds),
            SingleForecasterExperiment(TuckerForecaster, tensor_sweep, folds=self.folds),
            SingleForecasterExperiment(FourierBasisRegressionForecaster, fbm_sweep, folds=self.folds),
            SingleForecasterExperiment(HoltWintersForecaster, [1], folds=self.folds),
        ]

        return experiments

    def run(self):
        def process_dataset(data, exp_id_):
            frames_ = []

            time_index = pd.date_range(
                start=pd.Timestamp(data["start"]),
                periods=len(data["target"]),
                freq=self.data_freq,
            )

            orig_data_df = pd.Series(
                data["target"],
                index=time_index,
            )
            tc, vals = trend_cycle_decompose(orig_data_df, w=int(2 * np.prod(self.folds)))
            vals = vals / (vals.max() - vals.min())  # scale the residuals

            exps_ = self._get_experiments()

            for experiment in exps_:
                ins, outs, pars, _ = experiment(vals, nr_in_cycles=self.nr_in_cycles)

                result_columns = {}
                for err in ins:
                    result_columns[f"in_{err}"] = ins[err]
                for err in outs:
                    result_columns[f"out_{err}"] = outs[err]

                results = pd.DataFrame(
                    dict(
                        parameters=pars,
                        **result_columns,
                    )
                )
                results["experiment_id"] = exp_id_
                results["dataset"] = self.dataset_name
                results["model"] = experiment.forecaster_class.__name__
                results["data_id"] = data["id"]

                frames_.append(results)

            return pd.concat(frames_)

        dataset = self._get_dataset()
        experiment_id = str(uuid4())

        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            frames = Parallel(n_jobs=self.n_jobs)(
                delayed(process_dataset)(dataset[i], experiment_id) for i in tqdm(range(len(dataset)))
            )
        else:
            frames = [process_dataset(data, experiment_id) for data in tqdm(dataset)]

        return pd.concat(frames)
