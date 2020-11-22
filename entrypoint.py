from tensorseason.experiment import SingleForecasterExperiment, TensorSeasonExperiment


if __name__ == "__main__":
    ts_experiments = TensorSeasonExperiment(
        dataset_name="traffic_75",
        folds=(24, 7),
        nr_in_cycles=70,
        nr_examples=5,
        n_jobs=1,
    )
    traffic = ts_experiments.run()

    print("done")
