"""Main file to run."""

import os
import warnings

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score

from data_aug_4_tsc.data_augs import _get_data_augmentation_function
from data_aug_4_tsc.feature_extractors import LITE_CLASSIFIER
from data_aug_4_tsc.utils import (
    create_directory,
    load_data_aeon,
    plot_generated_only,
    plot_generated_with_nn,
    plot_parallel_axes,
    plot_same_axes,
)
from metric_calculator import METRIC_CALCULATOR

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@hydra.main(config_name="config_hydra.yaml", config_path="config")
def main(args: DictConfig):
    """Run experiments.

    Main function to run experiments.

    Parameters
    ----------
    args: DictConfig
        The input configuration.

    Returns
    -------
    None
    """
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    xtrain, ytrain, xtest, ytest = load_data_aeon(file_name=args.dataset_name)

    output_directory = args.output_directory
    create_directory(output_directory)
    output_directory_task = os.path.join(output_directory, args.task)
    create_directory(output_directory_task)
    output_directory_dataset = os.path.join(output_directory_task, args.dataset_name)
    create_directory(output_directory_dataset)

    if args.task == "evaluate_generation":

        _folder_suffix = ""

        if (
            args.evaluate_generation.method == "Scaling"
            or args.evaluate_generation.method == "Jittering"
            or args.evaluate_generation.method == "AW"
        ):
            output_directory_generation_method = os.path.join(
                output_directory_dataset, args.evaluate_generation.method
            )
            create_directory(output_directory_generation_method)
        elif args.evaluate_generation.method == "WW":
            if (
                args.evaluate_generation.window_size_ratio is None
                and args.evaluate_generation.warp_scale is None
                and args.evaluate_generation.window_start is None
            ):
                output_directory_generation_method = os.path.join(
                    output_directory_dataset, args.evaluate_generation.method
                )
                create_directory(output_directory_generation_method)
            else:
                _folder_suffix = "_window_size_ratio=" + str(
                    args.evaluate_generation.window_size_ratio
                )
                _folder_suffix = (
                    _folder_suffix
                    + "_warp_scale="
                    + str(args.evaluate_generation.warp_scale)
                )
                _folder_suffix = (
                    _folder_suffix
                    + "_window_start="
                    + str(args.evaluate_generation.window_start)
                )

                output_directory_generation_method = os.path.join(
                    output_directory_dataset,
                    args.evaluate_generation.method + _folder_suffix,
                )
                create_directory(output_directory_generation_method)
        elif (
            args.evaluate_generation.method == "RGW"
            or args.evaluate_generation.method == "DGW"
            or args.evaluate_generation.method == "WBA"
        ):

            if args.evaluate_generation.distance == "dtw":
                distance_params = {
                    "window": args.evaluate_generation.window,
                }
            elif args.evaluate_generation.distance == "shape_dtw":
                distance_params = {
                    "reach": args.evaluate_generation.reach,
                }
            elif args.evaluate_generation.distance == "msm":
                if args.evaluate_generation.c == 2.0:
                    distance_params = {"c": 2}
                else:
                    distance_params = {"c": args.evaluate_generation.c}
            else:
                raise ValueError("Supported distances are: dtw, shape_dtw and msm")

            _folder_suffix = (
                "_distance="
                + args.evaluate_generation.distance
                + "_"
                + "_".join(f"{k}={v}" for k, v in distance_params.items())
            )
            output_directory_generation_method = os.path.join(
                output_directory_dataset,
                args.evaluate_generation.method + _folder_suffix,
            )
            create_directory(output_directory_generation_method)

        output_directory_feature_extractor = os.path.join(
            output_directory_generation_method,
            args.evaluate_generation.feature_extractor.estimator,
        )
        create_directory(output_directory_feature_extractor)

        for _gen in range(args.evaluate_generation.n_generations):
            output_directory_generation_run = os.path.join(
                output_directory_feature_extractor, "gen_" + str(_gen)
            )
            create_directory(output_directory_generation_run)

            xtrain_generated = np.load(
                os.path.join(
                    output_directory,
                    "generate_data",
                    args.dataset_name,
                    args.evaluate_generation.method + _folder_suffix,
                    "gen_" + str(_gen),
                    "xtrain_generated.npy",
                )
            )

            ytrain_generated = np.load(
                os.path.join(
                    output_directory,
                    "generate_data",
                    args.dataset_name,
                    args.evaluate_generation.method + _folder_suffix,
                    "gen_" + str(_gen),
                    "ytrain_generated.npy",
                )
            )

            for _run in range(args.evaluate_generation.feature_extractor.runs):
                output_directory_feature_extractor_run = os.path.join(
                    output_directory_generation_run, "run_" + str(_run)
                )
                create_directory(output_directory_feature_extractor_run)

                if os.path.exists(
                    os.path.join(
                        output_directory_feature_extractor_run,
                        "generative_metrics.csv",
                    )
                ):
                    continue

                feature_extractor_estimator = tf.keras.models.load_model(
                    os.path.join(
                        output_directory,
                        "train_feature_extractor",
                        args.dataset_name,
                        args.evaluate_generation.feature_extractor.estimator,
                        "run_" + str(_run),
                        "best_model.keras",
                    ),
                    compile=False,
                )

                args_metric_claculator = {
                    "fid": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "density": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "apd": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                        "metric_params": {"Sapd": 200},
                    },
                    "acpd": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "coverage": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "aog": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "mms": {
                        "classifier": feature_extractor_estimator,
                        "batch_size": 64,
                    },
                    "wpd": {
                        "metric_params": {"Swpd": 200},
                    },
                }

                metric_calculator = METRIC_CALCULATOR(args=args_metric_claculator)

                df = metric_calculator.get_metrics_csv(
                    xgenerated=xtrain_generated,
                    ygenerated=ytrain_generated,
                    xreal=xtrain,
                    yreal=ytrain,
                )

                df.to_csv(
                    os.path.join(
                        output_directory_feature_extractor_run,
                        "generative_metrics.csv",
                    ),
                    index=False,
                )

    elif args.task == "train_feature_extractor":

        output_directory_feature_extractor = os.path.join(
            output_directory_dataset, args.train_feature_extractor.estimator
        )

        for _run in range(args.train_feature_extractor.runs):
            output_directory_feature_extractor_run = os.path.join(
                output_directory_feature_extractor, "run_" + str(_run) + "/"
            )

            if os.path.exists(
                os.path.join(output_directory_feature_extractor_run, "metrics.csv")
            ):
                continue

            if args.train_feature_extractor.estimator == "LITE":
                estimator = LITE_CLASSIFIER(
                    output_directory=output_directory_feature_extractor_run,
                    n_epochs=args.train_feature_extractor.n_epochs,
                    batch_size=args.train_feature_extractor.batch_size,
                )
            else:
                raise NotImplementedError(
                    f"Such model {args.train_feature_extractor.estimator}\
                    is not implemented."
                )

            estimator.fit(X=xtrain, y=ytrain)

            model = tf.keras.models.load_model(
                os.path.join(output_directory_feature_extractor_run, "best_model.keras")
            )

            ytrain_pred_probas = model.predict(xtrain)
            ytrain_pred = np.argmax(ytrain_pred_probas, axis=1)
            accuracy_train = accuracy_score(
                y_pred=ytrain_pred, y_true=ytrain, normalize=True
            )

            ytest_pred_probas = model.predict(xtest)
            ytest_pred = np.argmax(ytest_pred_probas, axis=1)
            accuracy_test = accuracy_score(
                y_pred=ytest_pred, y_true=ytest, normalize=True
            )

            df = pd.DataFrame(columns=["accuracy-train", "accuracy-test"])
            df.loc[len(df)] = {
                "accuracy-train": accuracy_train,
                "accuracy-test": accuracy_test,
            }

            df.to_csv(
                os.path.join(output_directory_feature_extractor_run, "metrics.csv"),
                index=False,
            )

            tf.keras.backend.clear_session()

    elif args.task == "generate_data":
        if (
            args.generate_data.method == "Scaling"
            or args.generate_data.method == "Jittering"
            or args.generate_data.method == "AW"
        ):
            output_directory_method = os.path.join(
                output_directory_dataset, args.generate_data.method
            )
            create_directory(output_directory_method)
        elif args.generate_data.method == "WW":
            if (
                args.generate_data.window_size_ratio is None
                and args.generate_data.warp_scale is None
                and args.generate_data.window_start is None
            ):
                output_directory_method = os.path.join(
                    output_directory_dataset, args.generate_data.method
                )
                create_directory(output_directory_method)
            else:
                _folder_suffix = "_window_size_ratio=" + str(
                    args.generate_data.window_size_ratio
                )
                _folder_suffix = (
                    _folder_suffix + "_warp_scale=" + str(args.generate_data.warp_scale)
                )
                _folder_suffix = (
                    _folder_suffix
                    + "_window_start="
                    + str(args.generate_data.window_start)
                )

                output_directory_method = os.path.join(
                    output_directory_dataset,
                    args.generate_data.method + _folder_suffix,
                )
                create_directory(output_directory_method)
        elif (
            args.generate_data.method == "RGW"
            or args.generate_data.method == "DGW"
            or args.generate_data.method == "WBA"
        ):

            if args.generate_data.distance == "dtw":
                distance_params = {
                    "window": args.generate_data.window,
                }
            elif args.generate_data.distance == "shape_dtw":
                distance_params = {
                    "reach": args.generate_data.reach,
                }
            elif args.generate_data.distance == "msm":
                distance_params = {"c": args.generate_data.c}
            else:
                raise ValueError("Supported distances are: dtw, shape_dtw and msm")

            _folder_suffix = (
                "_distance="
                + args.generate_data.distance
                + "_"
                + "_".join(f"{k}={v}" for k, v in distance_params.items())
            )
            output_directory_method = os.path.join(
                output_directory_dataset,
                args.generate_data.method + _folder_suffix,
            )
            create_directory(output_directory_method)

        for _gen in range(args.generate_data.n_generations):
            output_directory_generation = os.path.join(
                output_directory_method, "gen_" + str(_gen)
            )
            create_directory(output_directory_generation)

            if os.path.exists(
                os.path.join(output_directory_generation, "xtrain_generated.npy")
            ):
                continue

            data_augmentaiton_function = _get_data_augmentation_function(
                method=args.generate_data.method
            )

            if args.generate_data.method in ["RGW", "DGW"]:
                xtrain_generated = data_augmentaiton_function(
                    xtrain,
                    ytrain,
                    distance=args.generate_data.distance,
                    window=args.generate_data.window,
                    reach=args.generate_data.reach,
                    c=args.generate_data.c,
                )
            elif args.generate_data.method in ["WBA"]:
                xtrain_generated = data_augmentaiton_function(
                    xtrain,
                    ytrain,
                    distance=args.generate_data.distance,
                    distance_params=distance_params,
                )
            elif args.generate_data.method == "WW":
                xtrain_generated = data_augmentaiton_function(
                    xtrain,
                    ytrain,
                    window_size_ratio=args.generate_data.window_size_ratio,
                    warp_scale=args.generate_data.warp_scale,
                    window_start=args.generate_data.window_start,
                )
            else:
                xtrain_generated = data_augmentaiton_function(xtrain, ytrain)

            np.save(
                file=os.path.join(output_directory_generation, "xtrain_generated.npy"),
                arr=xtrain_generated,
            )
            np.save(
                file=os.path.join(output_directory_generation, "ytrain_generated.npy"),
                arr=ytrain,
            )

            for c in range(len(np.unique(ytrain))):
                output_directory_class_generated = os.path.join(
                    output_directory_generation, "class_" + str(c)
                )
                create_directory(output_directory_class_generated)
                xtrain_generated_c = xtrain_generated[ytrain == c]
                xtrain_c = xtrain[ytrain == c]

                plot_generated_only(
                    output_directory=output_directory_class_generated,
                    file_name="generated_only",
                    xgenerated=xtrain_generated_c,
                )
                plot_parallel_axes(
                    output_directory=output_directory_class_generated,
                    file_name="parallel",
                    xgenerated=xtrain_generated_c,
                    xreal=xtrain_c,
                )
                plot_same_axes(
                    output_directory=output_directory_class_generated,
                    file_name="same_axes",
                    xgenerated=xtrain_generated_c,
                    xreal=xtrain_c,
                )
                plot_generated_with_nn(
                    output_directory=output_directory_class_generated,
                    file_name="with_nn",
                    xgenerated=xtrain_generated_c,
                    xreal=xtrain_c,
                )


if __name__ == "__main__":
    main()
