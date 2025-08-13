"""Metric calculator tool."""

import pandas as pd

from data_aug_4_tsc.metrics.diversity import ACPD, APD, COVERAGE, MMS, WPD
from data_aug_4_tsc.metrics.fidelity import AOG, DENSITY, FID


class METRIC_CALCULATOR:
    """Metric calculator tool.

    Parameters
    ----------
    args : dict
        The arguments containing the metrics used
        as {"metric_name" : metric_params_sub_dict}.

    Returns
    -------
    None
    """

    def __init__(self, args: dict) -> None:
        self.str_to_metric = {
            "fid": FID,
            "density": DENSITY,
            "apd": APD,
            "acpd": ACPD,
            "coverage": COVERAGE,
            "mms": MMS,
            "aog": AOG,
            "wpd": WPD,
        }

        self.metric_to_str = {
            FID: "fid",
            DENSITY: "density",
            APD: "apd",
            ACPD: "acpd",
            COVERAGE: "coverage",
            MMS: "mms",
            AOG: "aog",
            WPD: "wpd",
        }

        self.args = args

        self.used_metrics_names = list(self.args.keys())
        self.used_metrics = [
            self.str_to_metric[metric_name] for metric_name in self.used_metrics_names
        ]

        self.df_results = pd.DataFrame(columns=["On"] + self.used_metrics_names)

    def get_metrics_csv(
        self,
        xgenerated=None,
        ygenerated=None,
        xreal=None,
        yreal=None,
    ):
        """Generate a DataFrame containing the metric values.

        Parameters
        ----------
        xgenerated : np.ndarray, default = None
            The generated samples.
        ygenerated : np.ndarray, default = None
            The labels of the generated samples.
        xreal : np.ndarray, default = None
            The real samples.
        yreal : np.ndarray, default = None
            The labels of the real samples.

        Returns
        -------
        self.df_results : pd.DataFrame
            The results of the metrics on real and generated
            in a pandas DataFrame.
        """
        # on real samples
        row_to_add = {"On": "real"}

        for METRIC in self.used_metrics:
            if "metric_params" in self.args[self.metric_to_str[METRIC]].keys():
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                    **self.args[self.metric_to_str[METRIC]]["metric_params"],
                )
            else:
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                )

            metric_value = metric.calculate(
                xreal=xreal,
                yreal=yreal,
            )

            row_to_add[self.metric_to_str[METRIC]] = metric_value

        self.df_results.loc[len(self.df_results)] = row_to_add

        # on generated samples
        row_to_add = {"On": "generated"}

        for METRIC in self.used_metrics:
            if "metric_params" in self.args[self.metric_to_str[METRIC]].keys():
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                    **self.args[self.metric_to_str[METRIC]]["metric_params"],
                )
            else:
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                )

            metric_value = metric.calculate(
                xgenerated=xgenerated,
                ygenerated=ygenerated,
                xreal=xreal,
                yreal=yreal,
            )

            row_to_add[self.metric_to_str[METRIC]] = metric_value

        self.df_results.loc[len(self.df_results)] = row_to_add

        return self.df_results
