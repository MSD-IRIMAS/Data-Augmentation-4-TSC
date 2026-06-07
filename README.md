> ⚠️ **Alert:** If you are using this code with **Keras v3**, make sure you are using **Keras ≥ 3.6.0**.
> Earlier versions of Keras v3 do not honor `trainable=False`, which will result in **training hand-crafted filters** in **LITEMV** unexpectedly.

| Overview        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**       | [![github-actions-main](https://github.com/MSD-IRIMAS/Data-Augmentation-4-TSC/actions/workflows/pytest.yml/badge.svg?branch=main&logo=github&label=build%20(main))](https://github.com/MSD-IRIMAS/Data-Augmentation-4-TSC/actions/workflows/pytest.yml) [![github-actions-tests](https://github.com/MSD-IRIMAS/Data-Augmentation-4-TSC/actions/workflows/pre-commit.yml/badge.svg?logo=github&label=build%20(tests))](https://github.com/MSD-IRIMAS/Data-Augmentation-4-TSC/actions/workflows/pre-commit.yml)  |
| **Code**        | [![pypi](https://img.shields.io/pypi/v/data-aug-4-tsc?logo=pypi&color=blue)](https://pypi.org/project/data-aug-4-tsc/) [![python-versions](https://img.shields.io/pypi/pyversions/data-aug-4-tsc?logo=python)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![license](https://img.shields.io/badge/license-GPL3.0-green)](https://github.com/MSD-IRIMAS/Data-Augmentation-4-TSC/blob/main/LICENSE) |
| **Community**   | [![website](https://img.shields.io/static/v1?label=Website&message=msd-irimas.github.io&color=blue&logo=githubpages)](https://msd-irimas.github.io/msd-irimas.github.io/)


# Re-framing Time Series Augmentation Through the Lens of Generative Models

Authors: [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/)<sup>1</sup>, [Maxime Devanne](https://maxime-devanne.com/)<sup>1</sup>, [Stefano Berreti](https://www.micc.unifi.it/berretti/)<sup>2</sup>, [Jonathan Weber](https://www.jonathan-weber.eu/)<sup>1</sup> and [Germain Forestier](https://germain-forestier.info/)<sup>1,3</sup>

<sup>1</sup> IRIMAS, Universite de Haute-Alsace, France<br>
<sup>2</sup> MICC, University of Florence, Italy<br>
<sup>3</sup> DSAI, Monash University, Australia

This repository is the source code of the article titled "[Re-framing Time Series Augmentation Through the Lens of Generative Models](#)" accepted in the [10th Workshop on Advanced Analytics and Learning on Temporal Data (AALTD 2025)](https://ecml-aaltd.github.io/aaltd2025/) in conjunction with the [2025 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2025)](https://ecmlpkdd.org/2025/).
In this article, we present a benchmark comparison between 22 data augmentation techniques on 131 time series classification datasets of the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

<img id="img-overview" src="https://raw.githubusercontent.com/MSD-IRIMAS/Data-Augmentation-4-TSC/refs/heads/main/static/summary-methods.png" class="interpolation-image" style="width: 100%; height: 100%; border: none;"> </img>

## Abstract

Time series classification is widely used in many fields, but it often suffers from a lack of labeled data. To address this, researchers commonly apply data augmentation techniques that generate synthetic samples through transformations such as jittering, warping, or resampling. However, with an increasing number of available augmentation methods, it becomes difficult to choose the most suitable one for a given task. In many cases, this choice is based on intuition or visual inspection. Assessing the impact of this choice on classification accuracy requires training models, which is time-consuming and depends on the dataset. In this work, we adopt a generative model perspective and evaluate augmentation methods prior to training any classifier, using metrics that quantify both fidelity and diversity of the generated samples. We benchmark 22 augmentation techniques on 131 public datasets using eight metrics. Our results provide a practical and efficient way to compare augmentation methods without relying solely on classifier performance.

## Data

In this work we utilize 131 datasets of the UCR archive taken from the [original repository](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) and the [new added datasets](https://link.springer.com/content/pdf/10.1007/s10618-024-01022-1.pdf).

However you are not obligated to download them as our code loads the datasets through the [Time Series Classification webpage](https://timeseriesclassification.com/) using [aeon-toolkit](https://aeon-toolkit.org/).

## Docker

This repository supports the usage of docker. In order to create the docker image using the [dockerfile](dockerfile), simply run the following command (assuming you have docker installed and nvidia cuda container as well):
```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t data-augmentation-review-image .
```
After the image has been successfully built, you can create the docker container using the following command:
```bash
docker run --gpus all -it --name data-augmentation-review-container -v "$(pwd):/home/myuser/code" --user $(id -u):$(id -g) data-augmentation-review-image bash
```

The code will be stored under the directory `/home/myuser/code/` inside the docker container. This will allow you to use GPU acceleration.

## Requirements

If you do not want to use docker, simply install the project using the following command:
```bash
python3 -m venv ./data-augmentation-review-venv
source ./data-augmentation-review-venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Make sure you have [`jq`](https://jqlang.org/) installed on your system. This project supports `python>=3.10` only.

You can see the list of dependencies and their required version in the [pyptoject.toml](pyproject.toml) file.


## Running the code on a single experiment

If you wish to run a single experiment on a single dataset, using a single augmentation method, using a single model then first you have to execute your docker container to open a terminal inside if you're not inside the container:
```bash
docker exec -it data-augmentation-review-container bash
```
Then you can run the following command for example to run Amplitude Warping on the Adiac dataset:
```bash
python3 main.py task=generate_data dataset_name=Adiac generate_data.method=AW
```
The code uses [hydra](https://hydra.cc/docs/intro/) for the parameter configuration, simply see the [hydra configuration file](config/config_hydra.yaml) for a detailed view on the parameters of our experiments.

## Running the whole benchmark

If you wish to run all the experiments to reproduce the results of our article simply run the following for data generation experiments:
```bash
chmod +x run_generate_data.sh
nohup ./run_generate_data.sh &
```
and the following for training the feature extractor:
```bash
chmod +x run_train_feature_extractor.sh
nohup ./run_train_feature_extractor.sh &
```
and the following for evaluation of the generations:
```bash
chmod +x run_evaluate_generation.sh
nohup ./run_evaluate_generation.sh &
```

## Cite this work

If you use this work please cite the following:
```bibtex
@inproceedings{ismail-fawaz2025Data-Aug-4-TSC,
  author = {Ismail-Fawaz, Ali and Devanne, Maxime and Berretti, Sefano and Weber, Jonathan and Forestier, Germain},
  title = {Re-framing Time Series Augmentation Through the Lens of Generative Models},
  booktitle = {ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data},
  city = {Porto},
  country = {Portugal},
  year = {2025}
}
```

## Acknowledgments

This work was supported by the ANR DELEGATION project (grant ANR-21-CE23-0014) of the French Agence Nationale de la Recherche. The authors would like to acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific sup- port and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d’Avenir) and the CPER Alsacalcul/Big Data. The authors would also like to thank the creators and providers of the UCR Archive
