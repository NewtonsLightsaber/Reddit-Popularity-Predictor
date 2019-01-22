# Reddit Popularity Predictor

*COMP 551 Machine Learning Project 1 - McGill University*

## Prerequisites

* Python 3.7
* `miniconda` (or `anaconda`)

    ```
    # On Arch Linux
    pip install miniconda3
    ```

## Usage
### Setting up the environment
1. In the root directory of the project (shown in the **Project Organization** section below), create the `conda` virtual environment
    ```
    make create_environment
    ```
2. If not inside the virtual environment already, **enter it,** then install the required packages
    ```
    make requirements
    ```

### Building the datasets and features
The dataset is already split into 3 files in `data/processed/`:

    # data/processed/
    test_data.json
    training_data.json
    validation_data.json

The features are also built and ready in `src/features/`:

    # src/features/
    test_X_counts.json
    test_y.json
    training_X_counts.json
    training_y.json
    validation_X_counts.json
    validation_y.json

The commands for both are:

    # Create 3 .json datasets in data/processed/ from the raw dataset in data/raw/
    make data

    # Create files storing X_counts and y in src/features/ from the datasets in data/processed/
    make features

### Training models
The trained models are in `models/`:

    # models/
    ClosedForm.pkl
    GradientDescent.pkl

Train and save models to `models/` with:

    # Check environment requirements, build datasets and features,
    # then run src/models/train_model.py to train and save models to models/
    make train

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
