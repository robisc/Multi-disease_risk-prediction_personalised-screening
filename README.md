# Multi-disease_risk-prediction_personalised-screening
Multi-disease risk prediction models for population-scale personalised screening: https://doi.org/10.64898/2026.02.02.26345271

The notebooks in this repository were used for the data processing, model training and evaluation and plotting for the Multi-disease risk prediction models for population-scale personalised screening project.

Notebooks were numbered in accordance with the order of the workflow.

SupplementSignificance.xslx details the results of two-sided paired t-tests with Benjamini-Hochberg correction for multiple testing

## Software versions for reproducability

The following software versions can be used to reproduce our results (as specified in the requirements.txt):
- Python                    3.9.16
- scikit-survival            0.23.1
- scikit-learn               1.5.2
- lifelines                  0.30.1
- pandas                     2.2.3
- numpy                      1.26.4

## Reproducability

As our code (in the form of Juypter Notebooks) was run on UKBiobank, we provide a folder mock_usage in this repository, in which
- mock data with the columns used in our project can be created
- the main training pipeline can be executed

In order to execute run_mock_pipeline.py, please use the requirements.txt to create a virtual environment using your python 3.9 distribution (you can also use a newer python version and ignore the package verisons specified in requirements.txt):

```bash
cd mock_usage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After activating the environment, create the mock data using

```bash
python create_mock_data.py
```

and run the training pipeline using 

```bash
python run_mock_pipeline.py
```

Creating the venv, the mock data and training the pipeline should not take longer than 15 minutes. The results will be random though, as the mock data is just randomly generated.

As we transferred the code from our Jupyter Notebooks to these scripts, we cannot guarantee that outcomes will be identical to our original code. In case of questions please reach out to rafael.oexner@kcl.ac.uk .
