# Requirements

In order to run these components you need to have conda (Miniconda or Anaconda) and MLflow installed.
Install it with::

    > conda install mlflow=1.14.1

then run::

    > mlflow [url to this repo] -e help

to get a description of the commands.

# Ropository to the solution:
https://github.com/abimael-dominguez/build-ml-pipeline-for-short-term-rental-prices 

# How to run a release of the solved project:
´´´mlflow run https://github.com/abimael-dominguez/build-ml-pipeline-for-short-term-rental-prices \
             -v 1.0.1 \
             -P hydra_options="etl.sample='sample2.csv'"
             ´´´

# wandb project link:
https://wandb.ai/abimael-dominguez-perez/nyc_airbnb

