#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    # args.input_artifact=sample.csv:latest")

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Drop outliers")
    logger.info("Drop outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)  # bug fix 2023-04-02
    df = df[idx].copy()
    
    # Save file
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    #os.remove(filename)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic cleaning.")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min price, set this value to discard outliers.",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Max price, set this value to discart outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
