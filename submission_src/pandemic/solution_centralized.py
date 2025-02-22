from pathlib import Path

from loguru import logger
import pandas as pd

from src.sir_model import SirModel
from src.gnn_model import BatchGCN

NUM_ROW = 2000
LOOKAHEAD = 7


def fit(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
):
    logger.info("Running fit...")
    disease_outcome_df = pd.read_csv(disease_outcome_data_path, nrows=NUM_ROW)
    person_df = pd.read_csv(person_data_path, nrows=NUM_ROW)
    population_network_df = pd.read_csv(population_network_data_path, nrows=NUM_ROW)
    model = SirModel(lookahead=LOOKAHEAD)
    model.fit(disease_outcome_df)
    logger.info("...done running fit")
    logger.info("Saving model checkpoint...")
    model.save(model_dir / "model.json")
    logger.info("...done saving model checkpoint")


def predict(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    logger.info("Running predict to make test predictions...")
    disease_outcome_df = pd.read_csv(disease_outcome_data_path).iloc[0:NUM_ROW]
    preds_format_df = pd.read_csv(preds_format_path, index_col="pid")
    model = SirModel.load(model_dir / "model.json")

    logger.info("Computing predictions...")
    predictions = model.predict(disease_outcome_df)

    logger.info("Saving predictions...")
    predictions.loc[preds_format_df.index].to_csv(preds_dest_path)
    logger.info("...done with test predictions.")
