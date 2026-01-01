import logging
import os
import subprocess

import pandas as pd
import yaml

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class DataIngestion:
    def __init__(self, params_path: str = "params.yaml"):
        """
        Initialize DataIngestion with parameters from YAML configuration.

        Args:
            params_path: Path to parameters configuration file
        """
        self.params = self.load_params(params_path)
        self.raw_data_path = self.params["data_ingestion"]["raw_data_file"]
        self.interim_data_path = self.params["data_ingestion"]["interim_data_path"]
        self.drop_columns = self.params["data_ingestion"]["drop_columns"]

    def load_params(self, params_path: str) -> dict:
        """
        Load parameters from YAML configuration file.

        Args:
            params_path: Path to YAML file

        Returns:
            Dictionary containing configuration parameters
        """
        try:
            with open(params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters retrieved from %s", params_path)
            return params
        except FileNotFoundError:
            logger.error("File not found: %s", params_path)
            raise
        except yaml.YAMLError as e:
            logger.error("YAML error: %s", e)
            raise

    def fetch_data_from_dvc(self) -> None:
        """
        Pull data from DVC remote storage if not present locally.
        DVC handles downloading from configured remote (DagsHub).
        """
        try:
            if os.path.exists(self.raw_data_path):
                logger.info("Data already exists at %s", self.raw_data_path)
                return

            logger.info("Data not found locally. Pulling from DVC remote")
            result = subprocess.run(
                ["dvc", "pull", self.raw_data_path + ".dvc"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Data pulled successfully from DVC remote")
            logger.debug("DVC output: %s", result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error("Failed to pull data from DVC: %s", e.stderr)
            raise
        except FileNotFoundError:
            logger.error("DVC not installed or not in PATH")
            raise

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load raw data and perform initial cleaning.
        Drops unnecessary columns as specified in params.

        Returns:
            Cleaned pandas DataFrame
        """
        try:
            logger.info("Loading data from %s", self.raw_data_path)
            df = pd.read_csv(self.raw_data_path)
            logger.info("Data loaded successfully. Shape: %s", df.shape)

            logger.info("Dropping columns: %s", self.drop_columns)
            df = df.drop(columns=self.drop_columns, errors="ignore")
            logger.info("Data shape after dropping columns: %s", df.shape)

            return df
        except Exception as e:
            logger.error("Failed to load and clean data: %s", e)
            raise

    def save_interim_data(self, df: pd.DataFrame) -> None:
        """
        Save cleaned data to interim directory.

        Args:
            df: Pandas DataFrame to save
        """
        try:
            os.makedirs(self.interim_data_path, exist_ok=True)
            file_path = os.path.join(self.interim_data_path, "loan_data_cleaned.csv")
            df.to_csv(file_path, index=False)
            logger.info("Interim data saved to %s", file_path)
        except Exception as e:
            logger.error("Failed to save interim data: %s", e)
            raise

    def run(self) -> pd.DataFrame:
        """
        Execute the complete data ingestion pipeline.
        Pulls data from DVC if needed, loads and cleans data, saves to interim.

        Returns:
            Cleaned pandas DataFrame
        """
        try:
            logger.info("Starting data ingestion pipeline")
            self.fetch_data_from_dvc()
            df = self.load_and_clean_data()
            self.save_interim_data(df)
            logger.info("Data ingestion completed successfully")
            return df
        except Exception as e:
            logger.error("Data ingestion pipeline failed: %s", e)
            raise


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.run()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
