"""
Promote Loan Status model from staging (challenger) to production (champion).
Compares with production baseline or auto-promotes if no production exists.
"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_promotion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_promotion_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelPromoter:
    """Handle model promotion from staging to production for Loan Status prediction."""

    def __init__(self):
        """Initialize ModelPromoter with Loan Status configuration."""
        self.model_name = "loan-approval-xgboost"
        self.min_f1_score = 0.95  # Minimum acceptable F1 for production model
        self.improvement_threshold = 0.00  # Require 1% F1 improvement
        self.mlflow_uri = MLFLOW_TRACKING_URI
        self.client = None
        self.new_metrics = None
        self.prod_metrics = None
        self.old_champion_version = None
        self.challenger_version_number = None

    def get_challenger_metrics(self, challenger) -> dict:
        """Get metrics from challenger model's MLflow run."""
        try:
            run = self.client.get_run(challenger.run_id)
            metrics = run.data.metrics
            return {
                "f1_score": metrics.get("test_f1_score", metrics.get("f1_score", 0)),
                "precision": metrics.get("test_precision", metrics.get("precision", 0)),
                "recall": metrics.get("test_recall", metrics.get("recall", 0)),
                "version": challenger.version,
            }
        except Exception as e:
            logger.error(f"Failed to get challenger metrics: {e}")
            return None

    def get_production_metrics(self) -> dict:
        """Get metrics from current production model."""
        try:
            # Get production model version
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            run_id = champion.run_id
            self.old_champion_version = champion.version

            # Get run metrics
            run = self.client.get_run(run_id)
            metrics = run.data.metrics

            prod_metrics = {
                "f1_score": metrics.get("test_f1_score", metrics.get("f1_score", 0)),
                "precision": metrics.get("test_precision", metrics.get("precision", 0)),
                "recall": metrics.get("test_recall", metrics.get("recall", 0)),
                "version": champion.version,
            }

            logger.info(f"Production model found: version {champion.version}")
            return prod_metrics

        except MlflowException:
            logger.info("No production model found")
            self.old_champion_version = None
            return None

    def log_pre_promotion_state(self):
        """Log current state before promotion decision."""
        logger.info("=" * 70)
        logger.info("PRE-PROMOTION STATE:")
        logger.info(
            f"  Champion: version {self.prod_metrics['version'] if self.prod_metrics else 'None'}"
        )
        logger.info(f"  Challenger: version {self.challenger_version_number}")
        logger.info(f"  Challenger F1: {self.new_metrics['f1_score']:.4f}")
        if self.prod_metrics:
            logger.info(f"  Champion F1: {self.prod_metrics['f1_score']:.4f}")
        logger.info("=" * 70)

    def should_promote(self) -> tuple:
        """
        Decision logic for promotion.

        Case 1: No production model -> Auto-promote if meets minimum F1 threshold
        Case 2: Production exists -> Promote if new model improves F1 by threshold
        Case 3: Same version -> Skip (already champion)
        """
        new_f1 = self.new_metrics["f1_score"]

        # Case 3: Same version already champion
        if (
            self.prod_metrics
            and self.challenger_version_number == self.old_champion_version
        ):
            logger.info("DECISION: Challenger is same as current champion - skipping")
            return False, "same_version"

        # Case 1: No production model
        if self.prod_metrics is None:
            if new_f1 >= self.min_f1_score:
                logger.info("DECISION: No production model exists")
                logger.info(
                    f"New model F1 ({new_f1:.4f}) meets minimum threshold ({self.min_f1_score})"
                )
                return True, "first_production"
            else:
                logger.warning(
                    f"DECISION: New model F1 ({new_f1:.4f}) below minimum threshold ({self.min_f1_score})"
                )
                return False, "below_minimum"

        # Case 2: Production model exists - compare F1 performance
        prod_f1 = self.prod_metrics["f1_score"]
        improvement = new_f1 - prod_f1

        logger.info("MODEL COMPARISON:")
        logger.info(
            f"  Production F1: {prod_f1:.4f} (version {self.prod_metrics['version']})"
        )
        logger.info(f"  New Model F1:  {new_f1:.4f}")
        logger.info(f"  Improvement:   {improvement:+.4f} ({improvement*100:+.2f}%)")

        if improvement >= self.improvement_threshold:
            logger.info(
                f"DECISION: New model improves F1 by {improvement*100:.2f}% (threshold: {self.improvement_threshold*100:.0f}%)"
            )
            return True, "better_than_production"
        else:
            logger.warning(
                f"DECISION: F1 improvement ({improvement*100:.2f}%) below threshold ({self.improvement_threshold*100:.0f}%)"
            )
            return False, "not_better_enough"

    def retire_old_champion(self):
        """Remove champion alias from old production model and mark as retired."""
        if self.old_champion_version is not None:
            try:
                logger.info(
                    f"Retiring old production model (version {self.old_champion_version})"
                )

                # Remove champion alias from old version
                self.client.delete_registered_model_alias(self.model_name, "champion")

                # Mark as retired
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=self.old_champion_version,
                    key="deployment_status",
                    value="retired",
                )

                logger.info(f"Version {self.old_champion_version} marked as retired")

            except Exception as e:
                logger.warning(f"Failed to retire old champion: {e}")
                raise

    def log_post_promotion_state(self, version_number: int):
        """Log final state after successful promotion."""
        try:
            final_champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            logger.info("POST-PROMOTION STATE:")
            logger.info(f"  Champion: version {final_champion.version}")

            # Verify challenger was removed (next DVC run will create new challenger)
            try:
                _ = self.client.get_model_version_by_alias(
                    self.model_name, "challenger"
                )
                logger.warning("  Challenger: Still exists (cleanup incomplete)")
            except MlflowException:
                logger.info("  Challenger: None (ready for next registration)")

        except Exception as e:
            logger.warning(f"Could not verify post-promotion state: {e}")

    def write_promotion_flag(self, promoted: bool):
        """Write promotion result to file for CI/CD pipeline."""
        flag_file = Path("model_promoted.txt")
        flag_file.write_text("true" if promoted else "false")
        logger.info(f"Wrote promotion flag: {promoted}")

    def promote_to_production(self) -> bool:
        """Execute complete model promotion workflow."""
        try:
            # Connect to MLflow first
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.client = MlflowClient()

            # Fetch challenger model version
            try:
                challenger = self.client.get_model_version_by_alias(
                    self.model_name, "challenger"
                )
                self.challenger_version_number = challenger.version
            except MlflowException:
                logger.error("No challenger alias found. Run registration first.")
                return False

            # Get challenger metrics from MLflow
            self.new_metrics = self.get_challenger_metrics(challenger)
            if self.new_metrics is None:
                logger.error("Failed to load challenger metrics from MLflow")
                return False

            logger.info("=" * 70)
            logger.info("CHALLENGER MODEL METRICS (from MLflow):")
            logger.info(f"  F1 Score:    {self.new_metrics['f1_score']:.4f}")
            logger.info(f"  Precision:   {self.new_metrics['precision']:.4f}")
            logger.info(f"  Recall:      {self.new_metrics['recall']:.4f}")
            logger.info("=" * 70)

            # Get production model metrics (if exists)
            self.prod_metrics = self.get_production_metrics()

            # Log pre-promotion state
            self.log_pre_promotion_state()

            # Decision: Should we promote?
            should_promote_flag, reason = self.should_promote()

            if not should_promote_flag:
                logger.warning(f"Model NOT promoted: {reason}")
                self.write_promotion_flag(False)
                return False

            logger.info(
                f"Promoting model version {self.challenger_version_number} to production..."
            )

            # ATOMIC PROMOTION: Retire old → promote new → cleanup aliases
            self.retire_old_champion()

            # Promote new champion
            self.client.set_registered_model_alias(
                self.model_name, "champion", self.challenger_version_number
            )

            # Remove challenger alias (prevents dual alias issue)
            self.client.delete_registered_model_alias(self.model_name, "challenger")

            # Update deployment status tag
            self.client.set_model_version_tag(
                name=self.model_name,
                version=self.challenger_version_number,
                key="deployment_status",
                value="production",
            )

            logger.info(
                f"SUCCESS: Model v{self.challenger_version_number} is now CHAMPION (production)"
            )
            logger.info(f"Promotion reason: {reason}")

            # Log post-promotion state verification
            self.log_post_promotion_state(self.challenger_version_number)

            self.write_promotion_flag(True)
            return True

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            self.write_promotion_flag(False)
            raise

    def run(self):
        """Main execution method for model promotion."""
        logger.info("=" * 70)
        logger.info("LOAN STATUS MODEL PROMOTION PROCESS STARTED")
        logger.info("=" * 70)

        success = self.promote_to_production()

        logger.info("=" * 70)
        if success:
            logger.info("RESULT: MODEL PROMOTED TO PRODUCTION")
        else:
            logger.info("RESULT: MODEL REMAINS IN STAGING")
        logger.info("=" * 70)

        return success


if __name__ == "__main__":
    promoter = ModelPromoter()
    promoter.run()
