import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra


def write_records(cfg, train_records, validate_records, test_records):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    # Write Out Records
    with pd.ExcelWriter(output_dir / "results.xlsx") as writer:
        df_train_records = pd.DataFrame(train_records, columns=["Epoch", "RMSE"])
        df_train_records.to_excel(writer, sheet_name="train_records")
        df_validate_records = pd.DataFrame(
            validate_records, columns=["Epoch", "MAE", "RMSE"]
        )
        df_validate_records.to_excel(writer, sheet_name="validate_records")
        df_test_records = pd.DataFrame(test_records, columns=["RMSE"])
        df_test_records.to_excel(writer, sheet_name="test_records")
