import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
import hydra


logger = logging.getLogger(__name__)


def visualize_predictions(
    model: nn.Module,
    test_predictions_np: np.ndarray,
    y_test_np: np.ndarray,
    cfg: DictConfig,
) -> None:
    # Visualize the predictions for each province
    for i in range(y_test_np.shape[1]):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_np[:, i], label="Actual")
        plt.plot(test_predictions_np[:, i], label="Predicted")
        plt.title(f"Province {i+1} Carbon Emissions Prediction")
        plt.xlabel("Days")
        plt.ylabel("Carbon Emissions")
        plt.legend()
        # Save the figure
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        plt.savefig(output_dir / f"province_{i+1}_carbon_emissions_prediction.png")

        # Close the figure to free up memory
        plt.close()
