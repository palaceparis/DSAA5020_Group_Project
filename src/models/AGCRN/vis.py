import numpy as np
import matplotlib.pyplot as plt
import csv

# Load data
pred = np.load("outputs/AGCRN/CARBON_pred.npy")
truth = np.load("outputs/AGCRN/CARBON_true.npy")

time_steps = pred.shape[0]
num_nodes = pred.shape[2]

# Reshape the pred and truth arrays
pred_reshaped = pred.reshape(time_steps, num_nodes)
truth_reshaped = truth.reshape(time_steps, num_nodes)

# Define custom node names
filename = "data/interim/emissions.csv"
with open(filename, "r") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
node_names = header[1:]

# Visualize the predictions for the whole test dataset and each node
for node_idx in range(num_nodes):
    plt.figure(figsize=(12, 6))
    plt.plot(range(time_steps), pred_reshaped[:, node_idx], label="Predicted")
    plt.plot(range(time_steps), truth_reshaped[:, node_idx], label="True")
    plt.title(f"{node_names[node_idx]} Carbon Emissions Prediction")
    plt.xlabel("Time step")
    plt.ylabel("Carbon Emissions")
    plt.legend()

    # Save the figure
    plt.savefig(f"outputs/AGCRN/results_{node_names[node_idx]}.png")

    # Close the figure to free up memory
    plt.close()
