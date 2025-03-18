import yaml

# Load the YAML file
with open("./output/accelergy_output_scale_example_run_128x128_os/energy_estimation.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract the total energy value
total_energy = data.get("energy_estimation", {}).get("Total", None)

# Print the result
if total_energy is not None:
    print(f"Total Energy (Array size: 128x128; Dataflow: Output Stationary): {total_energy / 1e9}")
else:
    print("Total value not found.")


# Load the YAML file
with open("./output/accelergy_output_scale_example_run_128x128_ws/energy_estimation.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract the total energy value
total_energy = data.get("energy_estimation", {}).get("Total", None)

# Print the result
if total_energy is not None:
    print(f"Total Energy (Array size: 128x128; Dataflow: Weight Stationary): {total_energy / 1e9}")
else:
    print("Total value not found.")


# Load the YAML file
with open("./output/accelergy_output_scale_example_run_128x128_is/energy_estimation.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract the total energy value
total_energy = data.get("energy_estimation", {}).get("Total", None)

# Print the result
if total_energy is not None:
    print(f"Total Energy (Array size: 128x128; Dataflow: Input Stationary): {total_energy / 1e9}")
else:
    print("Total value not found.")