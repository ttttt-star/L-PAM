# L-PAM
# L-PAM: Privacy-Preserving Task-Location Perturbation for Spatial Crowdsourcing

This repository contains the experimental implementation of **L-PAM** for privacy-preserving task-location perturbation and task allocation in spatial crowdsourcing.

The experiments evaluate the task-allocation utility of L-PAM under local differential privacy and compare it with representative LDP baselines, including GRR, HR, OLH-H, and an OUE-style unary-encoding baseline.

> Note: In the current code, L-PAM is implemented as `SRR-SelfFirst-Greedy`, and the OUE-style baseline is named `PLDP-Greedy` in the output files.

## 1. Requirements

The code is written in Python and only depends on common scientific computing packages.

```bash
pip install numpy matplotlib
```

`matplotlib` is used only for plotting figures. If it is not installed, the CSV result files can still be generated.

## 2. Dataset Preparation

### Foursquare

Download and extract the Foursquare dataset, then place the extracted files under a local folder, for example:

```text
Your file location
```

The script will recursively search this folder for files whose names contain `venues` and `checkins`. It first reads venue latitude/longitude coordinates, and then samples locations according to venue check-in counts. If the check-in file cannot be parsed, the script falls back to uniform venue sampling.

The default data path in the script is:

```text
Your file location
```

You can also specify the dataset path manually using:

```bash
--foursquare-root "Your file location"
```

## 3. Code Structure

```text
.
├── SRR_Foursquare_Task_Assignment_formal_v1.py
├── README.md
└── results/
```

The main script supports three groups of experiments:

```text
1. Main experiment
2. Grid-size sensitivity experiment
3. Worker/task number sensitivity experiment
```

## 4. Main Experiment

The main experiment evaluates different privacy mechanisms under multiple privacy budgets.

### Run command

```bash
python SRR_Foursquare_Task_Assignment_formal_v1.py ^
  --foursquare-root "Your file location" ^
  --grid-size 128 ^
  --num-workers 2000 ^
  --num-tasks 1000 ^
  --epsilons 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 ^
  --repeats 10 ^
  --out-dir results/foursquare_main
```

For Linux or macOS, use `\` instead of `^`:

```bash
python SRR_Foursquare_Task_Assignment_formal_v1.py \
  --foursquare-root "/path/to/foursquare" \
  --grid-size 128 \
  --num-workers 2000 \
  --num-tasks 1000 \
  --epsilons 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
  --repeats 10 \
  --out-dir results/foursquare_main
```

### Compared methods

The main experiment includes:

```text
GRR-Greedy
HR-Greedy
OLH-H-Greedy
PLDP-Greedy
SRR-SelfFirst-Greedy
```

Here, `SRR-SelfFirst-Greedy` corresponds to the proposed L-PAM method.

### Main outputs

After running the experiment, the output folder contains:

```text
raw_results.csv
summary_results.csv
metadata.txt
avg_distance_vs_epsilon.png
avg_distance_vs_epsilon.txt
perturbation_distance_vs_epsilon.png
perturbation_distance_vs_epsilon.txt
ldp_only_avg_distance_vs_epsilon.png
ldp_only_perturbation_distance_vs_epsilon.png
reference_avg_distance_vs_epsilon.png
reference_perturbation_distance_vs_epsilon.png
```

The most important files are:

```text
summary_results.csv
avg_distance_vs_epsilon.png
perturbation_distance_vs_epsilon.png
```

`summary_results.csv` reports the mean and standard deviation over repeated runs.

## 5. Grid-Size Sensitivity Experiment

The grid-size sensitivity experiment evaluates how different grid resolutions affect task-allocation utility.

### Run command

```bash
python SRR_Foursquare_Task_Assignment_formal_v1.py ^
  --foursquare-root "Your file location" ^
  --run-grid-sensitivity ^
  --grid-sensitivity-sizes 8,16,32 ^
  --grid-sensitivity-epsilons 2.0,4.0,6.0,8.0 ^
  --grid-sensitivity-repeats 5 ^
  --num-workers 2000 ^
  --num-tasks 1000 ^
  --out-dir results/foursquare_grid_sensitivity
```

### Output files

```text
grid_sensitivity_combined_summary.csv
grid_sensitivity_avg_distance.png
```

The script also creates separate subfolders for each grid size:

```text
results/foursquare_grid_sensitivity/
├── grid_8/
├── grid_16/
├── grid_32/
├── grid_sensitivity_combined_summary.csv
└── grid_sensitivity_avg_distance.png
```

Each `grid_*` folder contains the raw and summary results for the corresponding grid size.

## 6. Worker/Task Number Sensitivity Experiment

The workload sensitivity experiment evaluates how the method performs when the number of workers or tasks changes.

It contains two parts:

```text
1. Worker-count sensitivity:
   The number of workers changes while the number of tasks is fixed.

2. Task-count sensitivity:
   The number of tasks changes while the number of workers is fixed.
```

### Run command

```bash
python SRR_Foursquare_Task_Assignment_formal_v1.py ^
  --foursquare-root "Your file location" ^
  --run-workload-sensitivity ^
  --grid-size 128 ^
  --workload-sensitivity-epsilon 5.0 ^
  --workload-sensitivity-repeats 5 ^
  --worker-sensitivity-workers 1000,2000,3000,4000,5000 ^
  --worker-sensitivity-num-tasks 1000 ^
  --task-sensitivity-tasks 500,1000,1500,2000,3000 ^
  --task-sensitivity-num-workers 4000 ^
  --out-dir results/foursquare_workload_sensitivity
```

### Output files

```text
workload_sensitivity_combined_summary.csv
total_distance_vs_workers.png
total_distance_vs_workers.txt
total_distance_vs_tasks.png
total_distance_vs_tasks.txt
```

The output directory is organized as follows:

```text
results/foursquare_workload_sensitivity/
├── workers/
│   ├── workers_1000_tasks_1000/
│   ├── workers_2000_tasks_1000/
│   ├── workers_3000_tasks_1000/
│   ├── workers_4000_tasks_1000/
│   └── workers_5000_tasks_1000/
├── tasks/
│   ├── workers_4000_tasks_500/
│   ├── workers_4000_tasks_1000/
│   ├── workers_4000_tasks_1500/
│   ├── workers_4000_tasks_2000/
│   └── workers_4000_tasks_3000/
├── workload_sensitivity_combined_summary.csv
├── total_distance_vs_workers.png
├── total_distance_vs_workers.txt
├── total_distance_vs_tasks.png
└── total_distance_vs_tasks.txt
```

## 7. Running Main Experiment and Workload Sensitivity Together

The script can also run the main experiment first and then automatically run workload sensitivity.

```bash
python SRR_Foursquare_Task_Assignment_formal_v1.py ^
  --foursquare-root "Your file location" ^
  --grid-size 128 ^
  --num-workers 2000 ^
  --num-tasks 1000 ^
  --repeats 10 ^
  --also-run-workload-sensitivity ^
  --out-dir results/foursquare_main_with_workload
```

The workload sensitivity results will be saved under:

```text
results/foursquare_main_with_workload/workload_sensitivity/
```

## 8. Optional Arguments

### Specify a bounding box

By default, the script automatically selects a dense local region. You can manually specify a bounding box:

```bash
--foursquare-bbox "40.55,40.95,-74.10,-73.70"
```

The format is:

```text
min_lat,max_lat,min_lon,max_lon
```

### Disable automatic dense-region selection

```bash
--disable-auto-bbox
```

### Disable check-in weighting

```bash
--disable-checkin-weighting
```

When this option is used, the script samples venues uniformly instead of using check-in counts as weights.

### Include SRR-Local ablation

```bash
--include-srr-local
```

This adds:

```text
SRR-Local-Greedy
```

to the output results.

### Include HSTGreedy diagnostic curves

```bash
--include-hst-greedy
```

This adds HST-based matching results for diagnostic comparison. It is disabled by default to keep the figures compact.

## 9. Result Metrics

The main metrics include:

```text
total_true_distance_km
avg_true_distance_km
worker_perturbation_distance_km
task_perturbation_distance_km
runtime_sec
```

Among them:

```text
avg_true_distance_km
```

is the main task-allocation utility metric used in the main experiment.

For workload sensitivity, the main metric is:

```text
total_true_distance_km
```

which measures the total system travel cost as the number of workers or tasks changes.

## 10. Reproducibility

The default random seed is:

```text
42
```

You can change it using:

```bash
--seed 123
```

Each experiment is repeated several times, and the final results are averaged in `summary_results.csv`.

## 11. Citation

If you use this code, please cite our paper:

```bibtex
@article{lpam,
  title   = {L-PAM: Layered Perturbation and Allocation Mechanism for Privacy-Preserving Spatial Crowdsourcing},
  author  = {Anonymous},
  journal = {TBD},
  year    = {2026}
}
```
