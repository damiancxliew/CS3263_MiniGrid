# CS3263 AI Project Repository

This repository is now organized into two self-contained project areas:

- `minigrid_project/` for the MiniGrid DoorKey reinforcement learning and A* planning work
- `grid_universe_project/` for the Grid-Universe planning and CNN+A* hybrid work

The goal of the reorganization is to keep code, scripts, and outputs for each environment together so the repository is easier to navigate and report on.

## Repository Layout

```text
minigrid_project/
  MiniGridSolve.py
  configs/
  scripts/
    run_experiment.py
    plot_results.py
  src/
    minigrid_solver/
  results/
    logs/

grid_universe_project/
  final.py
  gameplay_levels.py
  tile_cnn_loader.py
  train_logistic_regression.py
  train_tile_cnn.py
  utils.py
  scripts/
    evaluate_grid_universe.py
    render_grid_universe_video.py
  data/
    assets/
    cipher_objective.csv
    generated_tiles/
  results/
    evaluation/
    videos/

References/
requirements.txt
README.md
```

## MiniGrid Project

The MiniGrid part of the project contains:

- a tabular Q-learning baseline
- an explainable A* planning agent
- symbolic state abstraction over DoorKey environments
- experiment logs and comparison outputs

Important files:

- `minigrid_project/src/minigrid_solver/agents/q_learning.py`
- `minigrid_project/src/minigrid_solver/agents/hybrid.py`
- `minigrid_project/src/minigrid_solver/planning/astar.py`
- `minigrid_project/src/minigrid_solver/planning/symbolic_model.py`
- `minigrid_project/results/logs/`

### Run MiniGrid Experiments

Run the hybrid planning baseline:

```bash
python minigrid_project/scripts/run_experiment.py --agent hybrid --env MiniGrid-DoorKey-5x5-v0 --episodes 5
```

Run the Q-learning baseline:

```bash
python minigrid_project/scripts/run_experiment.py --agent qlearning --env MiniGrid-DoorKey-5x5-v0 --train-episodes 200 --episodes 5
```

Run both for comparison:

```bash
python minigrid_project/scripts/run_experiment.py --agent compare --env MiniGrid-DoorKey-5x5-v0 --train-episodes 200 --episodes 5
```

Plot saved MiniGrid comparison results:

```bash
python minigrid_project/scripts/plot_results.py --comparison-json minigrid_project/results/logs/doorkey_baseline/comparison.json
```

Legacy entrypoint:

```bash
python minigrid_project/MiniGridSolve.py
```

### MiniGrid Outputs

MiniGrid experiment outputs are stored under:

- `minigrid_project/results/logs/`

These include:

- per-episode JSON traces
- summary JSON files
- optional training curves
- optional recorded videos from Gymnasium wrappers

## Grid-Universe Project

The Grid-Universe part of the project contains:

- the final full-state and image-observation agent in `final.py`
- handcrafted level generators in `gameplay_levels.py`
- model export helpers and training scripts for the CNN and ciphertext decoder
- evaluation and rendering scripts
- notebook material and generated outputs

Important files:

- `grid_universe_project/final.py`
- `grid_universe_project/gameplay_levels.py`
- `grid_universe_project/scripts/evaluate_grid_universe.py`
- `grid_universe_project/scripts/render_grid_universe_video.py`
- `grid_universe_project/results/evaluation/`
- `grid_universe_project/results/videos/`

### Run Grid-Universe Evaluation

Evaluate the Grid-Universe agent across all authored levels:

```bash
python grid_universe_project/scripts/evaluate_grid_universe.py
```

Render a Grid-Universe episode to MP4:

```bash
python grid_universe_project/scripts/render_grid_universe_video.py --level build_level_required_two --fps 2
```

Open the notebook material:

- `grid_universe_project/notebooks/mini-project.ipynb`

### Grid-Universe Outputs

Grid-Universe outputs are stored under:

- `grid_universe_project/results/evaluation/`
- `grid_universe_project/results/videos/`

The evaluation folder contains structured JSON and CSV summaries. The videos folder contains rendered MP4 trajectories.

## Environment Setup

Use Python 3.10+.

If you are using the local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the larger conda environment, use:

- `requirements-conda.txt`

## Notes

- The MiniGrid and Grid-Universe codebases are intentionally separated now, but they still share the same repository for reporting and comparison.
- Historical outputs were preserved and moved into the corresponding `results/` folder for each project.
- Some generated or unused artifacts were removed during cleanup, including cache folders, backup scripts, and stray top-level files that were no longer referenced.
