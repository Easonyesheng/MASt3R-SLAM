# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MASt3R-SLAM is a real-time dense visual SLAM system that uses the MASt3R foundation model for 3D reconstruction. It processes monocular RGB images to produce camera poses and dense 3D point clouds. The system can operate with or without known camera intrinsics.

## Environment & Build

Python 3.11, CUDA required. Uses conda environment `mast3r-slam`.

```bash
# Install dependencies (third-party packages must be installed first)
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```

The `setup.py` compiles C++/CUDA backend extensions (`mast3r_slam_backends`) using PyTorch's `CUDAExtension`. Sources are in `mast3r_slam/backend/src/`:
- `gn.cpp` + `gn_kernels.cu` — Gauss-Newton bundle adjustment (pose + pointmap optimization)
- `matching_kernels.cu` — Iterative projective matching and match refinement

CUDA architectures 6.0 through 8.6 are compiled by default. The backend depends on thirdparty/eigen for linear algebra.

Model checkpoints must be downloaded to `checkpoints/`:
```
MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

## Running the System

```bash
# Standard run (known calibration)
python main.py --dataset <path> --config config/calib.yaml

# Without calibration
python main.py --dataset <path> --config config/base.yaml

# With custom intrinsics file
python main.py --dataset <path> --config config/base.yaml --calib config/intrinsics.yaml

# Headless evaluation
python main.py --dataset <path> --no-viz --save-as <savedir> --config config/eval_calib.yaml
```

Dataset download and batch evaluation scripts are in `scripts/`.

## Architecture

### Three-Process Design

The system runs as three parallel processes communicating via shared memory (`torch.multiprocessing.Manager`):

1. **Main process** (`main.py`): Frame-by-frame tracking loop — loads images, creates `Frame` objects, calls `FrameTracker.track()`, decides when to add keyframes
2. **Backend process** (`run_backend`): Global bundle adjustment via `FactorGraph` — constructs pose graph, runs Gauss-Newton optimization, handles relocalization
3. **Visualization process** (`run_visualization`): Renders 3D view + camera trajectory via ModernGL/OpenGL (defined in `mast3r_slam/visualization.py`)

Communication is through `SharedStates` (mode, frame data, task queue) and `SharedKeyframes` (keyframe buffer with poses, pointmaps, confidences), both using `manager.RLock()` for synchronization.

### System Modes (State Machine)

Defined in `frame.py`: `INIT -> TRACKING <-> RELOC -> TERMINATED`
- **INIT**: First frame — run MASt3R mono inference, set as keyframe, queue global optimization
- **TRACKING**: Normal operation — match each frame against the last keyframe, estimate relative pose via Sim(3) optimization
- **RELOC**: Tracking lost — run mono inference on current frame, attempt to match against the keyframe database

### Core Data Structures

- **`Frame`** (`frame.py`): Holds image, canonical pointmap (`X_canon`), confidence (`C`), MASt3R features, Sim(3) pose. Multiple `filtering_mode` strategies (e.g., `weighted_pointmap`) control how repeated observations are merged.
- **`SharedKeyframes`** (`frame.py`): Fixed-buffer (default 512) shared-memory array of keyframes. Supports append, getitem, update_T_WCs.
- **`SharedStates`** (`frame.py`): Shared state for current frame + mode control + global optimization task queue.

### Tracking Pipeline (`tracker.py`)

`FrameTracker.track()`:
1. Run MASt3R asymmetric inference between frame and last keyframe
2. Match points using iterative projective matching + descriptor refinement
3. Filter matches by confidence thresholds (`C_conf`, `Q_conf`)
4. Gauss-Newton Sim(3) pose optimization (ray-distance or reprojection error depending on calibration)
5. Update keyframe pointmap with new observations
6. Decide if a new keyframe is needed (`match_frac_thresh`)

### Global Optimization (`global_opt.py`)

`FactorGraph` maintains edges (`ii`, `jj`) between keyframes. `add_factors()` runs symmetric MASt3R matching between keyframe pairs, stores correspondences. `solve_GN_calib()` / `solve_GN_rays()` calls C++/CUDA backend (`mast3r_slam_backends.gauss_newton_calib` / `gauss_newton_rays`) for bundle adjustment.

The first `pin` keyframes are frozen during optimization.

### Retrieval Database (`retrieval_database.py`)

Extends MASt3R's `Retriever` with an online-updatable inverted-file index for loop closure / relocalization. Features are extracted from the retrieval head, quantized via IVF, and scored with similarity kernels. Used by both the backend (loop closure) and relocalization.

### MASt3R Integration (`mast3r_utils.py`)

Wraps MASt3R model inference:
- `mast3r_inference_mono()` — single-frame pointmap (first frame init, reloc)
- `mast3r_asymmetric_inference()` — frame-to-keyframe matching (tracking)
- `mast3r_symmetric_inference()` / `mast3r_decode_symmetric_batch()` — bidirectional matching (global BA)

### Matching (`matching.py`)

Iterative projective matching of MASt3R pointmaps in ray space. Uses the CUDA backend for alignment iterations (`mast3r_slam_backends.iter_proj`) and descriptor-based refinement (`mast3r_slam_backends.refine_matches`).

### Configuration System (`config.py`)

YAML-based with `inherit` support (e.g., `config/calib.yaml` inherits `config/base.yaml`). A global `config` dict is set once at startup and read throughout the codebase via `from mast3r_slam.config import config`.

### Datasets (`dataloader.py`)

`load_dataset()` dispatches by path content to: `TUMDataset`, `EurocDataset`, `ETH3DDataset`, `SevenScenesDataset`, `RealsenseDataset`, `Webcam`, `MP4Dataset` (requires optional torchcodec), `RGBFiles`. All inherit from `MonocularDataset`.

### Geometry Utilities (`geometry.py`)

Sim(3) action (`act_Sim3`), ray-distance representation (`point_to_ray_dist`), calibrated reprojection (`project_calib`), backprojection, and related Jacobians. These are pure PyTorch — no autograd, explicit analytic Jacobians.

## Key Dependencies

- **lietorch**: Lie group operations (Sim3, SE3) by DROID-SLAM team
- **MASt3R** (thirdparty/mast3r): AsymmetricMASt3R model, retrieval head
- **in3d** (thirdparty/in3d): ModernGL-based 3D visualization
- **eigen** (thirdparty/eigen): C++ linear algebra for CUDA backend
- **evo**: Trajectory evaluation (ATE/RPE) for benchmark scripts

## Evaluation

Scripts in `scripts/eval_*.sh` run the system headless (`--no-viz`) with configs from `config/eval_calib.yaml` or `config/eval_no_calib.yaml`, then compute metrics with `evo_ape`. Results are saved under `logs/`.
