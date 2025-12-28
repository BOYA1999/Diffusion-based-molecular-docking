# Diffusion-Guided Molecular Docking Generation

A diffusion-based molecular docking generation method that incorporates DiffGui's "diffusion guidance" principle, introducing differentiable geometric constraints and docking score guidance during the diffusion sampling process.

## Core Idea

This work extends DiffGui's diffusion guidance concept from bond prediction to molecular docking:

- **DiffGui**: Uses a differentiable bond predictor network to guide diffusion sampling
- **Our Method**: Uses a differentiable docking score predictor network to guide ligand generation

## Key Components

### 1. Geometry Guidance (`geometry_guidance.py`)

Analytical geometric energy-based guidance that requires no training:

- **Pocket center guidance**: Encourages ligand atoms to distribute near the pocket center
- **Collision penalty**: Prevents severe internal atom overlaps
- **Gradient-based update**: `new_pos = pos - step_size * grad(energy)`

### 2. Docking Score Predictor (`docking_score_predictor.py`)

A lightweight differentiable neural network that approximates Vina docking scores:

- **Input**: Protein/ligand atom types and 3D coordinates
- **Architecture**: Atom embeddings + radial basis functions (RBF) + MLP
- **Output**: Scalar docking score per complex
- **Training**: Supervised learning on MOAD dataset with Vina scores

### 3. Docking Guidance Step (`docking_score_predictor.py`)

Gradient-based guidance following DiffGui's `bond_guidance` pattern:

```python
ligand_pos_in = pred_pos.detach().clone().requires_grad_(True)
score = predictor(...)
grad = torch.autograd.grad(score.mean(), ligand_pos_in)[0]
new_pos = ligand_pos_in + step_size * grad
```

### 4. Two-Stage Guidance Strategy

- **Stage A (t > T/2)**: Geometry guidance for fast pocket localization
- **Stage B (t ≤ T/2)**: Docking score guidance for fine-tuning binding modes

## File Structure

```
GitHub/
├── README.md                    # Documentation
├── requirements.txt              # Python dependencies
├── geometry_guidance.py          # Geometry guidance module
├── docking_score_predictor.py    # Docking score predictor and guidance
├── docking_guided_model.py      # Main model integrating guidance
├── compare_baselines.py          # Baseline comparison script
├── mindmap.md                    # Architecture mindmap
└── __init__.py                  # Package initialization
```

## Code Overview

### `geometry_guidance.py`
- `_center_and_clash_energy()`: Computes geometric energy (pocket center + collision penalty)
- `geometry_guidance_step()`: Performs one gradient-guided update on ligand coordinates

### `docking_score_predictor.py`
- `DockingScorePredictor`: Differentiable neural network that approximates Vina docking scores
- `docking_guidance_step()`: Gradient-based guidance following DiffGui's pattern

### `docking_guided_model.py`
- `DockingGuidedDiffusionModel`: Main model that integrates geometry and docking guidance
- `sample_with_guidance()`: Sampling method with two-stage guidance

### `compare_baselines.py`
- `BaselineComparator`: Compares different guidance methods (no guidance, geometry-only, docking-only, two-stage)

## Usage

### Basic Usage

```python
from geometry_guidance import geometry_guidance_step
from docking_score_predictor import DockingScorePredictor, docking_guidance_step
from docking_guided_model import DockingGuidedDiffusionModel

# Initialize model
# Note: Provide your own sample_function in docking_config or ensure models.sample.sample_loop3 is available
model = DockingGuidedDiffusionModel(
    base_model=base_diffusion_model,
    docking_config={
        'use_geometry_guidance': True,
        'geometry_guidance_step_size': 1.0,
        'geometry_center_radius': 5.0,
        'sample_function': your_sample_function,  # Optional: provide custom sampling function
        'tmp_dir': './tmp_docking',  # Optional: temporary directory for docking
    },
    use_docking_guidance=True,
)

# Sample with guidance
results = model.sample_with_guidance(
    batch=batch,
    noiser=noiser,
    device='cuda',
    num_samples=20,
    top_k=5,
)
```

### Baseline Comparison

```python
from compare_baselines import BaselineComparator

comparator = BaselineComparator(
    base_model=base_model,
    docking_config=docking_config,
    device='cuda',
    num_samples_per_method=20,
    num_test_proteins=10,
)

summary, detailed_results = comparator.compare_all_methods(test_batches, noiser)
comparator.print_comparison_table(summary)
```

## Key Design Principles

### 1. Differentiability

All components are differentiable to enable gradient-based guidance:
- Geometry energy function: `E_geo(x)` with `grad(E_geo)`
- Docking score predictor: Neural network with `grad(score)`

### 2. Real-Time Guidance

Guidance occurs during diffusion sampling, not post-hoc:
- Each time step: `x_{t-1} = diffusion_update(x_t) + guidance_delta`
- Follows DiffGui's design pattern exactly

### 3. Two-Stage Strategy

- **Geometry guidance**: Fast, no training data needed
- **Docking guidance**: Precise, requires trained predictor
- **Combination**: Best of both worlds

## Experimental Results

| Method | Avg Docking Score | Best Score | Success Rate |
|--------|------------------|-----------|--------------|
| No Guidance | [Results] | [Results] | [Results] |
| Geometry Only | [Results] | [Results] | [Results] |
| Docking Only | [Results] | [Results] | [Results] |
| **Two-Stage (Ours)** | **[Results]** | **[Results]** | **[Results]** |

**Note**: Please refer to the paper for detailed experimental results.

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- RDKit (for molecular operations)
- AutoDock Vina (for docking evaluation, install via conda: `conda install -c conda-forge vina`)

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Diffusion-Guided Molecular Docking Generation: Applying DiffGui's Guidance Principle to Structure-Based Drug Design},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Your License]

## Acknowledgments

- Inspired by DiffGui's diffusion guidance mechanism
- Built on FlashDiff diffusion model framework
- Uses MOAD dataset for training

