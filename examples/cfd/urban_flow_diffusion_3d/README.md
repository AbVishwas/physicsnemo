<!-- markdownlint-disable -->
# 3D Diffusion Prior for Unconditional Sampling in a Simplified Turbulent Urban Environment

## Problem overview

This example trains a 3D denoising diffusion model as a generative prior over
turbulent flow fields (streamwise/wall-normal/spanwise velocity volumes) in a
simplified urban environment (flow past a single obstacle), then draws
unconditional samples from it, using
[`physicsnemo.experimental.models.diffusion_unets.DiffusionUNet3D`](../../../physicsnemo/experimental/models/diffusion_unets/diffusion_unet_3d.py)
as the denoising backbone, EDM preconditioning
(`physicsnemo.diffusion.preconditioners.EDMPreconditioner`), and the
`physicsnemo.diffusion` training/sampling utilities (noise schedulers,
`MSEDSMLoss`, `sample`).

The denoising backbone architecture is based on
[Diff-SPORT: Diffusion-based Sensor Placement Optimization and Reconstruction
of Turbulent flows in urban environments](https://arxiv.org/abs/2506.00214).
This example implements only the unconditional generative-prior training and
sampling stage of that pipeline -- not Diff-SPORT's sensor-placement
optimization or its conditional/posterior reconstruction from sparse
observations.

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

This example targets the **full-resolution** configuration: `288x88x88`
volumes (`D,H,W`), 3 channels (U, V, W), 4-level U-Net (`channel_mult:
[1,2,2,2]`, `model_channels: 64`).

The dataset used to develop and smoke-test this example is available at
[`abvish/UrbanFlow-oneObstacle-NoTrip`](https://huggingface.co/datasets/abvish/UrbanFlow-oneObstacle-NoTrip)
on Hugging Face (3D and 2D velocity fields from a Nek5000 spectral-element
CFD simulation of urban flow past a single obstacle, CC-BY-4.0, public).

Point the config at your dataset -- an HDF5 file containing either a single
combined `data` dataset of shape `(N, 3, 288, 88, 88)`, or separate `U`/`V`/`W`
datasets each of shape `(N, 288, 88, 88)` (both layouts are auto-detected by
`UflowDataset3D`; the latter is the raw/un-repacked layout, the former is
what an `-optimized` repacking pass produces):

```bash
python train.py paths.dataset=/path/to/your/data.h5
```

Generate unconditional samples from a checkpoint:

```bash
python generate.py paths.dataset=/path/to/your/data.h5 generate.io.inf_ckpt=<epoch>
```

### Configuration basics

Configuration is managed through [Hydra](https://hydra.cc/docs/intro/), with
config groups under `conf/`: `dataset/`, `model/`, `train/`, `generate/`,
`evaluate/`, `visualize/`. Override any value on the command line, e.g.
`python train.py train.hp.epochs=100 model.model_args.model_channels=64`.

## Smoke test

Trained for real against the full dataset (10,000 snapshots, no
subsampling) on a single L40S GPU, fp32, `batch_size_per_gpu=1`:

Loss decreases across both completed epochs, confirming the migrated
pipeline (`DiffusionUNet3D` + `EDMPreconditioner` + `EDMNoiseScheduler` +
`MSEDSMLoss`) trains correctly end-to-end on real data. 

