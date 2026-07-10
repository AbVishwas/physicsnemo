# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, time, warnings
import h5py, hydra, psutil, torch, tqdm, wandb
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from physicsnemo import Module
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.experimental.models.diffusion_unets import DiffusionUNet3D
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from torch.nn.parallel import DistributedDataParallel

from src.utils import deep_clean_omegaconf
from src.metrics.statistics import stat_eval_plot_reynolds_stress_planes
from src.dataloaders.dataset_builder import get_dataset_and_dataloader
from src.dataloaders.dataset_utils import (
    get_precision,
    move_batch_to_device,
    select_random_field,
)
from src.train_utils.train_helpers import (
    configure_cuda_for_consistent_precision,
    handle_and_clip_gradients,
    is_time_for_periodic_task_epoch,
    set_seed,
    setup_distributed_and_logging,
)
from src.gen_utils.gen_helpers import generate_samples, pack_uncond_preds

os.environ.setdefault(
    "WANDB_MODE", "online"
)  # respect an externally-set mode (e.g. "offline")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the 3D diffusion prior, then periodically sample and evaluate.

    Builds the ``DiffusionUNet3D`` + ``EDMPreconditioner`` stack and trains it
    with ``MSEDSMLoss`` over the urban-flow dataset. On the configured cadence
    it checkpoints, draws unconditional samples, and logs a Reynolds-stress
    comparison to Weights & Biases. Configured through Hydra (see ``conf/``).
    """
    warnings.filterwarnings(
        "ignore", message="Grad strides do not match bucket view strides"
    )  # https://github.com/pytorch/pytorch/issues/47163
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Performance settings
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Initialize distributed environment
    dist, logger, logger0 = setup_distributed_and_logging(cfg)

    fp_optimizations = cfg.train.perf.fp_optimizations
    enable_amp = fp_optimizations.startswith("amp")  # Flag for mixed precision
    amp_dtype = get_precision(fp_optimizations)  # autocast compute dtype
    # NOTE: pure "fp16" (non-AMP) is not supported here; reduced precision runs
    # through torch.autocast. Supported modes: "fp32" (default) and
    # "amp-{bf16,fp16}".

    logger.info(f"Saving the outputs in {os.getcwd()}")

    checkpoint_dir = cfg.train.io.checkpoint_dir

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    train_dataloader, dataset = get_dataset_and_dataloader(
        cfg, Train=True, seed=dist.rank
    )

    dataset_channels = dataset.num_channels()

    net = DiffusionUNet3D(
        x_channels=dataset_channels,
        num_levels=len(cfg.model.model_args.channel_mult),
        model_channels=cfg.model.model_args.model_channels,
        channel_mult=cfg.model.model_args.channel_mult,
        attention_levels=cfg.model.model_args.attention_levels,
        num_blocks=cfg.model.model_args.num_blocks,
        dropout=cfg.model.model_args.dropout,
        embedding_type=cfg.model.model_args.embedding_type,
        encoder_type=cfg.model.model_args.encoder_type,
        decoder_type=cfg.model.model_args.decoder_type,
        channel_mult_emb=cfg.model.model_args.channel_mult_emb,
        checkpoint_level=cfg.train.perf.songunet_checkpoint_level,
    )
    model = EDMPreconditioner(net, sigma_data=cfg.model.model_args.sigma_data)
    noise_scheduler = EDMNoiseScheduler(sigma_data=cfg.model.model_args.sigma_data)

    # Initialize loggers
    if dist.rank == 0:
        wandb.init(
            project="nvidia-physicsnemo-diffusion",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.train.wandb_params.run_name,
        )
        wandb.watch(
            model,
            log=cfg.train.wandb_params.log,
            log_freq=cfg.train.wandb_params.log_freq,
        )

    # Ensure contiguous memory layout before initializing DDP
    for param in model.parameters():
        with torch.no_grad():
            param.data = param.data.contiguous()

    model.train().requires_grad_(True).to(dist.device)

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    loss_fn = MSEDSMLoss(model, noise_scheduler)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.train.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()
    ## Resume training from previous checkpoints if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        epoch = load_checkpoint(
            path=checkpoint_dir,
            models=model,
            optimizer=optimizer,
            device=dist.device,
        )
    except Exception:
        epoch = 0
    epoch += 1

    # Compile the model if applicable
    if cfg.train.perf.compile:
        model = torch.compile(model)
        print("Model compiled successfully!")

    if dist.rank == 0:
        logger.info(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    batch_size_per_gpu = cfg.train.hp.batch_size_per_gpu
    logger0.info(
        f"Training for {cfg.train.hp.epochs} epochs...Starting from epoch {epoch}"
    )
    done = False
    x_axis = y_axis = z_axis = None

    for ep in range(epoch, cfg.train.hp.epochs + 1):
        tick_start_time = time.time()
        loss_accum = 0
        train_dataloader.sampler.set_epoch(epoch)
        num_batches = len(train_dataloader)
        pbar = tqdm.tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), disable=True
        )

        base_lr = cfg.train.hp.lr
        lr_decay = cfg.train.hp.lr_decay
        decay_from = cfg.train.hp.lr_decay_from
        lr_rampup = cfg.train.hp.lr_rampup

        for g in optimizer.param_groups:
            if ep < lr_rampup:
                g["lr"] = base_lr * min(ep / lr_rampup, 1)
            elif ep >= decay_from:
                g["lr"] = base_lr * max(
                    (lr_decay ** ((ep - decay_from) / decay_from)), 0
                )
            else:
                g["lr"] = base_lr
            current_lr = g["lr"]

        for batch_index, batch in pbar:
            # Compute & accumulate gradients
            optimizer.zero_grad(set_to_none=True)

            # Keep inputs in fp32; torch.autocast casts eligible ops to the AMP
            # dtype internally when enabled (standard bf16/fp16 AMP practice).
            batch = move_batch_to_device(batch, device=dist.device, dtype=torch.float32)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                # MSEDSMLoss samples t, adds noise, calls the (precond-wrapped)
                # model, and returns the mean-reduced loss directly.
                loss = loss_fn(batch["field"], condition=None)
            loss.backward()

            # Clip gradients
            handle_and_clip_gradients(
                model, grad_clip_threshold=cfg.train.hp.grad_clip_threshold
            )

            optimizer.step()
            loss_accum += loss.detach().item() / num_batches

        loss_sum = torch.tensor([loss_accum], device=dist.device)

        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()

        if dist.rank == 0:
            wandb.log(
                {
                    "train/loss": average_loss,
                    "train/lr": current_lr,
                    "epoch": ep,
                }
            )

        ptt = is_time_for_periodic_task_epoch(
            ep,
            cfg.train.io.print_progress_freq,
            done,
            dist.rank,
            rank_0_only=True,
        )

        done = ep >= cfg.train.hp.epochs

        if ptt:
            tick_end_time = time.time()
            fields = []
            fields += [f"epoch {ep:<6}"]
            fields += [f"avg_training_loss {average_loss:<7.2f}"]
            fields += [f"batch_size:{dist.world_size:<3.1f}x{batch_size_per_gpu:<3.1f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
            fields += [f"sec_per_epoch {(tick_end_time - tick_start_time):<7.1f}"]
            fields += [
                f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
            ]
            logger0.info(" ".join(fields))
            torch.cuda.reset_peak_memory_stats()

        # Unwrap model first
        unwrapped = model.module if dist.world_size > 1 else model

        # Clean the model._args dictionary
        if hasattr(unwrapped, "_args"):
            unwrapped._args = deep_clean_omegaconf(unwrapped._args)

        # Save checkpoints
        if dist.world_size > 1:
            torch.distributed.barrier()

        if is_time_for_periodic_task_epoch(
            ep,
            cfg.train.io.save_checkpoint_freq,
            done,
            dist.rank,
            rank_0_only=True,
        ):
            save_checkpoint(
                path=checkpoint_dir,
                models=unwrapped,
                optimizer=optimizer,
                epoch=ep,
            )

        if is_time_for_periodic_task_epoch(
            ep,
            cfg.train.io.save_checkpoint_freq,
            done,
            dist.rank,
            rank_0_only=False,
        ):
            if dist.world_size > 1:
                torch.distributed.barrier()

            # NOTE: checkpoint filename derives from the wrapped class name --
            # renamed from EDMPrecond.*.mdlus to EDMPreconditioner.*.mdlus
            # since preconditioning.py's EDMPrecond3D -> EDMPreconditioner.
            model_gen = Module.from_checkpoint(
                cfg.train.io.checkpoint_dir + f"/EDMPreconditioner.0.{ep}.mdlus"
            )
            generate_samples(
                cfg, model_gen, noise_scheduler, dist, logger, logger0, val_mode=True
            )

            base_filename_h5 = cfg.generate.io.gen_dir + "/" + cfg.generate.io.filename

            if dist.rank == 0:
                preds_np = pack_uncond_preds(
                    base_filename_h5=base_filename_h5, dist=dist
                )

                gtruth_np = select_random_field(
                    dataset.data,
                    num_elements=cfg.generate.total_images,
                    seed=dist.rank,
                    combined_channels=dataset.combined_channels,
                )
                # NOTE: the ds4/downsampled config needed a `[:22, :22]` crop
                # here to reconcile generation's padded-to-24 output against
                # the dataset's unpadded 22x22 ground truth. Not needed at
                # full resolution -- generation isn't padded (see
                # gen_helpers.generate_samples), so gtruth/preds already
                # match in shape.
                if x_axis is None and y_axis is None and z_axis is None:
                    ds_ratio = cfg.dataset.ds_ratio
                    with h5py.File(cfg.paths.dataset, "r") as file:
                        x_axis, y_axis, z_axis = (
                            file["x"][:],
                            file["y"][:],
                            file["z"][:],
                        )

                fig = stat_eval_plot_reynolds_stress_planes(
                    gtruth_np[:, :, :, cfg.evaluate.plots.z_plane],
                    preds_np[:, :, :, cfg.evaluate.plots.z_plane],
                    x=None,
                    y=None,
                    z=None,
                    data="plane",
                    input_data_type="2D",
                    ds_ratio=ds_ratio,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    plot_config=cfg.evaluate.plots,
                )
                wandb.log({f"re_stress_from_epoch_{ep}": wandb.Image(fig)})
                plt.close(fig)

    logger0.info("Training Completed.")
    wandb.finish()


if __name__ == "__main__":
    main()
