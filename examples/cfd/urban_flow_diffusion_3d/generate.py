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

import hydra
from omegaconf import DictConfig
from physicsnemo import Module
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from src.train_utils.train_helpers import (
    set_seed,
    configure_cuda_for_consistent_precision,
    setup_distributed_and_logging,
)
from src.gen_utils.gen_helpers import generate_samples


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dist, logger, logger0 = setup_distributed_and_logging(cfg)
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    model = Module.from_checkpoint(cfg.generate.io.inf_ckpt_filepath)
    model.eval().to(dist.device)

    noise_scheduler = EDMNoiseScheduler(sigma_data=cfg.model.model_args.sigma_data)

    if dist.rank == 0:
        logger.info(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )

    generate_samples(cfg, model, noise_scheduler, dist, logger, logger0)


if __name__ == "__main__":
    main()
