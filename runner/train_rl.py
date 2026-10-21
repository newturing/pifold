import datetime
import hashlib
import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from rdkit import RDLogger

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from pifold.config.config import parse_configs, parse_sys_args, save_config
from pifold.data.pipeline.dataloader import get_dataloaders, skip_first_batches
from pifold.model.loss import GRPOLoss
from pifold.model.pifold import PiFold
from pifold.model.reward import RewardModel
from pifold.utils.distributed import DIST_WRAPPER
from pifold.utils.lr_scheduler import FinetuneLRScheduler, get_lr_scheduler
from pifold.utils.metrics import SimpleMetricAggregator
from pifold.utils.seed import seed_everything
from pifold.utils.torch_utils import autocasting_disable_decorator, to_device
from pifold.utils.training import (
    get_optimizer,
    is_loss_nan_check,
    shuffle_tensor_dict,
    split_tensor_dict,
)
from runner.ema import EMAWrapper
from runner.dumper import DataDumper

# Disable WANDB's console output capture to reduce unnecessary logging
os.environ["WANDB_CONSOLE"] = "off"
RDLogger.DisableLog("rdApp.*")  # disables all RDKit warnings


class PiFoldTrainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_log()
        self.init_model()
        self.init_loss()
        self.init_data()
        self.init_dumper()
        self.init_reward_model()
        self.init_noise_schedule()
        self.try_load_checkpoint()

    def init_basics(self) -> None:
        """
        Initialize basic training parameters and directory structure.
        """
        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate

        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.structure_dir = f"{self.run_dir}/structures"
        self.dump_dir = f"{self.run_dir}/dumps"
        self.eval_dump_dir = f"{self.run_dir}/eval_dumps"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir)
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.prediction_dir)
            os.makedirs(self.structure_dir)
            os.makedirs(self.dump_dir)
            os.makedirs(self.eval_dump_dir)
            os.makedirs(self.error_dir)
            save_config(
                self.configs,
                os.path.join(self.configs.base_dir, self.run_name, "config.yaml"),
            )

        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"checkpoint_dir: {self.checkpoint_dir}, "
            f"prediction_dir: {self.prediction_dir}, "
            f"structure_dir: {self.structure_dir}, "
            f"error_dir: {self.error_dir}"
        )

    def init_log(self) -> None:
        """
        Initialize logging and metrics aggregation.
        """
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=vars(self.configs),
                id=self.configs.wandb_id or None,
            )
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])

    def init_env(self) -> None:
        """
        Initialize the PyTorch and CUDA environment for training.
        Sets up distributed training if world_size > 1 and sets random seeds.
        """
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # Different DDP processes get different seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH env as per "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
        logging.info("Finished environment initialization.")

    def init_loss(self) -> None:
        """
        Initialize loss functions.
        """
        self.loss = GRPOLoss(**self.configs.rl.loss)

    def init_model(self):
        self.raw_model = PiFold(self.configs).to(self.device)
        self.print("Train π-Fold with RL")

        # Freeze entire model
        for p in self.raw_model.parameters():
            p.requires_grad = False

        # Unfreeze diffusion module only
        for p in self.raw_model.diffusion_module.parameters():
            p.requires_grad = True

        self.ref_model = PiFold(self.configs).to(self.device)
        self.ref_model.eval()

        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.print("Using DistributedDataParallel (DDP)")
            self.use_ddp = True
            # Fix DDP/checkpoint compatibility:
            # https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            self.model = DDP(
                self.raw_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        else:
            self.model = self.raw_model

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model Parameters: {count_parameters(self.model):.2f}M")
        if self.configs.get("ema_decay", -1) > 0:
            assert self.configs.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.model,
                self.configs.ema_decay,
                self.configs.ema_mutable_param_keywords,
            )
            self.ema_wrapper.register()

        torch.cuda.empty_cache()
        self.optimizer = get_optimizer(
            self.configs,
            self.model,
            param_names=self.configs.get("finetune_params_with_substring", [""]),
        )
        self.init_scheduler()

    def init_scheduler(self, **kwargs: Any) -> None:
        """
        Initialize the learning rate scheduler.
        Supports both standard and fine-tuning schedulers.

        Args:
            **kwargs: Additional arguments passed to the scheduler.
        """
        # init finetune lr scheduler if available
        finetune_params = self.configs.get("finetune_params_with_substring", [""])
        is_finetune = len(finetune_params[0]) > 0

        if is_finetune:
            self.lr_scheduler = FinetuneLRScheduler(
                self.optimizer,
                self.configs,
                self.configs.finetune,
                **kwargs,
            )
        else:
            self.lr_scheduler = get_lr_scheduler(self.configs, self.optimizer, **kwargs)

    def init_data(self) -> None:
        """
        Initialize training and test dataloaders.
        """
        self.train_dl, self.test_dls = get_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = False
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
            dist_setting=DIST_WRAPPER.world_size > 1,
        )

    def init_noise_schedule(self):
        inference_noise_scheduler = self.raw_model.inference_noise_scheduler
        self.N_sample = self.configs.rl.sample_diffusion["N_sample"]
        self.N_step = self.configs.rl.sample_diffusion["N_step"]
        self.noise_schedule = inference_noise_scheduler(
            N_step=self.N_step,
            device=self.device,
        )

    def init_reward_model(self):
        reward_cfg = self.configs.rl.reward
        self.scorer = RewardModel(
            pyrosetta_config=reward_cfg.pyrosetta,
            smina_config=reward_cfg.smina,
            strain_energy_config=reward_cfg.strain_energy,
            clean_files=reward_cfg.clean_files,
            ligand=reward_cfg.ligand,
        )

    def save_checkpoint(
        self, epoch: int = 0, batches_in_epoch: int = 0, ema_suffix: str = ""
    ) -> None:
        """
        Save the current model state, optimizer, and scheduler to a checkpoint file.

        Args:
            ema_suffix (str): Optional suffix for EMA checkpoints.
            epoch (int): Current epoch number.
            batches_in_epoch (int): Number of batches processed in the current epoch.
        """
        if DIST_WRAPPER.rank == 0:
            path = f"{self.checkpoint_dir}/{self.step}{ema_suffix}.pt"
            optimizer_path = f"{self.checkpoint_dir}/{self.step}_optimizer.pt"
            if ema_suffix:
                checkpoint = {"model": self.model.state_dict()}
            else:
                checkpoint = {
                    "model": self.model.state_dict(),
                    "step": self.step,
                    "epoch": epoch,
                    "batches_in_epoch": batches_in_epoch,
                }
                optimizer = {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None
                        else None
                    ),
                }
                torch.save(optimizer, optimizer_path)
                self.batches_in_epoch = batches_in_epoch
                self.epoch = epoch
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def sync_ref_model(self, checkpoint_path: str = None):
        """
        Syncs the reference model's parameters.

        If a checkpoint_path is provided, it loads from the file.
        Otherwise, it updates the reference model from the current policy model (`self.model`).
        """
        state_dict = None
        if checkpoint_path:
            self.print(f"Loading ref model from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, self.device)
            state_dict = checkpoint["model"]
            if next(iter(state_dict)).startswith("module."):
                state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        else:
            self.print(
                f"Updating ref model from the current policy model at {self.step}."
            )
            model_to_copy = (
                self.model.module if isinstance(self.model, DDP) else self.model
            )
            state_dict = model_to_copy.state_dict()

        self.ref_model.load_state_dict(
            state_dict=state_dict,
            strict=self.configs.load_strict,
        )
        self.print("Reference model parameters were successfully synced.")

    def try_load_checkpoint(self):

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
        ) -> None:
            """
            Internal helper to load a single checkpoint.

            Args:
                checkpoint_path (str): Path to the checkpoint file.
                load_params_only (bool): If True, only load model parameters.
                skip_load_optimizer (bool): If True, do not load optimizer state.
                skip_load_step (bool): If True, do not load training step.
                skip_load_scheduler (bool): If True, do not load scheduler state.
            """
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Given checkpoint path does not exist: [{checkpoint_path}]"
                )
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            sample_key = list(checkpoint["model"].keys())[0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not self.use_ddp:
                # DDP checkpoint has module. prefix, remove it if not using DDP
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            self.model.load_state_dict(
                state_dict=checkpoint["model"],
                strict=self.configs.load_strict,
            )
            if not load_params_only:
                optimizer_path = os.path.join(
                    os.path.dirname(checkpoint_path),
                    f"{checkpoint['step']}_optimizer.pt",
                )
                optimizer_ckpt = torch.load(optimizer_path, self.device)
                if not skip_load_optimizer:
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(optimizer_ckpt["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.epoch = checkpoint.get("epoch", 0)
                    self.batches_in_epoch = checkpoint.get("batches_in_epoch", 0)
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print(f"Loading scheduler state")
                    self.lr_scheduler.load_state_dict(optimizer_ckpt["scheduler"])
                else:
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.init_scheduler(last_epoch=self.step - 1)
            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load EMA model parameters if configured
        if self.configs.load_ema_checkpoint_path:
            _load_checkpoint(
                self.configs.load_ema_checkpoint_path,
                load_params_only=True,
            )
            self.ema_wrapper.register()

        # Load standard model checkpoint if configured
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
            )

        self.sync_ref_model(checkpoint_path=self.configs.rl.load_ref_checkpoint_path)

    def print(self, msg: str) -> None:
        """
        Print message to log only on the master rank (rank 0).

        Args:
            msg (str): The message to log.
        """
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def model_forward(
        self, batch: dict, mode: str = "train", model_type: str = "policy"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass of the model.

        Args:
            batch (Dict[str, Any]): Input batch containing features and labels.
            mode (str): Execution mode, either 'train' or 'eval'.
            model_type (str): Model type, either 'policy' or 'ref'.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Updated batch with
                predictions and logging dictionary.
        """
        assert mode in ["train", "eval", "rl_sample", "rl_train"]
        assert model_type in ["policy", "ref"], "model_type must be 'policy' or 'ref'"
        model_to_use = self.model if model_type == "policy" else self.ref_model
        pred_dict, batch["label_dict"], log_dict = model_to_use(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=None,
        )
        return pred_dict, log_dict

    def get_loss(
        self,
        batch: dict,
        pred_dict_cur: dict,
        pred_dict_ref: dict,
        t_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the loss for a given batch.

        Args:
            batch (Dict[str, Any]): Batch containing features, labels, and predictions.
            pred_dict_cur (Dict[str, Any]): Predicted dictionary from the current model.
            pred_dict_ref (Dict[str, Any]): Predicted dictionary from the reference model.
            t_idx (torch.Tensor): Index of the current timestep.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Total loss tensor.
                - Dictionary containing individual loss components.
        """
        loss, loss_dict = autocasting_disable_decorator(self.configs.skip_amp.loss)(
            self.loss
        )(
            log_prob=pred_dict_cur["log_prob"],
            log_prob_old=batch["log_prob_old"][:, t_idx],
            x_l_next_mean=pred_dict_cur["x_l_next_mean"],
            x_l_next_mean_ref=pred_dict_ref["x_l_next_mean"],
            std_c_tau=pred_dict_cur["std_c_tau"],
            rewards=batch["rewards"],
            lig_mask=batch["lig_mask"],
        )
        return loss, loss_dict

    def reward_fn(
        self, pred_dict: dict, batch: dict, mode: str = "rl_sample"
    ) -> torch.Tensor:
        assert mode in ["eval", "rl_sample"]
        score_ligand = self.configs.rl.reward.ligand
        lig_w = self.configs.rl.reward.lig_w
        strain_w = self.configs.rl.reward.strain_w
        struc_w = self.configs.rl.reward.struc_w
        dump_dir = self.dump_dir if mode == "rl_sample" else self.eval_dump_dir
        struc_paths, lig_paths, lig_mask = self.dumper.dump_structures(
            dump_dir=dump_dir,
            pdb_id=batch["basic"]["pdb_id"],
            seed=self.configs.seed,
            pred_coordinates=pred_dict["coordinate"],
            atom_array=batch["cropped_atom_array"],
            entity_poly_type=None,
            return_path=True,
            score_ligand=score_ligand,
        )
        reward_results = self.scorer.score(
            structure_files=struc_paths,
            ligand_files=lig_paths,
        )
        struc_rewards, lig_rewards, strain_rewards = (r[1] for r in reward_results)
        if score_ligand:
            rewards = (
                struc_w * struc_rewards
                + lig_w * lig_rewards
                + strain_w * strain_rewards
            )
        else:
            rewards = struc_rewards

        rewards = to_device(rewards, self.device)

        return rewards, struc_rewards, lig_rewards, strain_rewards, lig_mask

    @torch.no_grad()
    def evaluate(self, mode: str = "eval") -> None:
        """
        Evaluate the model on all test datasets.
        Handles both standard and EMA model evaluation if applicable.

        Args:
            mode (str): Execution mode, typically 'eval'.
        """
        if not self.configs.eval_ema_only:
            self._evaluate(mode=mode)
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()
            self._evaluate(ema_suffix=f"ema{self.ema_wrapper.decay}_", mode=mode)
            self.ema_wrapper.restore()

    @torch.no_grad()
    def _evaluate(self, ema_suffix: str = "", mode: str = "eval") -> None:
        """
        Internal evaluation loop for a specific model state.

        Args:
            ema_suffix (str): Suffix for metric names if using EMA.
            mode (str): Execution mode.
        """
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks to avoid duplicate evaluation
                    # when drop_last=False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Dropping data_id {pid} "
                            "as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    pred_dict, _ = self.model_forward(batch, mode=mode)
                    # Rosetta metrics
                    rewards, struc_rewards, lig_rewards, _ = self.reward_fn(
                        pred_dict, batch, mode
                    )
                    rewards = rewards[torch.isfinite(rewards)]

                    if rewards.numel() > 0:
                        simple_metrics.update(
                            {
                                f"{name}_{stat}": getattr(tensor, stat)().item()
                                for name, tensor in {
                                    "reward": rewards,
                                    "struc_reward": struc_rewards,
                                    "lig_reward": lig_rewards,
                                }.items()
                                for stat in ["mean", "std"]
                            }
                        )

                # Update metric aggregator
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        f"{ema_suffix}{key}", value, namespace=test_name
                    )

                del pred_dict, batch, simple_metrics
                if index % 5 == 0:
                    # Release memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    @torch.no_grad()
    def sample_diffusion(self, batch: dict, mode: str = "rl_sample") -> dict:
        """
        SAMPLING: Generates experience data, and structured metric tracking.
        """
        # 1. SETUP for Sampling Phase
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        self.print("Sampling new experience batch...")
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()
        # 2. Generate Trajectories
        with enable_amp:
            pred_dict, _ = self.model_forward(batch, mode, model_type="policy")
        # 3. Extract Data
        trajectory_data = pred_dict["x_ls"]
        log_prob_old = pred_dict["log_probs"]

        # 4. Calculate Rewards and Track Metrics
        rewards, struc_rewards, lig_rewards, strain_rewards, lig_mask = self.reward_fn(
            pred_dict, batch, mode
        )
        simple_metric_wrapper.add("reward_mean", rewards.mean())
        simple_metric_wrapper.add("reward_std", rewards.std())
        simple_metric_wrapper.add("struct_reward_mean", struc_rewards.mean())

        if self.configs.rl.reward.ligand:
            simple_metric_wrapper.add("lig_reward_mean", lig_rewards.mean())
            simple_metric_wrapper.add("strain_reward_mean", strain_rewards.mean())

        # Calculate and log the aggregated metrics
        reward_metrics = simple_metric_wrapper.calc()
        self.print(f"Sampling Metrics: {reward_metrics}")

        if (
            self.configs.use_wandb
            and DIST_WRAPPER.rank == 0
            and torch.isfinite(rewards).all()
        ):
            wandb.log(
                {f"sample/{k}": v for k, v in reward_metrics.items()}, step=self.step
            )

        timesteps = torch.arange(self.N_step - 1, device=self.device)

        # 5. Populate the Cache, and Remove the last step (deterministic, no noise added)
        sample_buffer = {
            "x_ls": torch.stack(trajectory_data, dim=1)[
                :, : self.N_step - 1
            ],  # [G, N_step-1, N_atm, 3]
            "x_l_nexts": torch.stack(trajectory_data, dim=1)[
                :, 1 : self.N_step
            ],  # [G, N_step-1, N_atm, 3]
            "timesteps": timesteps,
            "log_prob_old": torch.stack(log_prob_old[:-1], dim=1),  # [G, N_step-1]
            "rewards": rewards,  # [G]
            "s": pred_dict["s"],
            "z": pred_dict["z"],
            "s_inputs": pred_dict["s_inputs"],
            "input_feature_dict": batch["input_feature_dict"],
            "label_full_dict": batch["label_full_dict"],
            "label_dict": batch["label_dict"],
            "lig_mask": lig_mask,
        }

        return sample_buffer

    def update(self) -> None:
        """
        Apply gradient clipping to model parameters.
        """
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict, t_idx: int, mode: str = "rl_train") -> None:
        """
        Performs a single gradient update on a mini-batch for a SPECIFIC TIMESTEP t.
        """
        self.model.train()
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "float16"),
        )

        with enable_amp:
            timestep_index = batch["timesteps"][t_idx]
            c_tau = self.noise_schedule[timestep_index]
            c_tau_next = self.noise_schedule[timestep_index + 1]

            forward_batch = {
                "input_feature_dict": {
                    **batch["input_feature_dict"],
                    "x_l": batch["x_ls"][:, t_idx],
                    "x_l_next": batch["x_l_nexts"][:, t_idx],
                    "c_tau": c_tau,
                    "c_tau_next": c_tau_next,
                    "s": batch["s"],
                    "z": batch["z"],
                    "s_inputs": batch["s_inputs"],
                },
                "label_full_dict": batch["label_full_dict"],
                "label_dict": batch["label_dict"],
            }

            pred_dict_cur, _ = self.model_forward(
                forward_batch, mode, model_type="policy"
            )

            with torch.no_grad():
                pred_dict_ref, _ = self.model_forward(
                    forward_batch, mode, model_type="ref"
                )

            loss, loss_dict = self.get_loss(
                batch=batch,
                pred_dict_cur=pred_dict_cur,
                pred_dict_ref=pred_dict_ref,
                t_idx=t_idx,
            )

        if self.configs.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # Global training step used for accumulation logic
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            # Unscale gradients before clipping
            scaler.unscale_(self.optimizer)
            # Apply gradient clipping
            self.update()
            # Optimizer and scaler step
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            if hasattr(self, "ema_wrapper"):
                self.ema_wrapper.update()

        for key, value in loss_dict.items():
            if is_loss_nan_check(value):
                break
            self.train_metric_wrapper.add(key, value.detach(), namespace="train")

        torch.cuda.empty_cache()

    def progress_bar(self, desc: str = "") -> None:
        """
        Update the training progress bar on the master rank.

        Args:
            desc (str): Description to display on the progress bar.
        """
        if DIST_WRAPPER.rank != 0:
            return
        eval_steps = self.configs.eval_interval * self.iters_to_accumulate
        if self.global_step % eval_steps == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step % eval_steps,
                    eval_steps,
                )
            )
            self._ipbar = iter(self._pbar)

        try:
            step = next(self._ipbar)
            self._pbar.set_description(
                f"[step {self.step}: {step}/{eval_steps}] {desc}"
            )
        except StopIteration:
            # If the iterator is exhausted, just return
            pass

    def run(self) -> None:
        """
        Main entry for the RL Trainer.
        Handles the complete training cycle including evaluation, logging, and checkpointing.
        """
        if self.configs.eval_only or self.configs.eval_first:
            self.evaluate()
            if self.configs.eval_only:
                return
        use_ema = hasattr(self, "ema_wrapper")
        self.print(f"Using ema: {use_ema}")
        epoch = int(getattr(self, "epoch", 0))
        batches_in_epoch = int(getattr(self, "batches_in_epoch", 0))

        # Skip already-consumed batches exactly once on the first pass
        if batches_in_epoch > 0:
            self.train_dl = skip_first_batches(self.train_dl, batches_in_epoch)

        # Set sampler epoch BEFORE creating iterator so all ranks are aligned
        if self.use_ddp and hasattr(self.train_dl.sampler, "set_epoch"):
            self.train_dl.sampler.set_epoch(epoch)

        train_iterator = iter(self.train_dl)
        current_batches_in_epoch = batches_in_epoch
        while self.step < self.configs.max_steps:
            # --- 1. SAMPLING PHASE ---
            self.print(f"Step {self.step}: Sampling new experience batch...")
            try:
                batch = next(train_iterator)
                current_batches_in_epoch += 1
            except StopIteration:
                epoch += 1
                if self.use_ddp and hasattr(self.train_dl.sampler, "set_epoch"):
                    self.train_dl.sampler.set_epoch(epoch)
                train_iterator = iter(self.train_dl)
                batch = next(train_iterator)
                current_batches_in_epoch = 1

            batch = to_device(batch, self.device)
            experience_cache = self.sample_diffusion(batch)

            # Separate tensor data from non-tensor data
            tensor_cache = {
                "x_ls": experience_cache["x_ls"],
                "x_l_nexts": experience_cache["x_l_nexts"],
                "log_prob_old": experience_cache["log_prob_old"],
                "rewards": experience_cache["rewards"],
            }

            non_tensor_cache_data = {
                k: experience_cache[k]
                for k in (
                    "input_feature_dict",
                    "label_full_dict",
                    "label_dict",
                    "timesteps",
                    "s_inputs",
                    "s",
                    "z",
                    "lig_mask",
                )
            }

            # --- 2. SHUFFLE AND SPLIT THE CACHE ---
            shuffled_tensor_cache = shuffle_tensor_dict(tensor_cache)
            mini_batches = split_tensor_dict(
                shuffled_tensor_cache, self.configs.rl.update_batch_size
            )

            # --- 3. TRAINING PHASE ---
            self.print(
                f"Starting training phase on {len(mini_batches)} mini-batches..."
            )
            for mini_batch in mini_batches:
                if self.step >= self.configs.max_steps:
                    break
                update_batch = {**mini_batch, **non_tensor_cache_data}

                for _ in range(self.configs.rl.num_inner_iterations):
                    # Exclude the last deterministic step
                    for t_idx in torch.randperm(self.N_step - 1)[
                        : int(0.85 * (self.N_step - 1))
                    ]:
                        if self.step >= self.configs.max_steps:
                            break

                        self.train_step(update_batch, t_idx)

                        is_update_step = (
                            self.global_step + 1
                        ) % self.iters_to_accumulate == 0
                        step_need_log = (self.step + 1) % self.configs.log_interval == 0

                        is_update_ref_model = (
                            self.configs.rl.ref_update_interval > 0
                            and (self.step + 1) % self.configs.rl.ref_update_interval
                            == 0
                        )

                        step_need_eval = (
                            self.configs.eval_interval > 0
                            and (self.step + 1) % self.configs.eval_interval == 0
                        )
                        step_need_save = (
                            self.configs.checkpoint_interval > 0
                            and (self.step + 1) % self.configs.checkpoint_interval == 0
                        )

                        self.global_step += 1

                        if is_update_step:
                            self.step += 1
                            is_last_step = self.step >= self.configs.max_steps

                            if is_update_ref_model:
                                self.sync_ref_model()

                            if step_need_log or is_last_step:
                                metrics = self.train_metric_wrapper.calc()
                                self.print(f"Step {self.step} train: {metrics}")
                                last_lr = self.lr_scheduler.get_last_lr()
                                if DIST_WRAPPER.rank == 0:
                                    if self.configs.use_wandb:
                                        lr_dict = {"train/lr": last_lr[0]}
                                        for group_i, group_lr in enumerate(last_lr):
                                            lr_dict[f"train/group{group_i}_lr"] = (
                                                group_lr
                                            )
                                        wandb.log(lr_dict, step=self.step)
                                    self.print(f"Step {self.step}, lr: {last_lr}")
                                if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                                    wandb.log(metrics, step=self.step)

                            if step_need_save or is_last_step:
                                self.save_checkpoint(
                                    epoch=epoch,
                                    batches_in_epoch=current_batches_in_epoch,
                                )
                                if use_ema:
                                    self.ema_wrapper.apply_shadow()
                                    self.save_checkpoint(
                                        epoch=epoch,
                                        batches_in_epoch=current_batches_in_epoch,
                                        ema_suffix=f"_ema_{self.ema_wrapper.decay}",
                                    )
                                    self.ema_wrapper.restore()

                            if step_need_eval or is_last_step:
                                self.evaluate()

                if self.step >= self.configs.max_steps:
                    break


def main() -> None:
    """
    Parse configurations and start the training process.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logging.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    print(f"Run Name: {configs.run_name}")
    trainer = PiFoldTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
