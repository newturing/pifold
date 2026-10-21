import logging
import os
import traceback
import urllib.request
from contextlib import nullcontext
from collections import defaultdict
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Mapping, Dict, List
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict

import pandas as pd
from rdkit import RDLogger
from torch.utils.data import DataLoader
from pifold.data.pipeline.dataloader import KeySumBalancedSampler
from pifold.data.pipeline.dataset import get_eval_dataset
from pifold.model.reward import RewardModel
from pifold.utils.torch_utils import collate_fn_first

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from pifold.config import parse_configs, parse_sys_args
from pifold.data.inference.infer_dataloader import get_inference_dataloader
from pifold.model.pifold import PiFold
from pifold.utils.distributed import DIST_WRAPPER
from pifold.utils.seed import seed_everything
from pifold.utils.torch_utils import to_device
from pifold.utils.metrics import SimpleMetricAggregator
from pifold.web_service.dependency_url import URL
from runner.dumper import DataDumper
from runner.inference import download_ccd_cache

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")  # disables all RDKit warnings


def get_eval_dataloader(
    configs: Any, dataset_name: str, test_dataset: Any
) -> DataLoader:
    """
    Creates a dataloader for a specific evaluation dataset.
    """
    logger.info(f"Loading evaluation dataset: {dataset_name}")
    logger.info(f"Successfully loaded {len(test_dataset)} samples for evaluation.")

    sampler = None
    if DIST_WRAPPER.world_size > 1:
        sampler = KeySumBalancedSampler(
            test_dataset, key="num_tokens", seed=configs.seed
        )

    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=configs.num_workers,
        sampler=sampler,
        collate_fn=collate_fn_first,
    )
    return dataloader


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )
        self.init_reward_model()

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        self.print(
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
            dist.init_process_group(backend="nccl")

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH environment variable "
                "per instructions at "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
            logging.info(
                "Kernels will be compiled when DS4Sci_EvoformerAttention "
                "is first called."
            )

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "fast_layernorm")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Finished environment initialization.")

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """
        Initialize the Protenix model and move it to the appropriate device.
        """
        self.model = PiFold(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = (
            f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
            if self.configs.load_checkpoint_path == ""
            else self.configs.load_checkpoint_path
        )
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = list(checkpoint["model"].keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model parameters: {count_parameters(self.model):.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        """
        Initialize the data dumper for saving predictions.

        Args:
            need_atom_confidence (bool): Whether to dump atom-level confidence.
            sorted_by_ranking_score (bool): Whether to sort results by ranking score.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def init_reward_model(self) -> None:
        reward_cfg = self.configs.rl.reward
        self.scorer = RewardModel(
            pyrosetta_config=reward_cfg.pyrosetta,
            smina_config=reward_cfg.smina,
            strain_energy_config=reward_cfg.strain_energy,
            clean_files=True,
            ligand=reward_cfg.ligand,
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Run model prediction on the provided data.

        Args:
            data (Mapping[str, Mapping[str, Any]]): Input data dictionary.

        Returns:
            dict[str, torch.Tensor]: Prediction results.
        """
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

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                current_step=None,
                symmetric_permutation=None,
            )

        return prediction

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        """
        Update the model's configuration.

        Args:
            new_configs (Any): New configuration object.
        """
        self.model.configs = new_configs

    def process_rewards(
        self, sample_name: str, seed: int, structure_paths: Dict, lig_paths: Dict
    ) -> Dict:
        """Handles scoring and granular data extraction."""
        reward_cfg = self.configs.rl.reward
        lig_w, strain_w, struc_w = (
            reward_cfg.lig_w,
            reward_cfg.strain_w,
            reward_cfg.struc_w,
        )

        reward_results = self.scorer.score(structure_paths, lig_paths)
        struc_results, lig_results, strain_results = (r[0] for r in reward_results)
        struc_rewards, lig_rewards, strain_rewards = (r[1] for r in reward_results)
        rewards = (
            struc_w * struc_rewards + lig_w * lig_rewards + strain_w * strain_rewards
        )
        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] {sample_name} "
            f"Reward: {rewards.mean().item():.3f}; Std: {rewards.std().item():.3f}"
        )

        # 1. Structure Scores
        struc_scores = [
            float(v["score"])
            for v in struc_results.values()
            if v and v.get("score") is not None
        ]
        struc_tensor = torch.tensor(struc_scores)
        struc_mean = struc_tensor.mean().item() if struc_tensor.numel() > 0 else None
        struc_std = (
            struc_tensor.std().item() if struc_tensor.numel() > 1 else float("nan")
        )
        if struc_mean is not None:
            logger.info(
                f"[Rank {DIST_WRAPPER.rank}] {sample_name} "
                f"Rosetta: {struc_mean:.3f}; Std: {struc_std:.3f}"
            )

        # 2. Detailed File-Level results
        file_details = []
        ccd_stats = defaultdict(lambda: {"affinity": [], "strain": []})

        for i, (struct_path, lig_mapping) in enumerate(lig_paths.items()):
            res_data = struc_results.get(struct_path, {})

            # Group by CCD for this specific file
            file_lig_scores = defaultdict(list)
            file_strain_scores = defaultdict(list)

            if lig_results:
                for (s_p, l_p), v in lig_results.items():
                    if s_p == struct_path and v.get("success", False):
                        ccd = (
                            v.get("ccd")
                            or os.path.basename(l_p).split("_ligand_")[1].split("_")[0]
                        )
                        score = v["score"]
                        strain = strain_results.get((s_p, l_p), {}).get(
                            "score", float("nan")
                        )

                        file_lig_scores[ccd].append(score)
                        file_strain_scores[ccd].append(strain)
                        ccd_stats[ccd]["affinity"].append(score)
                        ccd_stats[ccd]["strain"].append(strain)

            file_details.append(
                {
                    "pdb_id": sample_name,
                    "seed": seed,
                    "file_name": os.path.basename(struct_path),
                    "reward": float(rewards[i]),
                    "rosetta": float(res_data.get("score", float("nan"))),
                    "ligand_affinity": dict(file_lig_scores),
                    "strain_energy": dict(file_strain_scores),
                }
            )

        for c, s in ccd_stats.items():
            v_a, v_s = (
                np.array(s["affinity"]),
                np.array(s["strain"])[~np.isnan(s["strain"])],
            )
            a_m, a_s = np.mean(v_a), (np.std(v_a) if len(v_a) > 1 else float("nan"))
            st_m, st_s = (
                (np.mean(v_s) if len(v_s) > 0 else float("nan")),
                (np.std(v_s) if len(v_s) > 1 else float("nan")),
            )

            logger.info(
                f"[Rank {DIST_WRAPPER.rank}] {sample_name} {c}: "
                f"Affinity {a_m:.3f}, Std {a_s:.3f} | "
                f"Strain {st_m:.3f}, Std {st_s:.3f}"
            )

        # 3. Aggregate Ligand Stats for the summary row
        affinity_means = [
            np.mean(v["affinity"]) for v in ccd_stats.values() if len(v["affinity"]) > 0
        ]

        strain_means = [
            np.mean(v["strain"]) for v in ccd_stats.values() if len(v["strain"]) > 0
        ]
        summary_ligands = {
            "ccd_list": list(ccd_stats.keys()),
            "affinity_mean": np.mean(affinity_means)
            if affinity_means
            else float("nan"),
            "strain_mean": np.mean(strain_means) if strain_means else float("nan"),
        }

        return {
            "summary": {
                "sample_name": sample_name,
                "seed": seed,
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "rosetta_mean": struc_mean,
                "rosetta_std": struc_std,
                **summary_ligands,
            },
            "details": file_details,
            "ccd_metrics": ccd_stats,
        }


def download_infercence_cache(configs: Any) -> None:
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        status = f"\r[{bar}] {percent:.1f}%"
        print(status, end="", flush=True)

        if downloaded >= total_size:
            print()

    def download_from_url(tos_url, checkpoint_path, check_weight=True):
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )
        if check_weight:
            try:
                ckpt = torch.load(checkpoint_path)
                del ckpt
            except:
                os.remove(checkpoint_path)
                raise RuntimeError(
                    "Download model checkpoint failed, please download by yourself with "
                    f"wget {tos_url} -O {checkpoint_path}"
                )

    download_ccd_cache(configs)

    for cache_name in (
        "pdb_cluster_file",
        "obsolete_release_data_csv",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    if configs.load_checkpoint_path:
        if not os.path.isfile(configs.load_checkpoint_path):
            raise FileNotFoundError(
                f"Invalid checkpoint path: {configs.load_checkpoint_path}"
            )
        checkpoint_path = configs.load_checkpoint_path
    else:
        checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint_dir = configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}...\n to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    if "esm" in configs.model_name:  # currently esm only support 3b model
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path)
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path2)
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"

        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path)

        esm_3b_ism_ckpt_path2 = (
            f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"
        )
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)


def update_inference_configs(configs: Any, n_token: int) -> Any:
    """
    Adjust inference configurations based on the number of tokens to avoid OOM.

    Args:
        configs (Any): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        Any: Updated configurations.
    """
    # Adjust configurations based on sequence length to manage memory usage
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    if configs.input_json_path and opexists(configs.input_json_path):
        logger.info(f"Mode: JSON Inference from {configs.input_json_path}")
        dataloaders = {"json": get_inference_dataloader(configs=configs)}
        is_eval_mode = False
    else:
        logger.info("Mode: Evaluation on all test datasets")
        error_dir = opjoin(configs.dump_dir, "ERR")
        dataloaders = get_eval_dataset(configs, error_dir=error_dir)
        is_eval_mode = True

    metric_aggregator = SimpleMetricAggregator(["avg"])

    for dataset_name, dataset in dataloaders.items():
        dataloader = (
            get_eval_dataloader(configs, dataset_name, dataset)
            if is_eval_mode
            else dataset
        )
        num_data = len(dataloader.dataset)
        results_list = []
        detailed_list = []
        for seed in configs.seeds:
            seed_everything(seed=seed, deterministic=configs.deterministic)
            for i, batch in enumerate(tqdm(dataloader)):
                error_msg = "None"
                try:
                    if is_eval_mode:
                        data = batch
                        atom_array = data["cropped_atom_array"]
                        basic = data["basic"]
                        sample_name = basic["pdb_id"]
                        entity_poly_type = None
                        n_asym, n_token, n_atom, n_msa = (
                            basic["N_asym"].item(),
                            basic["N_token"].item(),
                            basic["N_atom"].item(),
                            basic["N_msa"].item(),
                        )
                    else:
                        data, atom_array, error_msg = batch[0]
                        sample_name = data["sample_name"]
                        entity_poly_type = data["entity_poly_type"]
                        n_asym, n_token, n_atom, n_msa = (
                            data["N_asym"].item(),
                            data["N_token"].item(),
                            data["N_atom"].item(),
                            data["N_msa"].item(),
                        )

                    prefix = (
                        f"{dataset_name}/{sample_name}" if is_eval_mode else sample_name
                    )
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank} ({i + 1}/{num_data})] {prefix}: "
                        f"N_asym {n_asym}, N_token {n_token}, N_atom {n_atom}, N_msa {n_msa}"
                    )

                    runner.update_model_configs(
                        update_inference_configs(configs, n_token)
                    )
                    prediction = runner.predict(data)
                    struct_paths, lig_paths, _ = runner.dumper.dump_structures(
                        dump_dir=runner.dump_dir,
                        pdb_id=sample_name,
                        seed=seed,
                        pred_coordinates=prediction["coordinate"],
                        atom_array=atom_array,
                        entity_poly_type=entity_poly_type,
                        return_path=True,
                        score_ligand=configs.rl.reward.ligand,
                    )

                    processed = runner.process_rewards(
                        sample_name, seed, struct_paths, lig_paths
                    )

                    # Collect Results
                    results_list.append(processed["summary"])
                    detailed_list.extend(processed["details"])

                    detail_file = opjoin(
                        runner.dump_dir,
                        f"detailed_running_rank_{DIST_WRAPPER.rank}.csv"
                    )

                    pd.DataFrame(processed["details"]).to_csv(
                        detail_file,
                        mode="a",
                        header=not opexists(detail_file),
                        index=False,
                    )

                    # Update Running Metrics
                    if processed["summary"]["rosetta_mean"] is not None:
                        metric_aggregator.add(
                            "rosetta", processed["summary"]["rosetta_mean"]
                        )
                    if processed["summary"]["reward_mean"] is not None:
                        metric_aggregator.add(
                            "reward", processed["summary"]["reward_mean"]
                        )
                    for ccd, stats in processed["ccd_metrics"].items():
                        metric_aggregator.add(
                            f"{ccd}_affinity", np.mean(stats["affinity"])
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing {sample_name}: {e}\n{traceback.format_exc()}"
                    )
                    torch.cuda.empty_cache()

        final_metrics = metric_aggregator.calc()  # This calls all_gather internally

        if DIST_WRAPPER.world_size > 1:
            all_results = [
                item
                for sub in DIST_WRAPPER.all_gather_object(results_list)
                for item in sub
            ]
            all_details = [
                item
                for sub in DIST_WRAPPER.all_gather_object(detailed_list)
                for item in sub
            ]
        else:
            all_results, all_details = results_list, detailed_list

        # --- OUTPUT (Rank 0 Only) ---
        if DIST_WRAPPER.rank == 0:
            save_inference_results(
                dataset_name,
                all_results,
                all_details,
                final_metrics,
                runner.dump_dir,
                configs,
            )


def save_inference_results(
    ds_name: str,
    results: List,
    details: List,
    metrics: Any,
    dump_dir: str,
    configs: Any,
):
    ckpt_name = os.path.splitext(
        os.path.basename(configs.load_checkpoint_path or configs.model_name)
    )[0]

    pd.DataFrame(results).sort_values(by="sample_name").to_csv(
        opjoin(dump_dir, f"{ds_name}_{ckpt_name}_summary.csv"), index=False
    )

    pd.DataFrame(details).sort_values(by="file_name").to_csv(
        opjoin(dump_dir, f"{ds_name}_{ckpt_name}_detailed.csv"), index=False
    )

    logger.info(f"Results for {ds_name} saved to {dump_dir}")


def main(configs: Any) -> None:
    # Runner
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    checkpoint_path = (
        f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
        if configs.load_checkpoint_path == ""
        else configs.load_checkpoint_path
    )
    logger.info(
        f"Inference by Protenix: base_model={model_name}; checkpoint={checkpoint_path}"
    )
    model_specfics_configs = ConfigDict(model_configs[model_name])
    # update model specific configs
    model_specfics_configs.model.N_cycle = configs.model.N_cycle
    model_specfics_configs.sample_diffusion.N_sample = configs.sample_diffusion.N_sample
    model_specfics_configs.sample_diffusion.N_step = configs.sample_diffusion.N_step
    configs.update(model_specfics_configs)
    logger.info(
        f"Triangle_multiplicative kernel: {configs.triangle_multiplicative}, Triangle_attention kernel: {configs.triangle_attention}"
    )
    logger.info(
        f"enable_diffusion_shared_vars_cache: {configs.enable_diffusion_shared_vars_cache}, enable_efficient_fusion: {configs.enable_efficient_fusion}, enable_tf32: {configs.enable_tf32}"
    )
    download_infercence_cache(configs)
    main(configs)


if __name__ == "__main__":
    run()
