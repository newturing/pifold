# RL Training

This document describes how to train **$\pi$-Fold**: fine-tuning a pretrained Protenix co-folding model with GRPO and physics-based rewards (Rosetta energy, Vina affinity, ligand strain). RL training starts from a pretrained checkpoint and does not train from scratch.

Before you start, finish the [Installation](../README.md#-installation) steps in the README, including PyRosetta and Smina.

## 1. Preparing the datasets

### 1.1 Download protein data

To download the [wwPDB dataset](https://www.wwpdb.org/) and preprocessed training data, you need at least 2T disk space.

Set the `PROTENIX_ROOT_DIR` environment variable to the data directory, then download and extract the preprocessed databases with the provided script:

```bash
# Set your data root directory
export PROTENIX_ROOT_DIR=/path/to/your/data_root
mkdir -p $PROTENIX_ROOT_DIR

# --full downloads every component needed for training
bash scripts/database/download_protenix_data.sh --full
```

All dataset paths in [configs/configs_data.py](../configs/configs_data.py) are resolved relative to `PROTENIX_ROOT_DIR`. If the variable is not set, it defaults to your home directory.

The data processing scripts are also available. See [prepare_training_data.md](./prepare_training_data.md) to generate `{dataset}_bioassembly` and `indices`, and [msa_template_pipeline.md](./msa_template_pipeline.md) to generate `mmcif_msa_template` and `seq_to_pdb_index.json`.

### 1.2 Download PH4 data

The PH4 data is hosted on Hugging Face at [tai-dang11/pifold_dataset](https://huggingface.co/datasets/tai-dang11/pifold_dataset). All archives below are extracted into `$PROTENIX_ROOT_DIR`.

**CCD components.** Download the CCD component archives, which already include the PH4 ligands. Do this first, since processing the PH4 structures needs these ligand definitions:

```bash
cd $PROTENIX_ROOT_DIR

wget https://huggingface.co/datasets/tai-dang11/pifold_dataset/resolve/main/ccd_cif.tar.gz
wget https://huggingface.co/datasets/tai-dang11/pifold_dataset/resolve/main/ccd_rdkit_mol.tar.gz
tar -xzf ccd_cif.tar.gz -C $PROTENIX_ROOT_DIR         # -> $PROTENIX_ROOT_DIR/ccd_cif/{CCD}.cif
tar -xzf ccd_rdkit_mol.tar.gz -C $PROTENIX_ROOT_DIR   # -> $PROTENIX_ROOT_DIR/ccd_rdkit_mol/{CCD}.pkl
rm ccd_cif.tar.gz ccd_rdkit_mol.tar.gz
```

If you use these archives, you can skip steps 1–2 in [Section 1.4](#14-ligand-strain-energy-cache). You still need step 3 to compute strain energies for the new ligands.

**PH4 distillation.** Download the raw PH4 structures into `mmcif_ph4/`:

```bash
wget https://huggingface.co/datasets/tai-dang11/pifold_dataset/resolve/main/mmcif_ph4.tar.gz
tar -xzf mmcif_ph4.tar.gz -C $PROTENIX_ROOT_DIR       # -> $PROTENIX_ROOT_DIR/mmcif_ph4/
rm mmcif_ph4.tar.gz
```

Then build `mmcif_ph4_bioassembly/` and the PH4 indices, either by processing the raw structures yourself (see [prepare_training_data.md](./prepare_training_data.md)):

```bash
# -d disables the wwPDB filters, since the PH4 structures are distilled model outputs
python3 scripts/prepare_training_data.py \
    -i $PROTENIX_ROOT_DIR/mmcif_ph4 \
    -o $PROTENIX_ROOT_DIR/indices/ph4_distill_60k.csv \
    -b $PROTENIX_ROOT_DIR/mmcif_ph4_bioassembly \
    -n [num_cpu] \
    -d
```

**Filter the PH4 indices to interfaces.** After generating `ph4_distill_60k.csv`, keep only rows where `type == "interface"`, so the dataloader always samples a protein–ligand pair rather than a single chain. The filtered indices are written to a new `ph4_distill_60k_lig_prot.csv`, which is used for training. The original CSV is kept unchanged:

```python
import os
import pandas as pd

PROTENIX_ROOT_DIR = os.environ["PROTENIX_ROOT_DIR"]
df = pd.read_csv(f"{PROTENIX_ROOT_DIR}/indices/ph4_distill_60k.csv")
df[df["type"] == "interface"].to_csv(
    f"{PROTENIX_ROOT_DIR}/indices/ph4_distill_60k_lig_prot.csv", index=False
)
```



### 1.3 Data hierarchy

The data pipeline loads CCD components one at a time, from `ccd_cif/{CCD}.cif` (`ccd_components_dir`) and `ccd_rdkit_mol/{CCD}.pkl` (`ccd_components_rdkit_mol_dir`), with one file per CCD code.

After downloading both the protein data and the PH4 data, the data hierarchy should look like this:

```bash
├── ccd_cif # one CIF file per CCD component, e.g. ATP.cif
│   ├── ATP.cif
│   └── ...
├── ccd_rdkit_mol # one pickle per CCD component, e.g. ATP.pkl -> {"ATP": rdkit.Chem.Mol}
│   ├── ATP.pkl
│   └── ...
├── common
│   ├── clusters-by-entity-40.txt # cluster file for identity antibody
│   ├── components.cif.global_energy.pkl # UFF global-minimum energy per ccd, used by the strain reward (see Section 1.4)
│   ├── obsolete_release_date.csv # release date of obsolete pdb
│   ├── obsolete_to_successor.json # mapping from obsolete pdb to successor pdb
│   ├── release_date_cache.json # cache of release date
│   └── seq_to_pdb_index.json # mapping from sequence to directory ID
├── indices
│   ├── posebusters_indices_mainchain_interface.csv # indices for posebusters dataset
│   ├── recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt # indices for recentPDB evaluation
│   ├── recentPDB_low_homology_maxtoken1536.csv # indices for recentPDB evaluation
│   ├── ph4_distill_60k.csv # indices for PH4 distillation dataset
│   ├── ph4_distill_60k_lig_prot.csv # PH4 indices filtered to protein–ligand interfaces, used for training
│   └── weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz # indices for wwpdb training dataset
├── mmcif # raw mmcif data / template search database
├── mmcif_bioassembly # preprocessed wwPDB structural data / training cache
├── mmcif_ph4 # raw mmcif data for PH4 distillation
├── mmcif_ph4_bioassembly # preprocessed PH4 distillation structural data
├── mmcif_msa_template # msa / template files for training
├── posebusters_bioassembly # preprocessed posebusters structural data
├── posebusters_mmcif # raw mmcif data
├── recentPDB_bioassembly # preprocessed recentPDB structural data
├── rna_msa # RNA MSA files
└── search_database # sequence databases for MSA / template search
```

### 1.4 Ligand strain-energy cache

The strain-energy reward compares each predicted ligand pose with the ligand's global-minimum UFF energy. Those reference energies are read from `common/components.cif.global_energy.pkl` (`ccd_components_mol_energy_file` in [configs/configs_data.py](../configs/configs_data.py)). If you add ligands that are not in the CCD, such as the PH4 compounds, register them first and then rebuild the cache:

Each SDF file is registered under its file name in upper case, e.g. `abc123.sdf` becomes CCD code `ABC123`. Ligands whose fingerprint matches an existing CCD molecule are skipped, as are codes that already have a file. The scripts write to the `ccd_components_rdkit_mol_dir`, `ccd_components_dir` and `ccd_components_mol_energy_file` paths from [configs/configs_data.py](../configs/configs_data.py) unless you pass other paths.

```bash
# 1. Add new ligands (SDF files) to ccd_rdkit_mol/ as {CCD}.pkl
python scripts/update_rdkit_mols.py \
    --sdf_dir /path/to/ligand_sdfs \
    --rdkit_mol_dir $PROTENIX_ROOT_DIR/ccd_rdkit_mol

# 2. Add the same ligands to ccd_cif/ as {CCD}.cif
python scripts/update_components_cif.py \
    --sdf_dir /path/to/ligand_sdfs \
    --ccd_cif_dir $PROTENIX_ROOT_DIR/ccd_cif \
    --rdkit_mol_dir $PROTENIX_ROOT_DIR/ccd_rdkit_mol

# 3. Compute UFF global-minimum energies for molecules missing from the cache
#    (add --overwrite to recompute every molecule)
python scripts/update_mol_energy.py \
    --input_dir $PROTENIX_ROOT_DIR/ccd_rdkit_mol \
    --output $PROTENIX_ROOT_DIR/common/components.cif.global_energy.pkl \
    --num-confs 50
```

Run step 2 after step 1: it only writes CIF files for ligands that step 1 registered.

By default, the code runs PyRosetta with unrecognized CCD codes ignored, so ligands missing from its CCD library are skipped when scoring. This step is optional: run it only if you want PyRosetta to recognize the new ligands.

PyRosetta reads CCD definitions from per-letter `components.<X>.cif` files. To make the new ligands visible to PyRosetta, rebuild those files from `ccd_cif/` into PyRosetta's CCD library directory:

```bash
# Path is usually under site-packages, e.g. ~/miniconda3/envs/af3/lib/python3.11/site-packages/pyrosetta/database/chemical/pdb_components
python scripts/update_ccd_pyrosetta.py --lib-dir ~/miniconda3/envs/{env_name}/lib/python3.11/site-packages/pyrosetta/database/chemical/pdb_components
```

## 2. Download the pretrained checkpoint

RL fine-tuning needs a pretrained policy and a frozen reference model for the KL penalty. Normally both are the same checkpoint:

```bash
wget -P af3-dev/release_model/ https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_default_v0.5.0.pt
```

Other supported checkpoints, such as `protenix_base_default_v0.5.0` and `protenix_base_default_v1.0.0`, are listed in [pifold/web_service/dependency_url.py](../pifold/web_service/dependency_url.py) and [supported_models.md](./supported_models.md). If you switch checkpoints, change `--model_name` to match.

## 3. RL training demo

After installation and data preparation, launch RL training with:

```bash
chmod +x rl.sh
./rl.sh
```

By default the script runs on a single node with 8 GPUs through `torchrun`. Change `NUM_GPUS` in [rl.sh](../rl.sh) to fit your machine. Each launch picks a random `MASTER_PORT`, so several jobs can share one node.

### Key arguments

**Checkpoints**
* `model_name`: model configuration name. It must match the checkpoint, e.g. `"protenix_mini_default_v0.5.0"`.
* `load_checkpoint_path` / `load_ema_checkpoint_path`: initial weights for the policy and its EMA copy.
* `rl.load_ref_checkpoint_path`: weights for the frozen reference model used in the KL term. The reference model is re-synced from the policy every `rl.ref_update_interval` steps (default 50000).
* `load_params_only`: when `true`, loads only model weights and skips the optimizer state and step counter. Use `true` when starting from a pretrained model, and `false` to resume an interrupted RL run.

**GRPO sampling**
* `rl.sample_diffusion.N_sample`: number of structures sampled per input, i.e. the GRPO group size (default 24). Advantages are normalized within each group.
* `rl.sample_diffusion.N_step`: number of denoising steps in each RL trajectory (default 20).
* `rl.update_batch_size`: mini-batch size for one policy update step (default 32).
* `rl.num_inner_iterations`: number of policy updates per batch of rollouts (default 1).

**Loss**
* `rl.loss.beta1`: weight of the structure KL divergence against the reference model (default `2e-3`).
* `rl.loss.beta2`: weight of the ligand KL divergence (default `1e-6`).
* `rl.loss.clip_range`: PPO-style clipping range for the policy ratio (default 0.25).
* `rl.loss.adv_clip_max`: maximum absolute advantage, to limit outliers (default 5.0).

**Rewards**

The total reward is `struc_w * rosetta + lig_w * vina + strain_w * strain`:
* `rl.reward.ligand`: when `true`, adds the Smina/Vina affinity and ligand strain rewards to the PyRosetta structure reward. When `false`, only the structure reward is used.
* `rl.reward.struc_w` / `rl.reward.lig_w` / `rl.reward.strain_w`: reward weights (defaults 1.0 / 1.0 / 0.3).
* `rl.reward.pyrosetta.*`: Rosetta score function (default `beta_jan25`), extra term weights, worker count, timeout, and the score range used for normalization.
* `rl.reward.smina.executable_path`: path to the Smina binary (default `./smina`, the repository root, as installed in the README).
* `rl.reward.smina.*` / `rl.reward.strain_energy.*`: worker counts, timeouts, and score ranges for the ligand rewards.
* `rl.reward.clean_files`: delete the temporary PDB/SDF files written for scoring.

Reward computation runs on the CPU in worker pools. If rollouts are slow, check the `num_workers` settings against the number of CPU cores per GPU.

**Optimization**
* `lr`, `warmup_steps`, `lr_scheduler`: learning rate schedule. `rl.sh` uses `2e-5` with 1000 warmup steps and cosine annealing.
* `iters_to_accumulate`: gradient accumulation steps.
* `ema_decay`: EMA decay rate for the model weights (default 0.999).
* `train_crop_size`: token crop size for training samples (1024 in `rl.sh`).
* `max_steps`: total number of training steps.
* `dtype`: `"bf16"` for mixed precision or `"fp32"`.
* `triangle_attention` / `triangle_multiplicative`: kernels for the triangular modules. We recommend `"cuequivariance"`. See [kernels.md](./kernels.md).

**Data**
* `data.train_sets`: comma-separated training datasets, e.g.:
  * `weightedPDB_before2109_wopb_nometalc_0925`: wwPDB training set (from Section 1.1).
  * `ph4_distillation`: PH4 protein–ligand distillation set (from Section 1.2).
* `data.train_sampler.train_sample_weights`: sampling weight for each training set, in the same order.
* `data.<dataset>.base_info.indices_fpath`: overrides the indices file for a dataset. `rl.sh` points `ph4_distillation` at the filtered `$PROTENIX_ROOT_DIR/indices/ph4_distill_60k_lig_prot.csv`, which is also the config default.
* `data.weightedPDB_before2109_wopb_nometalc_0925.cropping_configs.method_weights`: weights for the contiguous / spatial / spatial-interface cropping methods. `1.0,0.0,0.0` uses contiguous cropping only.
* `data.test_sets`: evaluation datasets, e.g. `recentPDB_1536_sample384_0925,posebusters_0925`.
* `data.msa.enable_rna_msa false` / `data.template.enable_prot_template false`: RNA MSA and protein templates are disabled for RL training.

**Logging and checkpointing**
* `base_dir` / `run_name`: outputs are written to `${base_dir}/${run_name}`, with checkpoints under `checkpoints/`.
* `checkpoint_interval`, `log_interval`, `eval_interval`: how often, in steps, to save, log, and evaluate.
* `use_wandb` / `project`: log to Weights & Biases under the given project. rank 0 logs the reward metrics (`reward_mean`, `reward_std`, `struct_reward_mean`, `lig_reward_mean`, `strain_reward_mean`).
* `seed`, `deterministic`, `deterministic_seed`: settings for reproducibility.

You can pass any other config key as `--<ARGS_KEY> <ARGS_VALUE>`. For example:

```bash
torchrun --nproc_per_node=8 runner/train_rl.py \
    --model_name "protenix_mini_default_v0.5.0" \
    --load_checkpoint_path /path/to/protenix_mini_default_v0.5.0.pt \
    --load_ema_checkpoint_path /path/to/protenix_mini_default_v0.5.0.pt \
    --rl.load_ref_checkpoint_path /path/to/protenix_mini_default_v0.5.0.pt \
    [OTHER_ARGS]
```

### Resuming training

To resume an interrupted run, point `load_checkpoint_path` / `load_ema_checkpoint_path` at the latest `{step}.pt` / `{step}_ema_*.pt` files under `${base_dir}/${run_name}/checkpoints/` and set `--load_params_only false`. The optimizer state (`{step}_optimizer.pt`) and the step counter are then restored.
