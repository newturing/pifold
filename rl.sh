#!/bin/bash

# --- 1. Data Config ---
# All dataset paths in configs/configs_data.py are resolved relative to PROTENIX_ROOT_DIR.
: "${PROTENIX_ROOT_DIR:?Please set PROTENIX_ROOT_DIR to your data root (see docs/training_rl.md)}"
export PROTENIX_ROOT_DIR

# --- 2. Checkpoint Config ---
base_dir=./af3-dev/checkpoints/pifold/base/rl/mini_test
checkpoint_path=af3-dev/release_model/protenix_mini_default_v0.5.0.pt
ema_checkpoint_path=af3-dev/release_model/protenix_mini_default_v0.5.0.pt
ref_checkpoint_path=af3-dev/release_model/protenix_mini_default_v0.5.0.pt

# --- 3. Distributed Training Config ---
MASTER_PORT=$((29500 + RANDOM % 1000))
NUM_GPUS=8

# --- 4. Execution ---
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master-port=$MASTER_PORT ./runner/train_rl.py \
    --model_name "protenix_mini_default_v0.5.0" \
    --run_name mini \
    --seed 42 \
    --deterministic true \
    --deterministic_seed true \
    --base_dir ${base_dir} \
    --dtype bf16 \
    --project pifold_test \
    --use_wandb true \
    --eval_interval 100000 \
    --log_interval 10 \
    --checkpoint_interval 500 \
    --load_params_only true \
    --ema_decay 0.999 \
    --train_crop_size 1024 \
    --max_steps 100000 \
    --lr 2e-5 \
    --warmup_steps 1000 \
    --lr_scheduler "cosine_annealing" \
    --iters_to_accumulate 4 \
    --rl.loss.beta1 2e-3 \
    --rl.reward.ligand true \
    --rl.sample_diffusion.N_step 20 \
    --rl.sample_diffusion.N_sample 24 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --load_checkpoint_path ${checkpoint_path} \
    --load_ema_checkpoint_path ${ema_checkpoint_path} \
    --rl.load_ref_checkpoint_path ${ref_checkpoint_path} \
    --data.msa.enable_rna_msa false \
    --data.template.enable_prot_template false \
    --data.train_sampler.train_sample_weights 0.5,0.25,0.25 \
    --data.train_sets weightedPDB_before2109_wopb_nometalc_0925,ph4_distillation,monomer_distillation \
    --data.weightedPDB_before2109_wopb_nometalc_0925.cropping_configs.method_weights 1.0,0.0,0.0 \
    --data.ph4_distillation.base_info.indices_fpath ${PROTENIX_ROOT_DIR}/indices/ph4_distill_60k_lig_prot.csv \
    --data.monomer_distillation.base_info.indices_fpath ${PROTENIX_ROOT_DIR}/indices/monomer_distill_76k_lig_prot.csv \
    --data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
