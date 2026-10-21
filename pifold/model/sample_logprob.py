from typing import Any, Callable, Optional, Tuple

import math
import torch

from pifold.model.utils import append_dims, get_ancestral_step


def sde_step_with_logprob(
    x_l: torch.Tensor,
    x_denoised: torch.Tensor,
    c_tau: torch.Tensor,
    c_tau_next: torch.Tensor,
    noise_scale_lambda: float,
    step_scale_eta: float,
    x_l_next: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs one step of the reverse-time SDE and computes the log probability.

    Args:
        x_l (torch.Tensor): The current noisy sample (xt).
        x_denoised (torch.Tensor): The model's prediction of the clean sample (x0_hat).
        c_tau (torch.Tensor): The current noise level (sigma_t).
        c_tau_next (torch.Tensor): The next noise level (sigma_{t+1}).
        noise_scale_lambda (float): Scaling factor for the stochastic noise.
        step_scale_eta (float): Scaling factor for the step size.
        x_l_next (Optional[torch.Tensor]): The actual next state, if already known.
                                              Used for calculating log_prob of a given transition.

    Returns:
        Tuple containing:
        - x_l_next (torch.Tensor): The next sample in the trajectory (x_{t+1}).
        - log_prob (torch.Tensor): The log probability of the transition to x_l_next.
        - x_l_mean (torch.Tensor): The mean of the transition distribution.
        - std (torch.Tensor): The standard deviation of the transition distribution.
    """
    # Calculate step sizes based on the provided schedule and eta
    c_tau_down, c_tau_up = get_ancestral_step(c_tau, c_tau_next, eta=step_scale_eta)

    # Calculate the ODE step direction
    d = (x_l - x_denoised) / append_dims(c_tau, x_l.ndim)

    # Calculate the mean of the transition distribution (the deterministic part of the step)
    dt = c_tau_down - c_tau

    x_l_next_mean = x_l + d * dt

    # Calculate the standard deviation of the transition distribution (the stochastic part)
    std_c_tau = c_tau_up * noise_scale_lambda

    if x_l_next is None:
        # Sample the next state from the distribution N(mean, std^2)
        noise = torch.randn(
            size=x_l_next_mean.shape,
            device=x_l_next_mean.device,
            dtype=x_l_next_mean.dtype,
        )
        x_l_next = x_l_next_mean + noise * std_c_tau

    # Calculate the log probability using the Gaussian PDF formula
    # log P(x) = -0.5 * ((x - mu) / sigma)^2 - 0.5 * log(sigma**2) - 0.5 * log(2 * pi)
    log_prob = (
        -((x_l_next.detach() - x_l_next_mean) ** 2) / (2 * (std_c_tau**2))
        - torch.log(std_c_tau)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return x_l_next, log_prob, x_l_next_mean, std_c_tau


def compute_logprob(
    denoise_net: Callable,
    x_l: torch.Tensor,
    prev_sample: torch.Tensor,
    c_tau: torch.Tensor,
    c_tau_next: torch.Tensor,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    N_sample: int = 1,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    attn_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    enable_efficient_fusion: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the log probability of an action, with support for batch chunking to save memory.

    Args:
        ... (same as before) ...
        diffusion_chunk_size (Optional[int]): If specified, the batch dimension of inputs will be
                                              processed in chunks of this size.

    Returns:
        Tuple containing:
        - log_prob (torch.Tensor): The log probability of the action.
        - x_l_next (torch.Tensor): The next sample (same as input 'action').
        - x_l_next_mean (torch.Tensor): The mean of the transition distribution.
    """

    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _compute_chunk(
        x_l_chunk,
        x_l_next_chunk,
    ):
        """Processes a single chunk of the batch."""
        # 1. Prepare inputs
        x_l = x_l_chunk

        c_tau_batch = (
            c_tau.reshape((1,) * (len(batch_shape) + 1))
            .expand(*x_l.shape[:-2])
            .to(dtype)
        )

        # 2. Get model's prediction (x0_hat)
        x_denoised = denoise_net(
            x_noisy=x_l,
            t_hat_noise_level=c_tau_batch,
            input_feature_dict=input_feature_dict,  # Assumes this dict doesn't need chunking
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            chunk_size=attn_chunk_size,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        # 3. Use SDE step function to calculate log_prob, mean, and next_x
        x_l_next, log_prob, x_l_next_mean, std_c_tau = sde_step_with_logprob(
            x_l=x_l,
            x_denoised=x_denoised,
            c_tau=c_tau,
            c_tau_next=c_tau_next,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            x_l_next=x_l_next_chunk,
        )

        return log_prob, x_l_next, x_l_next_mean, std_c_tau

    batch_size = x_l.shape[0]

    if diffusion_chunk_size is None:
        log_prob, x_l_next, x_l_next_mean, std_c_tau = _compute_chunk(x_l, prev_sample)
    else:
        log_prob_chunks, x_l_next_chunks, x_l_next_mean_chunks, std_c_tau_chunks = (
            [],
            [],
            [],
            [],
        )
        num_chunks = (batch_size + diffusion_chunk_size - 1) // diffusion_chunk_size
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )

        for i in range(num_chunks):
            start_idx = i * diffusion_chunk_size
            end_idx = min((i + 1) * diffusion_chunk_size, batch_size)

            # Slice all tensors that have a batch dimension
            x_l_chunk = x_l[start_idx:end_idx]
            next_sample_chunk = prev_sample[start_idx:end_idx]

            # Process the chunk
            log_prob_chunk, x_l_next_chunk, x_l_next_mean_chunk, std_c_tau_chunk = (
                _compute_chunk(x_l_chunk, next_sample_chunk)
            )

            # Append results
            log_prob_chunks.append(log_prob_chunk)
            x_l_next_chunks.append(x_l_next_chunk)
            x_l_next_mean_chunks.append(x_l_next_mean_chunk)
            std_c_tau_chunks.append(std_c_tau_chunk)

        # Concatenate results from all chunks
        log_prob = torch.cat(log_prob_chunks, dim=0)
        x_l_next = torch.cat(x_l_next_chunks, dim=0)
        x_l_next_mean = torch.cat(x_l_next_mean_chunks, dim=0)
        std_c_tau = torch.cat(std_c_tau_chunks, dim=0)

    return log_prob, x_l_next, x_l_next_mean, std_c_tau


def sample_diffusion_with_logprob(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    beta: float = 0.004,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        beta (float): weight for KL loss.
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # Lists to store the trajectory information
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        all_x, all_log_probs, all_x_mean = [x_l.clone()], [], []

        for i, (c_tau, c_tau_next) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = x_l

            c_tau_batch = (
                c_tau.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_l,
                t_hat_noise_level=c_tau_batch,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            # Perform the SDE step and get log probability
            x_l_next, log_prob, x_l_next_mean, _ = sde_step_with_logprob(
                x_l, x_denoised, c_tau, c_tau_next, noise_scale_lambda, step_scale_eta
            )

            x_l = x_l_next

            # Store results
            all_x.append(x_l.clone())
            all_x_mean.append(x_l_next_mean.clone())
            all_log_probs.append(log_prob)

        return x_l, all_x, all_x_mean, all_log_probs

    if diffusion_chunk_size is None:
        x_final, all_x, all_x_mean, all_log_probs = _chunk_sample_diffusion(
            N_sample, inplace_safe=inplace_safe
        )
    else:
        x_final_chunks, all_x_chunks, all_x_mean_chunks, all_log_probs_chunks = (
            [],
            [],
            [],
            [],
        )
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_final, chunk_all_x, chunk_all_x_mean, chunk_all_log_probs = (
                _chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe)
            )
            x_final_chunks.append(chunk_x_final)
            all_x_chunks.append(chunk_all_x)
            all_x_mean_chunks.append(chunk_all_x_mean)
            all_log_probs_chunks.append(chunk_all_log_probs)

        x_final = torch.cat(x_final_chunks, dim=-3)  # [..., N_sample, N_atom, 3]
        all_x = [torch.cat(chunk, dim=-3) for chunk in zip(*all_x_chunks)]
        all_x_mean = [torch.cat(chunk, dim=-3) for chunk in zip(*all_x_mean_chunks)]
        all_log_probs = [
            torch.cat(chunk, dim=-1) for chunk in zip(*all_log_probs_chunks)
        ]
    return x_final, all_x, all_x_mean, all_log_probs
