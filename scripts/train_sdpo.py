from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import sdpo_pytorch.prompts
import sdpo_pytorch.rewards
from sdpo_pytorch.stat_tracking import PerStepPromptStatTracker
from sdpo_pytorch.diffusers_patch.pipeline_with_logprob_sdpo import pipeline_with_logprob
from sdpo_pytorch.diffusers_patch.ddim_with_logprob_sdpo import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image

# cpu_num = 128
# os.environ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config_sdpo.py:aesthetic", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name, config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not
    # required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # prepare prompt and reward fn
    prompt_fn = getattr(sdpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(sdpo_pytorch.rewards, config.reward_fn)(inference_dtype, accelerator.device)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_step_prompt_stat_tracking:
        stat_tracker = PerStepPromptStatTracker(
            config.per_step_prompt_stat_tracking.buffer_size,
            config.per_step_prompt_stat_tracking.min_count,
        )
        accelerator.register_for_checkpointing(stat_tracker)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Prepare everything with `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    lam = torch.tensor(config.train.lam, device=accelerator.device)
    lam = lam.pow(torch.arange(config.sample.num_steps, dtype=torch.float, device=accelerator.device))
    # lam = lam.pow(torch.arange(config.sample.num_steps - 1, -1, -1, dtype=torch.float, device=accelerator.device))
    lam = lam.repeat(config.sample.batch_size * config.sample.num_batches_per_epoch, 1)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        # When resuming from a checkpoint saved by this pipeline, we first re-run the sampling epochs before continuing
        # training, which is essential for reproducibility.
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        logger.info(f"Resampling from epoch 0 to epoch {first_epoch - 1}")
        global_step = 0
        for epoch in range(first_epoch):
            epoch_info = {"epoch": epoch}

            #################### SAMPLING ####################
            pipeline.unet.eval()
            samples = []
            for i in tqdm(
                range(config.sample.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
                )

                # encode prompts
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                # sample
                with autocast():
                    # generate the first trajectory
                    images1, latents1, log_probs1, _, _, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                    )
                    # generate the second trajectory
                    images2, latents2, log_probs2, _, _, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        latents=latents1[0],
                    )

                latents1 = torch.stack(latents1, dim=1)
                latents2 = torch.stack(latents2, dim=1)
                log_probs1 = torch.stack(log_probs1, dim=1)
                log_probs2 = torch.stack(log_probs2, dim=1)
                timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards = []
                for image1, image2 in zip(images1, images2):
                    reward = executor.submit(reward_fn, torch.cat([image1, image2]), prompts * 2, prompt_metadata)
                    rewards.append(reward)
                # yield to make sure reward computation starts
                time.sleep(0)

                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents1": latents1[:, :-1],  # each entry is the latent before timestep t
                        "next_latents1": latents1[:, 1:],  # each entry is the latent after timestep t
                        "latents2": latents2[:, :-1],  # each entry is the latent before timestep t
                        "next_latents2": latents2[:, 1:],  # each entry is the latent after timestep t
                        "log_probs1": log_probs1,
                        "log_probs2": log_probs2,
                        "rewards": rewards,
                    }
                )

            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                rewards1, rewards2 = [], []
                for rewards in sample["rewards"]:
                    rewards, _ = rewards.result()
                    rewards1.append(torch.as_tensor(rewards[:len(rewards) // 2], device=accelerator.device))
                    rewards2.append(torch.as_tensor(rewards[len(rewards) // 2:], device=accelerator.device))
                sample["rewards1"] = torch.stack(rewards1, dim=1)
                sample["rewards2"] = torch.stack(rewards2, dim=1)

                del sample["rewards"]

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images1[-1]):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                # only log rewards from process 0
                accelerator.log(
                    {
                        "images1": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                            for i, (prompt, reward) in enumerate(zip(prompts, sample["rewards1"][:, -1]))
                        ],
                    },
                    step=global_step,
                )
                for i, image in enumerate(images2[-1]):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                # only log rewards from process 0
                accelerator.log(
                    {
                        "images2": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                            for i, (prompt, reward) in enumerate(zip(prompts, sample["rewards2"][:, -1]))
                        ],
                    },
                    step=global_step,
                )

            del images1, images2

            # gather rewards across processes
            rewards = accelerator.gather(torch.cat([samples["rewards1"], samples["rewards2"]])).cpu().numpy()

            del samples["rewards1"], samples["rewards2"]

            # compute sample-mean rewards
            rwd_name = config.reward_name
            epoch_info[rwd_name + f"_{config.sample.num_steps}s"] = rewards[:, -1]
            epoch_info[rwd_name + f"_{config.sample.num_steps}s_mean"] = rewards[:, -1].mean()
            epoch_info[rwd_name + f"_{config.sample.num_steps}s_std"] = rewards[:, -1].std()
            epoch_info[rwd_name + "_1s"] = rewards[:, 0]
            epoch_info[rwd_name + "_1s_mean"] = rewards[:, 0].mean()
            epoch_info[rwd_name + "_1s_std"] = rewards[:, 0].std()

            # log debugging values for each epoch
            accelerator.log(epoch_info, step=global_step)
            del epoch_info

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
            assert num_timesteps == config.sample.num_steps
            _ = torch.randperm(total_batch_size, device=accelerator.device)
            _ = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])

            global_step += 1

            # load checkpoint for the next epoch
            resume_path = config.resume_from.rsplit('_', 1)[0] + '_' + str(epoch)
            logger.info(f"Resuming from {resume_path}")
            accelerator.load_state(resume_path)

    else:
        first_epoch = 0
        global_step = 0

    for epoch in range(first_epoch, config.num_epochs):
        epoch_info = {"epoch": epoch}

        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                # generate the first trajectory
                images1, latents1, log_probs1, anchor_steps1, sim_first1, sim_anchor1, sim_last1 = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
                # generate the second trajectory
                images2, latents2, log_probs2, anchor_steps2, sim_first2, sim_anchor2, sim_last2 = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    latents=latents1[0],
                )

            latents1 = torch.stack(latents1, dim=1)
            latents2 = torch.stack(latents2, dim=1)
            log_probs1 = torch.stack(log_probs1, dim=1)
            log_probs2 = torch.stack(log_probs2, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = []
            for image1, image2 in zip(images1, images2):
                reward = executor.submit(reward_fn, torch.cat([image1, image2]), prompts * 2, prompt_metadata)
                rewards.append(reward)
            # yield to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents1": latents1[:, :-1],  # each entry is the latent before timestep t
                    "next_latents1": latents1[:, 1:],  # each entry is the latent after timestep t
                    "latents2": latents2[:, :-1],  # each entry is the latent before timestep t
                    "next_latents2": latents2[:, 1:],  # each entry is the latent after timestep t
                    "log_probs1": log_probs1,
                    "log_probs2": log_probs2,
                    "rewards": rewards,
                    "anchor_steps1": anchor_steps1,
                    "anchor_steps2": anchor_steps2,
                    "sim_first1": sim_first1,
                    "sim_first2": sim_first2,
                    "sim_anchor1": sim_anchor1,
                    "sim_anchor2": sim_anchor2,
                    "sim_last1": sim_last1,
                    "sim_last2": sim_last2,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards1, rewards2 = [], []
            for rewards in sample["rewards"]:
                rewards, _ = rewards.result()
                rewards1.append(torch.as_tensor(rewards[:len(rewards) // 2], device=accelerator.device))
                rewards2.append(torch.as_tensor(rewards[len(rewards) // 2:], device=accelerator.device))
            rewards1 = torch.stack(rewards1, dim=1)
            rewards2 = torch.stack(rewards2, dim=1)

            del sample["rewards"]

            # compute dense rewards for the first trajectory
            sample["rewards1"] = (
                rewards1[:, 0:1].repeat(1, config.sample.num_steps) * sample["sim_first1"]
                + rewards1[:, 1:-1].repeat(1, config.sample.num_steps) * sample["sim_anchor1"]
                + rewards1[:, -1:].repeat(1, config.sample.num_steps) * sample["sim_last1"]
            ) / (sample["sim_first1"] + sample["sim_anchor1"] + sample["sim_last1"])
            index = sample["anchor_steps1"]
            sample["rewards1"][:, 0], sample["rewards1"][:, -1] = rewards1[:, 0], rewards1[:, -1]
            sample["rewards1"][torch.arange(len(index), device=accelerator.device), index] = rewards1[:, 1]

            # compute dense rewards for the second trajectory
            sample["rewards2"] = (
                 rewards2[:, 0:1].repeat(1, config.sample.num_steps) * sample["sim_first2"]
                 + rewards2[:, 1:-1].repeat(1, config.sample.num_steps) * sample["sim_anchor2"]
                 + rewards2[:, -1:].repeat(1, config.sample.num_steps) * sample["sim_last2"]
            ) / (sample["sim_first2"] + sample["sim_anchor2"] + sample["sim_last2"])
            index = sample["anchor_steps2"]
            sample["rewards2"][:, 0], sample["rewards2"][:, -1] = rewards2[:, 0], rewards2[:, -1]
            sample["rewards2"][torch.arange(len(index), device=accelerator.device), index] = rewards2[:, 1]

            del sample["anchor_steps1"], sample["sim_first1"], sample["sim_anchor1"], sample["sim_last1"]
            del sample["anchor_steps2"], sample["sim_first2"], sample["sim_anchor2"], sample["sim_last2"]

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images1[-1]):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            # only log rewards from process 0
            accelerator.log(
                {
                    "images1": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, sample["rewards1"][:, -1]))
                    ],
                },
                step=global_step,
            )
            for i, image in enumerate(images2[-1]):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            # only log rewards from process 0
            accelerator.log(
                {
                    "images2": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, sample["rewards2"][:, -1]))
                    ],
                },
                step=global_step,
            )

        del images1, images2

        # gather rewards across processes
        rewards = accelerator.gather(torch.cat([samples["rewards1"], samples["rewards2"]])).cpu().numpy()

        del samples["rewards1"], samples["rewards2"]

        # compute sample-mean rewards
        rwd_name = config.reward_name
        epoch_info[rwd_name + f"_{config.sample.num_steps}s"] = rewards[:, -1]
        epoch_info[rwd_name + f"_{config.sample.num_steps}s_mean"] = rewards[:, -1].mean()
        epoch_info[rwd_name + f"_{config.sample.num_steps}s_std"] = rewards[:, -1].std()
        epoch_info[rwd_name + "_1s"] = rewards[:, 0]
        epoch_info[rwd_name + "_1s_mean"] = rewards[:, 0].mean()
        epoch_info[rwd_name + "_1s_std"] = rewards[:, 0].std()

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps

        if config.sample.gamma > 0:
            # compute discounted returns
            for t in reversed(range(num_timesteps)):
                next_return = rewards[:, t + 1] if t < num_timesteps - 1 else 0.0
                rewards[:, t] += config.sample.gamma * next_return

        if config.per_step_prompt_stat_tracking:
            # per-step-prompt reward normalization
            prompt_ids = torch.cat([samples["prompt_ids"]] * 2)
            prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        del samples["prompt_ids"]

        advantages = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, total_batch_size * 2, -1)[accelerator.process_index]
            .to(accelerator.device)
        )
        samples["advantages1"], samples["advantages2"] = torch.chunk(advantages, chunks=2)

        samples["lam"] = lam

        # log debugging values for each epoch
        accelerator.log(epoch_info, step=global_step)
        del epoch_info

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            if config.train.shuffle_batch:
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}

            if config.train.shuffle_timestep:
                # shuffle along timestep dimension independently for each sample
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
                )
                for key in [
                    "timesteps",
                    "latents1",
                    "next_latents1",
                    "latents2",
                    "next_latents2",
                    "log_probs1",
                    "log_probs2",
                    "advantages1",
                    "advantages2",
                    "lam",
                ]:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for j in tqdm(
                range(num_train_timesteps),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # policy update
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc="Batch",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.train.cfg:
                        # concat negative prompts to sample prompts to avoid two forward passes
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                    else:
                        embeds = sample["prompt_embeds"]

                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred1 = unet(
                                    torch.cat([sample["latents1"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond1, noise_pred_text1 = noise_pred1.chunk(2)
                                noise_pred1 = noise_pred_uncond1 + config.sample.guidance_scale * (
                                    noise_pred_text1 - noise_pred_uncond1
                                )

                                noise_pred2 = unet(
                                    torch.cat([sample["latents2"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond2, noise_pred_text2 = noise_pred2.chunk(2)
                                noise_pred2 = noise_pred_uncond2 + config.sample.guidance_scale * (
                                    noise_pred_text2 - noise_pred_uncond2
                                )
                            else:
                                noise_pred1 = unet(
                                    sample["latents1"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample

                                noise_pred2 = unet(
                                    sample["latents2"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample

                            # compute the log-prob of next_latents given latents under the current model
                            _, _, log_prob1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred1,
                                sample["timesteps"][:, j],
                                sample["latents1"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents1"][:, j],
                            )

                            _, _, log_prob2 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred2,
                                sample["timesteps"][:, j],
                                sample["latents2"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents2"][:, j],
                            )

                        # compute advantage differences
                        advantages1 = torch.clamp(
                            sample["advantages1"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        advantages2 = torch.clamp(
                            sample["advantages2"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        advantage_diff = advantages1 - advantages2

                        # compute log-ratio differences
                        log_ratio1 = log_prob1 - sample["log_probs1"][:, j]
                        log_ratio2 = log_prob2 - sample["log_probs2"][:, j]
                        log_diff = log_ratio1 - log_ratio2

                        # compute log-ratio difference weights
                        log_weights = sample["lam"][:, j] / config.train.log_scale

                        loss_unclipped = torch.square(log_weights * log_diff - advantage_diff)

                        if config.train.clip_range > 0:
                            # compute the clipped objective
                            log_prob1_clipped = torch.clamp(
                                log_prob1,
                                sample["log_probs1"][:, j] - config.train.clip_range,
                                sample["log_probs1"][:, j] + config.train.clip_range
                            )
                            log_prob2_clipped = torch.clamp(
                                log_prob2,
                                sample["log_probs2"][:, j] - config.train.clip_range,
                                sample["log_probs2"][:, j] + config.train.clip_range
                            )
                            log_ratio1_clipped = log_prob1_clipped - sample["log_probs1"][:, j]
                            log_ratio2_clipped = log_prob2_clipped - sample["log_probs2"][:, j]
                            log_diff_clipped = log_ratio1_clipped - log_ratio2_clipped

                            loss_clipped = torch.square(log_weights * log_diff_clipped - advantage_diff)
                            loss = torch.mean(torch.maximum(loss_unclipped, loss_clipped))
                        else:
                            loss = torch.mean(loss_unclipped)

                        # debugging values
                        if config.train.clip_range > 0:
                            info["clipfrac"].append(torch.mean(torch.gt(loss_clipped, loss_unclipped).float()))
                        info["adv_clipfrac"].append(
                            (torch.mean(torch.gt(advantages1, sample["advantages1"][:, j]).float()) +
                             torch.mean(torch.gt(advantages2, sample["advantages2"][:, j]).float())) / 2
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

            # log training-related stuff
            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            info["inner_epoch"] = inner_epoch
            accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
