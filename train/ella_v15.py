import json
import os
import sys
import time
import warnings

import numpy as np
import requests
import wandb
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

sys.path.append("../")
import argparse
import inspect
import itertools
import logging
import math
import random
from io import BytesIO

import lovely_tensors as lt
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import load_from_disk
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load_model
from torch.utils.data import Dataset
from torchmetrics.functional.multimodal.clip_score import (
    _clip_score_update,
    _get_clip_model_and_processor,
)
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from modules.ella_model import ELLA, T5TextEmbedder
from modules.ella_utils import ELLAProxyUNet
from modules.patch_unet_ella import call_for_generate, register_encoding_adapters
from train.log_utils import get_console, get_progress

lt.monkey_patch()
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))


def wandb_include_fn(path):
    for p in ["clip_unet.py", "path_unet.py", "adapters.py", "lora.py"]:
        if p in path:
            return True
    return False


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


def get_lambda_entropy(current_step, warmup_steps, lambda_):
    if current_step < warmup_steps:
        return (current_step / warmup_steps) * lambda_
    else:
        return lambda_


def plot_weights(pipe):
    groups = pipe.unet.unet.adapter_groups

    # Extracting the coefficients
    coefficients = []
    scales = []
    for group in groups:
        scale = pipe.unet.unet.adapters[group].scale
        weights = pipe.unet.unet.adapters[group].weights
        weights = torch.tanh(weights) * scale
        coefficients.append(weights.tolist())
        scales.append(scale)

    coeff_matrix = np.array(coefficients).T  # Transpose to get shape [4, len(groups)]

    # Create a mask to highlight the maximum absolute values
    highlight_mask = np.zeros_like(coeff_matrix, dtype=bool)
    for col in range(coeff_matrix.shape[1]):
        max_idx = np.argmax(np.abs(coeff_matrix[:, col]))
        highlight_mask[max_idx, col] = True

    # Plotting the matrix using Seaborn
    plt.figure(figsize=(16, 4))
    ax = sns.heatmap(
        coeff_matrix,
        vmin=-max(scales),
        vmax=max(scales),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
    )

    # Add border to cells with the maximum absolute value in each column
    for i in range(coeff_matrix.shape[0]):
        for j in range(coeff_matrix.shape[1]):
            if highlight_mask[i, j]:
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2)
                )

    ax.set_title("Coefficients Heatmap")
    ax.set_xlabel("Groups")
    ax.set_ylabel("Coefficients")
    ax.set_xticklabels(groups)
    ax.set_yticklabels(["general", "object", "style", "color"])

    fig = ax.get_figure()

    return fig


class LaionRecaptionedDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.dataset = load_from_disk(root)
        self.transforms = transforms

    def __getitem__(self, index: int):
        entry = self.dataset[index]
        try:
            response = requests.get(entry["url"], timeout=5)
        except requests.RequestException:
            print(f"error loading image from: {entry['url']}")
            new_index = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(new_index)

        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except UnidentifiedImageError:
            print(f"error opening image from: {entry['url']}")
            new_index = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(new_index)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, entry

    def __len__(self) -> int:
        return len(self.dataset)


class OfflineLaionRecaptionedDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.dataset = load_from_disk(root)
        self.transforms = transforms

    def __getitem__(self, index: int):
        entry = self.dataset[index]
        image = entry.pop("image").convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        return image, entry

    def __len__(self) -> int:
        return len(self.dataset)


@torch.no_grad()
def parti_validation(
    t2i_model_type,
    t5_encoder,
    unet,
    ella,
    vae,
    accelerator,
    batch_size,
    step,
):
    clip_model, clip_processor = _get_clip_model_and_processor(
        "openai/clip-vit-large-patch14"
    )
    clip_model = clip_model.to(accelerator.device)

    pipe = StableDiffusionPipeline.from_pretrained(
        t2i_model_type,
        unet=unet,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(accelerator.device)
    setattr(pipe.__class__, "__call__", call_for_generate)
    pipe.unet = ELLAProxyUNet(ella, pipe.unet)
    pipe.set_progress_bar_config(disable=True)

    num_inference_steps = 50
    guidance_scale = 7.5

    parti_prompts_files = {
        "Complex": "/home/sean.man/236004_transformers/project/data/parti_prompts_complex_detailed.json",
        "Fine-Grained Details": "/home/sean.man/236004_transformers/project/data/parti_prompts_fine_grained_detail_detailed.json",
        "Properties & Positioning": "/home/sean.man/236004_transformers/project/data/parti_prompts_properties_positioning_detailed.json",
    }

    weights_fig = plot_weights(pipe)
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"Mixing Coefficients": wandb.Image(weights_fig)}, step=step)

    for category, parti_prompts_file in parti_prompts_files.items():
        with open(parti_prompts_file) as f:
            parti_prompts = json.load(f)
        # print(parti_prompts)
        generator = torch.Generator(device=accelerator.device).manual_seed(42)

        images = []
        for prompts in chunk(parti_prompts, batch_size):
            input_prompts = (
                [prompt["general"] for prompt in prompts]
                + [prompt["object"] for prompt in prompts]
                + [prompt["style"] for prompt in prompts]
                + [prompt["color"] for prompt in prompts]
            )
            prompt_embeds = t5_encoder(input_prompts, max_length=128).to(
                dtype=pipe.dtype
            )
            negative_prompt_embeds = t5_encoder([""] * len(prompts) * 4, max_length=128)
            negative_prompt_embeds = negative_prompt_embeds.to(pipe.device, pipe.dtype)
            images_ = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
            ).images
            images.extend([to_tensor(img_) for img_ in images_])

        images = torch.stack(images, dim=0).to(accelerator.device)
        # print(f" [] = {len([prompt['general'] for prompt in prompts])}")
        scores, _ = _clip_score_update(
            images * 255.0,
            [prompt["general"] for prompt in parti_prompts],
            clip_model,
            clip_processor,
        )
        scores.detach()

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {
                        f"{category} Clip (mean)": scores.mean().item(),
                        f"{category} Clip (std.)": scores.std().item(),
                        f"{category} Images": [
                            wandb.Image(
                                to_pil_image(image),
                                caption=f"score={score.item():.2f}\n{i}: {parti_prompts[i]}",
                            )
                            for i, (image, score) in enumerate(zip(images, scores))
                        ],
                    },
                    step=step,
                )

    del pipe
    torch.cuda.empty_cache()


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LaVi-Bridge Training")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--adapters_ckpt_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--entropy_reg", type=float, default=0)
    parser.add_argument("--entropy_temp", type=float, default=1)
    parser.add_argument("--entropy_warmup", type=int, default=0)
    parser.add_argument("--mix_scale", type=float, default=4.0)
    parser.add_argument("--adapters_design", type=str)
    parser.add_argument(
        "--notes", type=str, default=None, help="the descp to the wandb log"
    )
    parser.add_argument(
        "--group", type=str, default=None, help="the group of wandb log"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        help="wandb mode",
        choices=["disabled", "online", "offline"],
    )
    parser.add_argument(
        "--t2i_model_type",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    seed_everything(args.seed)

    console = get_console()
    progress = get_progress(console)

    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    accelerator = Accelerator(
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            automatic_checkpoint_naming=True,
            total_limit=5,
        ),
    )

    # Modules of T2I diffusion models
    t2i_model_type = args.t2i_model_type
    vae = AutoencoderKL.from_pretrained(t2i_model_type, subfolder="vae")
    vis = UNet2DConditionModel.from_pretrained(t2i_model_type, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(
        t2i_model_type, subfolder="scheduler"
    )
    t5_encoder = T5TextEmbedder()

    # Load ELLA
    ella = ELLA()
    ella_ckpt = "/home/sean.man/236004_transformers/project/checkpoints/ella-sd1.5-tsc-t5xl.safetensors"
    load_model(ella, ella_ckpt, strict=True)

    vis.requires_grad_(False)
    t5_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    ella.requires_grad_(False)

    def parse_groups_ckpt(config):
        group_dict = {}
        for group in config:
            if args.adapters_ckpt_dir is not None:
                dir_name = "adapter_" + "_".join(map(str, group))
                group_dict[tuple(group)] = os.path.join(
                    args.adapters_ckpt_dir, dir_name
                )
            else:
                group_dict[tuple(group)] = os.path.join(args.ckpt_dir, "adapter")
        return group_dict

    groups = json.loads(args.adapters_design)
    groups_ckpt = parse_groups_ckpt(groups)
    groups = list(groups_ckpt.keys())

    print(f"{groups_ckpt=}")

    register_encoding_adapters(
        vis,
        ckpt=None,
        # ckpt=groups_ckpt,
        groups=groups,
        model_name=t2i_model_type,
        mix_scale=args.mix_scale,
    )

    # Optimizer and scheduler
    params_to_optimize = [
        *[
            {"params": adpt.parameters(), "lr": args.lr_adapter}
            for adpt in vis.adapters.values()
        ],
    ]
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8
    )
    lr_scheduler = get_scheduler(
        "constant" if args.warmup_steps == 0 else "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Dataset and dataloader
    train_transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # train_dataset = CocoRandomCaption(
    #     args.coco_image_dir, args.coco_ann_file, transform=train_transform
    # )
    train_dataset = OfflineLaionRecaptionedDataset(
        root="/home/sean.man/236004_transformers/project/data/offline_dataset_0_8",
        transforms=train_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    (
        vis,
        ella,
        t5_encoder,
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vis,
        ella,
        t5_encoder,
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # ELLA was trained using T5 in float16, apperantly this is important in terms of image quality
    # t5_encoder = t5_encoder.to(dtype=torch.float16)

    global_step = 0
    last_save = 0
    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "ELLA-Plus",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "notes": args.notes,
                    "mode": args.wandb_mode,
                    "group": args.group,
                    "save_code": True,
                }
            },
        )
    args.output_dir += f"_{wandb.run.name}"
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    wandb.run.log_code(
        root=os.path.join(os.path.dirname(__file__), ".."),
        include_fn=wandb_include_fn,
        exclude_fn=lambda path, root: "wandb" in os.path.relpath(path, root),
    )
    # Log
    print(f"Num examples = {len(train_dataset)}")
    print(f"Total batch size = {args.train_batch_size * accelerator.num_processes}")
    print(f"Num Epochs = {num_train_epochs}")
    print(f"Total optimization steps = {args.max_train_steps}")

    train_task_id = progress.add_task(
        "Training",
        total=args.max_train_steps,
        epoch=0,
        loss=float("nan"),
    )
    progress.start()

    # Training
    for epoch in range(num_train_epochs):
        vis.train()
        t5_encoder.train()
        for adpt in vis.adapters.values():
            adpt.train()

        for _, (img, entry) in enumerate(train_dataloader):
            # Latent preparation
            latents = vae.encode(img).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Model prediction
            with torch.no_grad():
                input_prompts = (
                    entry["general"] + entry["object"] + entry["style"] + entry["color"]
                )
                encoder_hidden_states = t5_encoder(input_prompts, max_length=128).to(
                    dtype=vis.dtype
                )
                encoder_hidden_states = ella(encoder_hidden_states, timesteps.repeat(4))
                encoder_hidden_states = rearrange(
                    encoder_hidden_states, "(p b) t h -> b p t h", p=4
                )  # [batch, 4, tokens, hidden_size]

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = vis(noisy_latents, timesteps, encoder_hidden_states).sample

            # Optimization
            mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            entropy_loss = torch.tensor(0.0).to(mse_loss.device)
            if args.entropy_reg > 0:
                for group in groups:
                    coeffs = vis.adapters[group].weights
                    probs = F.softmax(coeffs * args.entropy_temp, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    entropy_loss += torch.mean(entropy)
            entropy_reg = get_lambda_entropy(
                global_step,
                args.entropy_warmup,
                args.entropy_reg,
            )
            loss = mse_loss + entropy_reg * entropy_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = itertools.chain(
                    *[adpt.parameters() for adpt in vis.adapters.values()],
                )
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            lr_scheduler.step()
            # print(vis.adapters[(4,5)].weights.grad)
            optimizer.zero_grad()

            # Validation
            if global_step % args.eval_steps == 0:
                parti_validation(
                    t2i_model_type,
                    t5_encoder,
                    vis,
                    ella,
                    vae,
                    accelerator,
                    args.train_batch_size,
                    global_step,
                )

            global_step += 1

            # Saving
            if (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and global_step - last_save >= args.save_steps
            ):
                accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                    inspect.signature(accelerator.unwrap_model).parameters.keys()
                )
                extra_args = (
                    {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                )
                d_name = f"{args.output_dir}/step_{global_step}"
                for group, adpt in vis.adapters.items():
                    g_name = "_".join(str(g) for g in group)
                    f_name = f"{d_name}/adapter_{g_name}"
                    accelerator.unwrap_model(adpt, **extra_args).save_pretrained(f_name)
                with open(f"{d_name}/groups_config.json", "w") as f:
                    json.dump(list(vis.adapters.keys()), f)
                last_save = global_step

            logs = {
                "loss": loss.detach().item(),
                "loss_mse": mse_loss.detach().item(),
                "loss_entropy": entropy_loss.detach().item(),
                "entropy_reg": entropy_reg,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress.update(train_task_id, advance=1, epoch=epoch, loss=loss.item())
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    progress.remove_task(train_task_id)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
