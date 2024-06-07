import json
import os
import sys
import time
import warnings

import numpy as np
import wandb
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
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

import lovely_tensors as lt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.functional.multimodal.clip_score import (
    _clip_score_update,
    _get_clip_model_and_processor,
)
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoTokenizer, T5EncoderModel

from modules.lora import inject_trainable_lora_extended, save_lora_weight
from modules.patch_unet_v2 import register_encoding_adapters
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


# Dataset
class ImageTextDataset(Dataset):
    def __init__(self, anno_path, image_size):
        f = open(anno_path)
        lines = f.readlines()
        f.close()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.data = [line.strip().split("\t") for line in lines]

    def __getitem__(self, index):
        try:
            image = Image.open(self.data[index][0]).convert("RGB")
            image = self.preprocess(image)
            prompt = self.data[index][1]
            return image, prompt
        except:
            # Resample a new one
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def __len__(self):
        return len(self.data)


class CocoRandomCaption(CocoCaptions):
    def __getitem__(self, index):
        img, caps = super().__getitem__(index)
        cap = random.choice(caps)
        return img, cap


@torch.no_grad()
def parti_validation(
    t2i_model_type,
    text_encoder,
    tokenizer,
    unet,
    vae,
    accelerator,
    batch_size,
    step,
    output_dir,
):
    clip_model, clip_processor = _get_clip_model_and_processor(
        "openai/clip-vit-large-patch14"
    )
    clip_model = clip_model.to(accelerator.device)

    pipeline = StableDiffusionPipeline.from_pretrained(
        t2i_model_type,
        unet=unet,
        vae=vae,
        # scheduler=DDIMScheduler.from_pretrained(t2i_model_type, subfolder="scheduler"),
        requires_safety_checker=False,
        safety_checker=None,
    )
    pipeline.text_encoder = text_encoder
    pipeline.tokenizer = tokenizer
    pipeline.tokenizer.model_max_length = 77
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    num_inference_steps = 50
    guidance_scale = 7.5

    parti_prompts_files = {
        "Complex": "/home/sean.man/236004_transformers/project/data/parti_prompts_complex.json",
        "Fine-Grained Details": "/home/sean.man/236004_transformers/project/data/parti_prompts_fine_grained_detail.json",
        "Properties & Positioning": "/home/sean.man/236004_transformers/project/data/parti_prompts_properties_positioning.json",
    }

    for category, parti_prompts_file in parti_prompts_files.items():
        with open(parti_prompts_file) as f:
            parti_prompts = json.load(f)

        generator = torch.Generator(device=accelerator.device).manual_seed(42)

        images = []
        for prompts in chunk(parti_prompts, batch_size):
            images_ = pipeline(
                list(prompts),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
            ).images
            images.extend([to_tensor(img_) for img_ in images_])

        images = torch.stack(images, dim=0).to(accelerator.device)

        scores, _ = _clip_score_update(
            images, parti_prompts, clip_model, clip_processor
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
                                caption=f"{score.item()=}\n{i}: {parti_prompts[i]}",
                            )
                            for i, (image, score) in enumerate(zip(images, scores))
                        ],
                    },
                    step=step,
                )

    del pipeline
    torch.cuda.empty_cache()


@torch.no_grad()
def log_validation(
    t2i_model_type,
    text_encoder,
    tokenizer,
    unet,
    vae,
    accelerator,
    step,
    output_dir,
):
    # create pipeline (note: unet and vae are loaded again in float32)
    # pipeline = DiffusionPipeline.from_pretrained(
    #     t2i_model_type,
    #     text_encoder=text_encoder,
    #     tokenizer=T5Tokenizer.from_pretrained("google-t5/t5-large"),
    #     unet=accelerator.unwrap_model(unet),
    #     vae=vae,
    #     requires_safety_checker=False,
    #     safety_checker=None,
    # )
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    #     pipeline.scheduler.config
    # )
    # pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    prompts = [
        "a beautiful landscape painting with a river flowing through the forest, best quality, extremely detailed, 4k resolution",
        "An astronaut's hand is reaching out from inside a 2D retro computer towards the viewer, hand open, in a flat illustration style with vector art, featuring thick lines, bright colors, and a simple design outlined in black.",
        "Film still of rabbit sitting at the counter of an art-deco loungebar, drinking whisky from a tumbler glass, in the style of 'Blade Runner' (1982), velvety, soft lights, long shot, high quality photo, sharp, look at that detail",
        "hyper realistic portrait of woman , she holding Walkman in hand , wearing an Adiddas jacket, lighting from the upper left corner",
    ]

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    noise_scheduler = DDIMScheduler.from_pretrained(
        t2i_model_type, subfolder="scheduler"
    )
    height = width = 512
    num_inference_steps = 50
    guidance_scale = 7.5

    images = []
    for prompt in prompts:
        # with torch.autocast("cuda"):
        #     image = pipeline(
        #         prompt, num_inference_steps=25, generator=generator
        #     ).images[0]
        # images.append(to_tensor(image))

        # Text embeddings
        text_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
            truncation=True,
        ).input_ids.to(accelerator.device)
        text_embeddings = text_encoder(input_ids=text_ids)[0]
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=77, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[
            0
        ]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Latent preparation
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=accelerator.device,
        )
        latents = latents * noise_scheduler.init_noise_sigma

        # Model prediction
        noise_scheduler.set_timesteps(num_inference_steps)
        for t in noise_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decoding
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = image.clamp(-1, 1)
        images.append(image)

    images = torch.cat(images, dim=0)

    scores = metric(images, prompts)
    scores.detach()

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "CLIP Score": scores.mean().item(),
                    "CLIP Score STD": scores.std().item(),
                    "validation": [
                        wandb.Image(
                            to_pil_image((image + 1) / 2),
                            caption=f"{score=}\n{i}: {prompts[i]}",
                        )
                        for i, (image, score) in enumerate(zip(images, scores))
                    ],
                }
            )

    # del pipeline
    torch.cuda.empty_cache()


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LaVi-Bridge Training")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("-ci", "--coco_image_dir", type=str)
    parser.add_argument("-ca", "--coco_ann_file", type=str)
    parser.add_argument("--anno_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--adapters_ckpt_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_vis", type=float, default=1e-4)
    parser.add_argument("--lr_text", type=float, default=5e-6)
    parser.add_argument("--adapters_design", type=str)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument(
        "--non_strict",
        action="store_false",
        default=True,
        help="non-strict checkpoint loading",
    )
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
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
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
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Blocks to inject LoRA
    VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
    TEXT_ENCODER_REPLACE_MODULES = {"T5Attention"}

    # Modules of T2I diffusion models
    t2i_model_type = args.pretrained_model_name_or_path
    vae = AutoencoderKL.from_pretrained(t2i_model_type, subfolder="vae")
    vis = UNet2DConditionModel.from_pretrained(t2i_model_type, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(
        t2i_model_type, subfolder="scheduler"
    )
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    text_encoder = T5EncoderModel.from_pretrained("t5-large")
    # adapter = TextAdapter(1024, 896, 768)
    # adapter = TextAdapter.from_pretrained(os.path.join(args.ckpt_dir, "adapter"))

    vis.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # LoRA injection
    vis_lora_params, _ = inject_trainable_lora_extended(
        vis,
        r=args.lora_rank,
        target_replace_module=VIS_REPLACE_MODULES,
        loras=os.path.join(args.ckpt_dir, "lora_vis.pt"),
    )
    text_encoder_lora_params, _ = inject_trainable_lora_extended(
        text_encoder,
        r=args.lora_rank,
        target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
        loras=os.path.join(args.ckpt_dir, "lora_text.pt"),
    )

    # groups = [(0, 1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13, 14, 15)]

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

    # we need to load the adapter after the lora injection
    # to avoid the adapter being injected with LoRA
    # register_encoding_adapters(vis, ckpt=os.path.join(args.ckpt_dir, "adapter"))
    register_encoding_adapters(
        vis,
        # ckpt={(_g,): os.path.join(args.ckpt_dir, "adapter") for _g in range(0, 16)},
        # ckpt={_g: os.path.join(args.ckpt_dir, "adapter") for _g in groups},
        # ckpt={tuple(i for i in range(0, 16)): os.path.join(args.ckpt_dir, "adapter")},
        ckpt=groups_ckpt,
        # groups=groups,
        # groups=[(i,) for i in range(0, 16)],  # seperate adapter per layer
        # groups=[tuple(i for i in range(0, 16))],
        groups=groups,
        model_name=t2i_model_type,
    )
    if args.baseline:
        raise NotImplementedError("Baseline is not implemented yet.")
        print("Using baseline LaVi-Bridge (without multiple-adapters).")
        vis.do_rs_single = True

    # Optimizer and scheduler
    optimizer_class = torch.optim.AdamW
    params_to_optimize = [
        {"params": itertools.chain(*vis_lora_params), "lr": args.lr_vis},
        {"params": itertools.chain(*text_encoder_lora_params), "lr": args.lr_text},
        # {"params": adapter.parameters(), "lr": 1e-4},
        # {"params": vis.rs_outer_adapter.parameters(), "lr": 1e-4},
        # {"params": vis.rs_mid_adapter.parameters(), "lr": 1e-4},
        # {"params": vis.rs_inner_adapter.parameters(), "lr": 1e-4},
        # {"params": vis.rs_bn_adapter.parameters(), "lr": 1e-4},
        *[
            {"params": adpt.parameters(), "lr": args.lr_adapter}
            for adpt in vis.adapters.values()
        ],
    ]
    optimizer = optimizer_class(
        params_to_optimize, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8
    )
    lr_scheduler = get_scheduler(
        "constant" if args.warmup_steps == 0 else "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Dataset and dataloader
    # train_dataset = ImageTextDataset(
    #     anno_path=args.anno_path, image_size=args.resolution
    # )
    train_transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = CocoRandomCaption(
        args.coco_image_dir, args.coco_ann_file, transform=train_transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    (
        vis,
        text_encoder,
        # adapter,
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vis,
        text_encoder,
        # adapter,
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    global_step = 0
    last_save = 0
    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "LaVi-Bridge-Plus",
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

    # progress_bar = tqdm(
    #     range(args.max_train_steps), disable=not accelerator.is_local_main_process
    # )
    # progress_bar.set_description("Steps")

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
        text_encoder.train()
        for adpt in vis.adapters.values():
            adpt.train()

        for _, batch in enumerate(train_dataloader):
            # Latent preparation
            latents = vae.encode(batch[0]).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Model prediction
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_input = tokenizer(
                batch[1],
                padding="max_length",
                max_length=77,  #! WARN: this is extremely short
                return_tensors="pt",
                truncation=True,
            ).input_ids.to(accelerator.device)
            encoder_hidden_states_pre = text_encoder(text_input)[0]
            # encoder_hidden_states = adapter(encoder_hidden_states_pre).sample
            encoder_hidden_states = encoder_hidden_states_pre
            model_pred = vis(noisy_latents, timesteps, encoder_hidden_states).sample

            # Optimization
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = itertools.chain(
                    vis.parameters(),
                    text_encoder.parameters(),
                    *[adpt.parameters() for adpt in vis.adapters.values()],
                )
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Validation
            if global_step % args.eval_steps == 0:
                parti_validation(
                    t2i_model_type,
                    text_encoder,
                    tokenizer,
                    vis,
                    vae,
                    accelerator,
                    args.train_batch_size,
                    global_step,
                    args.output_dir,
                )
            
            global_step += 1

            # progress_bar.update(1)

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
                save_lora_weight(
                    accelerator.unwrap_model(vis, **extra_args),
                    f"{args.output_dir}/s{global_step}_lora_vis.pt",
                    target_replace_module=VIS_REPLACE_MODULES,
                )
                save_lora_weight(
                    accelerator.unwrap_model(text_encoder, **extra_args),
                    f"{args.output_dir}/s{global_step}_lora_text.pt",
                    target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
                )
                # accelerator.unwrap_model(adapter, **extra_args).save_pretrained(
                #     f"{args.output_dir}/s{global_step}_adapter"
                # )
                # accelerator.unwrap_model(
                #     vis.rs_outer_adapter, **extra_args
                # ).save_pretrained(f"{args.output_dir}/s{global_step}/outer_adapter")
                # accelerator.unwrap_model(
                #     vis.rs_mid_adapter, **extra_args
                # ).save_pretrained(f"{args.output_dir}/s{global_step}/mid_adapter")
                # accelerator.unwrap_model(
                #     vis.rs_inner_adapter, **extra_args
                # ).save_pretrained(f"{args.output_dir}/s{global_step}/inner_adapter")
                # accelerator.unwrap_model(
                #     vis.rs_bn_adapter, **extra_args
                # ).save_pretrained(f"{args.output_dir}/s{global_step}/bn_adapter")
                d_name = f"{args.output_dir}/s{global_step}"
                for group, adpt in vis.adapters.items():
                    g_name = "_".join(str(g) for g in group)
                    f_name = f"{d_name}/adapter_{g_name}"
                    accelerator.unwrap_model(adpt, **extra_args).save_pretrained(f_name)
                with open(f"{d_name}/groups_config.json", "w") as f:
                    json.dump(list(vis.adapters.keys()), f)
                last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # progress_bar.set_postfix(**logs)
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
