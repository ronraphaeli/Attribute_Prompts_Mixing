# Attribute Prompts Mixing

![image](https://github.com/ronraphaeli/Attribute_Prompts_Mixing/assets/102682845/094ec0a1-d812-49e6-91af-2f9fdd61f564)


Welcome to the repository for the Attribute Prompts Mixing project! This project was done under the course 236004. It develops from several recent papers aiming to enhance the prompt-sample coherence of generations made with Stable Diffusion and to understand the different roles cross-attention layers have in the U-Net architecture proposed in other works. We approached the problem from the [**LaVi-Bridge**](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) and [**ELLA**](https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file) perspectives, incorporating insights from the paper ["P+: Extended Textual Conditioning in Text-to-Image Generation"](https://prompt-plus.github.io/).

![mixer_processing](https://github.com/ronraphaeli/Attribute_Prompts_Mixing/assets/102682845/e6c827b8-52d0-4e68-bd29-97cabb519b88)


## Features

- Enhances text alignment in Text-to-Image generation models.
- Leverages a novel decomposition of prompts into sub-prompts (style, color, and composition).
- Incorporates a mixer layer per cross-attention layer to improve prompt adherence and overall aesthetics of generated images.


## methods

### lavi-bridge+
in this method, we train different adapters to different cross attention layers. 
you can train it yourself when running the file "t5_unet.py".
```bash

python t5_unet.py --ckpt_dir (cheackpoint of lavi-bridge) --output_dir (output dir of the checkpoints) --train_batch_size (batch size) -ci (path to coco dataset root) -ca (path to coco annotations) --save_steps 2500 --eval_steps 500 --max_train_steps 100000 --warmup_steps 1000  --lr_adapter 1e-5 --lr_vis 5e-6 --adapters_design (the division of cross attention layers to adapters, for example "[[0, 1, 2, 3] , [4, 5] , [6, 7], [8, 9], [10, 11], [12, 13, 14, 15]]") 
```
### lavi-bridge+
in this method, we train a mixer to every cross attention layers. 
you can train it yourself when running the file "t5_unet.py".

```bash
python ella_v15.py --ckpt_dir ../checkpoints/ --output_dir ../results/ --train_batch_size 10 --num_workers 10 --save_steps 1000 --eval_steps 500 --max_train_steps 100000 --warmup_steps 1000 --lr_adapter 1e-3 --adapters_design [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]] --entropy_reg (entropy regularization, we recommand 0.000025) --entropy_temp ( temprature of the entropy regularization, for example 5.0) --mix_scale (maximum scale of the prompts weights 1.2) --entropy_warmup (warmup for the regularizing term, 5000)
```

## Getting Started

# Setting Up the Environment with Micromamba

This project uses a specific Python environment managed by Micromamba. Follow these instructions to recreate the environment on your own system.

## Prerequisites

Ensure you have Micromamba installed on your system. If Micromamba is not installed, you can install it by following the instructions on the [Mamba documentation](https://mamba.readthedocs.io/en/latest/installation.html).

## Create the Environment

**Clone the Repository**:
   If you haven't already, clone the repository to your local machine:
   ```bash
   git clone https://github.com/ronraphaeli/Attribute_Prompts_Mixing.git
   cd Attribute_Prompts_Mixing
```

### Prerequisites

we use python 3.10.13
Use the requirements.yaml file to create a new environment with Micromamba. Run the following command in the terminal:
```bash
   micromamba create -f requirements.yaml
```
after the installation, run 

```bash
   micromamba activate trans
```


## Citation
If you find our work useful, please consider citing:

```scss
  @article{raphaeli2024attribute,
  title={Improving Text Alignment in Diffusion Models using Attribute Prompts Mixing},
  author={Raphaeli, Ron and Man, Sean},
  year={2024}
}
```



