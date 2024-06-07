# Attribute Prompts Mixing

Welcome to the repository for the Attribute Prompts Mixing project! This project was done under the course 236004. It develops from several recent papers aiming to enhance the prompt-sample coherence of generations made with Stable Diffusion and to understand the different roles cross-attention layers have in the U-Net architecture proposed in other works. We approached the problem from the **lavi-bridge** and **ella** perspectives, incorporating insights from the paper "P+".

## Features

- Enhances text alignment in Text-to-Image generation models.
- Leverages a novel decomposition of prompts into sub-prompts (style, color, and composition).
- Incorporates a mixer layer per cross-attention layer to improve prompt adherence and overall aesthetics of generated images.

## Getting Started

# Setting Up the Environment with Micromamba

This project uses a specific Python environment managed by Micromamba. Follow these instructions to recreate the environment on your own system.

## Prerequisites

Ensure you have Micromamba installed on your system. If Micromamba is not installed, you can install it by following the instructions on the [Mamba documentation](https://mamba.readthedocs.io/en/latest/installation.html).

## Create the Environment

1. **Clone the Repository**:
   If you haven't already, clone the repository to your local machine:
   ```bash
   git clone https://github.com/ronraphaeli/Attribute_Prompts_Mixing.git
   cd Attribute_Prompts_Mixing


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
