# HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation (ICCV 2025)

## [<a href="https://lingxiao-li.github.io/hypdae.github.io/" target="_blank">Project Page</a>] 
[![arXiv](https://img.shields.io/badge/arXiv-TF--ICON-green.svg?style=plastic)](https://arxiv.org/abs/2411.17784)


Official implementation of [HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](https://github.com/lingxiao-li/HypDAE).

> **HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation**<br>
> [Lingxiao Li](https://gwang-kim.github.io/), Kaixuan Fan, [Boqing Gong](https://boqinggong.github.io/), [Xiangyu Yue](https://xyue.io/) <br>
> ICCV 2025
>
>**Abstract**: <br>
Few-shot image generation aims to generate diverse and high-quality images for an unseen class given only a few examples in that class. A key challenge in this task is balancing category consistency and image diversity, which often compete with each other. Moreover, existing methods offer limited control over the attributes of newly generated images. In this work, we propose Hyperbolic Diffusion Autoencoders (HypDAE), a novel approach that operates in hyperbolic space to capture hierarchical relationships among images from seen categories. By leveraging pre-trained foundation models, HypDAE generates diverse new images for unseen categories with exceptional quality by varying stochastic subcodes or semantic codes. Most importantly, the hyperbolic representation introduces an additional degree of control over semantic diversity through the adjustment of radii within the hyperbolic disk. Extensive experiments and visualizations demonstrate that HypDAE significantly outperforms prior methods by achieving a better balance between preserving category-relevant features and promoting image diversity with limited data. Furthermore, HypDAE offers a highly controllable and interpretable generation process.

<!-- ## [<a href="https://pnp-diffusion.github.io/" target="_blank">Project Page</a>] [<a href="https://github.com/MichalGeyer/pnp-diffusers" target="_blank">Diffusers Implementation</a>] -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-PnP-b31b1b.svg)](https://arxiv.org/abs/2211.12572) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/PnP-diffusion-features) <a href="https://replicate.com/arielreplicate/plug_and_play_image_translation"><img src="https://replicate.com/arielreplicate/plug_and_play_image_translation/badge"></a> [![TI2I](https://img.shields.io/badge/benchmarks-TI2I-blue)](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0) -->

![teaser](assets/tf-icon.png)

---

</div>

![framework](assets/framework_vector.png)

<!-- # Updates:

**19/06/23** 🧨 Diffusers implementation of Plug-and-Play is available [here](https://github.com/MichalGeyer/pnp-diffusers). -->

<!-- ## TODO:
- [ ] Diffusers support and pipeline integration
- [ ] Gradio demo
- [ ] Release TF-ICON Test Benchmark -->


<!-- ## Usage

**To plug-and-play diffusion features, please follow these steps:**

1. [Setup](#setup)
2. [Feature extraction](#feature-extraction)
3. [Running PnP](#running-pnp)
4. [TI2I Benchmarks](#ti2i-benchmarks) -->

---

</div>

## Contents
  - [Setup](#setup)
    - [Option 1: Using Conda](#option-1-using-conda)
  - [Running TF-ICON](#running-tf\-icon)
    - [Data Preparation](#data-preparation)
    - [Image Composition](#image-composition)
  - [TF-ICON Test Benchmark](#tf\-icon-test-benchmark)
  - [Acknowledgments](#acknowledgments)
  - [Citation](#citation)


<br>

## Setup

Our codebase is built on [TF-ICON](https://github.com/Shilin-LU/TF-ICON) and [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example).

### Using Conda

```bash
# Clone the repository
git clone https://github.com/lingxiao-li/HypDAE.git
cd HypDAE/HypDiffusion

# Create and activate the conda environment
conda env create -f hypdae_env.yaml.yaml
conda activate hypdae
```

### Downloading Stable-Diffusion Weights

Download the pre-trained HypDAE checkpoints from [HuggingFace](https://huggingface.co/lingxiao2049/HypDAE).

## Running HypDAE

### Inference

## Acknowledgments
Our codebase is built on [TF-ICON](https://github.com/Shilin-LU/TF-ICON) and [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example).


## Citation
If you find the repo useful, please consider citing:
```
@INPROCEEDINGS{Li25HypDAE,
title	= {HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation},
author	= {Lingxiao Li and Kaixuan Fan and Boqing Gong and Xiangyu Yue},
year	= {2025},
booktitle	= {International Conference on Computer Vision (ICCV)}
}
```
