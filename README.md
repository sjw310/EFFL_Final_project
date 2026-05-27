# EFFL Final Project

### Extension of LoQT (Track 2)
This repository is based on the official implementation of **LoQT**.

**Original paper:** LoQT: Low-Rank Adapters for Quantized Training  
https://arxiv.org/abs/2405.16528

**Original repository:** https://github.com/sebulo/LoQT

**Modifications:**
- Applied different dataset : KOREAN-WEBTEXT (HAERAE-HUB), mC4 (Japanese subset)
- Applied different model : GPT-2 (124m)
- Applied different hardware : Nvidia Geforce RTX 5070 (12GB VRAM)

This repository was created for academic final project purposes.

## Setup
To install the dependencies, run the following command:

```bash
conda env create -f environment.yml
```

This will create a conda environment with all the necessary packages. Make sure to activate the environment:

```bash
conda activate extension_loqt
```

## Usage Examples

### Pre-training
For example, to run a script for pre-training a 124m GPT-2 with LoQT model, use the following command:

```bash
bash scripts/pretraining/124m_loqt_gpt2.sh 
```
```text
scripts/
├── pretraining/
│   ├── 60m_llama2_loqt.sh
│   ├── 130m_llama2_loqt.sh
│   ├── 124m_loqt_gpt2.sh
|   ...
```
