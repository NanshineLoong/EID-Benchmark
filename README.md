# EID-Benchmark

<p align="center">
  <a href="http://arxiv.org/abs/2601.19773">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2601.19773-b31b1b.svg" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
  </a>
</p>

The evaluation framework simulates realistic clinical workflows where a doctor must interact with **patient** and **reporter** simulators to gather evidence before making a diagnosis.

<div align="center">
  <img src="assets/benchmark.png" alt="EID Benchmark Framework" width="800"/>
</div>


## Table of Contents

- [Quickstart](#quickstart)
- [Running Evaluations](#running-evaluations)
- [Datasets](#datasets)
  - [Add your own dataset](#add-your-own-dataset)
- [Citation](#citation)
- [Contact](#contact)

## Quickstart

### 1) Install

```bash
git clone https://github.com/NanshineLoong/EID-Benchmark.git
cd eid-benchmark

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 2) Configure API

```bash
cp .env.example .env
# edit .env with your endpoint:
# OPENAI_API_BASE_URL=...
# OPENAI_API_KEY=...
```

### 3) Run

```bash
eid-eval --datasets medqa --modes roleplay \
  --doctor-model gpt-5-mini \
  --max-turns 8 \
  --max-items 1
```
After running, you should see a result summary and per-case logs saved under `results/`.

## Running Evaluations

```bash
# Full evaluation across datasets and strategies
eid-eval --datasets medqa diagnosisarena rarearena derm \
  --modes cot roleplay react sc refine \
  --doctor-model gpt-5-mini \
  --skip-existing \
  --max-turns 16 \
  --max-items 200 \
  --max-workers 50
```


## Datasets
All available datasets are placed under `data/`.

- **[AgentClinic-MedQA](https://github.com/SamuelSchmidgall/AgentClinic)**: `data/agentclinic_medqa_segmented.jsonl`.
- **[DiagnosisArena](https://huggingface.co/datasets/shzyk/DiagnosisArena)**: `data/DiagnosisArena_segmented.jsonl`.
- **[RareArena](https://github.com/zhao-zy15/RareArena)**: `data/RDC_segmented.jsonl`.
- **[Derm (CRAFT-MD)](https://github.com/rajpurkarlab/craft-md)**: `data/derm_segmented.jsonl`.
- **[ClinicalBench](https://github.com/WeixiangYAN/ClinicalLab)**: we do **not** redistribute a segmented version here，please follow the original repo’s instructions to access the data.

### Add your own dataset

You can generate evidence-based dataset from a `.jsonl` dataset by running the following command:

```bash
eid-segment --dataset path-to-your-jsonl \
  --fields case_vignette \
  --model gpt-5-mini \
  --max-items 200 \
  --workers 10 \
  --out your-output-dataset-file
```


## Citation

```bibtex
@article{long2026strong,
  title={Strong Reasoning Isn't Enough: Evaluating Evidence Elicitation in Interactive Diagnosis},
  author={Long, Zhuohan and Bao, Zhijie and Wei, Zhongyu},
  journal={arXiv preprint arxiv: 2601.19773},
  year={2026}
}
```

## Contact

You can propose issues if any problems on this code hub.
For questions, feel free to reach out via email at [zhlong24@m.fudan.edu.cn](mailto:zhlong24@m.fudan.edu.cn)