<p align="center">
  <img src="assets/physical-ai-bench-logo-20250923.png" alt="Physical AI Bench Logo" width="64%">
</p>

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Hugging Face - Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-orange)](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard)
[![Hugging Face - Generation](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Generation-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation)
[![Hugging Face - Conditional Generation](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Conditional%20Generation-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-conditional-generation)
[![Hugging Face - Understanding](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Understanding-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-understanding)
[![arXiv](https://img.shields.io/badge/arXiv-2512.01989-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.01989)
![Georgia Tech](https://img.shields.io/badge/Affiliation-Georgia%20Tech-ad9e66)
![CMU](https://img.shields.io/badge/Affiliation-CMU-c41230)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SHI-Labs/physical-ai-bench)

## Introduction

Physical AI Bench (PAI-Bench) is a comprehensive benchmark suite for evaluating physical AI generation and understanding. PAI-Bench covers physical scenarios including autonomous vehicle (AV) driving, robotics, industry (smart space) and ego-centric everyday. PAI-Bench contains three subtasks:

- **PAI-Bench-G (Video Generation)**: Evaluates world foundation models' ability to predict future states given current states and control signals
- **PAI-Bench-C (Conditional Video Generation)**: Focuses on world model generation capabilities with more complex control signals such as edges, segmentation masks, depth, etc.
- **PAI-Bench-U (Video Understanding)**: Evaluates understanding of physical scenes.

<p align="center">
  <img src="assets/physical-ai-bench-teaser-20251125.png" alt="Physical AI Bench Overview" width="100%">
</p>

## Datasets

| Tasks           | Data                                                                                                                             | Usage                            |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **PAI-Bench-G** | [🤗 physical-ai-bench-generation](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation)                         | [Link](./generation)             |
| **PAI-Bench-C** | [🤗 physical-ai-bench-conditional-generation](https://huggingface.co/datasets/shi-labs/physical-ai-bench-conditional-generation) | [Link](./conditional_generation) |
| **PAI-Bench-U** | [🤗 physical-ai-bench-understanding](https://huggingface.co/datasets/shi-labs/physical-ai-bench-understanding)                   | [Link](./understanding)          |

## Leaderboard

Leaderboard is available on [🤗 physical-ai-bench-leaderboard](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard).

## Citation

If you use Physical AI Bench in your research, please cite:

```bibtex
@misc{zhou2025paibenchcomprehensivebenchmarkphysical,
      title={PAI-Bench: A Comprehensive Benchmark For Physical AI}, 
      author={Fengzhe Zhou and Jiannan Huang and Jialuo Li and Deva Ramanan and Humphrey Shi},
      year={2025},
      eprint={2512.01989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01989}, 
}
```

## Acknowledgements

We would like to thank NVIDIA Research, especially the Cosmos team for their support which led to the creation of PAI-Bench. We also thank [Yin Cui](https://ycui.me/), [Jinwei Gu](https://www.gujinwei.org/), [Heng Wang](https://hengcv.github.io/), [Prithvijit Chattopadhyay](https://prithv1.xyz/), Andrew Z. Wang, [Imad El Hanafi](https://imadelh.gitlab.io/), and [Ming-Yu Liu](https://mingyuliu.net/) for their valuable feedback and collaboration that helped shaped the project. This research was supported in part by National Science Foundation under Award #2427478 - CAREER Program, and by National Science Foundation and the Institute of Education Sciences, U.S. Department of Education under Award #2229873 - National AI Institute for Exceptional Education. This project was also partially supported by cyberinfrastructure resources and services provided Georgia Institute of Technology.
