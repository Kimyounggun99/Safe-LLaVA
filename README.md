# 🌟 Safe-LLaVA: Removing Biometric Information from Vision Language Models

This repository reproduces and extends the [LLaVA](https://github.com/haotian-liu/LLaVA) project by systematically **removing biometric information** (e.g., gender, race, age) from the training data.  
We introduce **Safe-LLaVA-OneVision**, a privacy-conscious version of LLaVA, and propose a new evaluation protocol called **BIAS (Biometric Information Awareness and Safety Benchmark)**.

For more details on environment setup and advanced usage, please refer to the original [LLaVA GitHub page](https://github.com/haotian-liu/LLaVA).

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/Kimyounggun99/Safe-LLaVA.git
cd Safe-LLaVA

conda create -n safe-llava python=3.10 -y
conda activate safe-llava

### Setup environment

```
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
