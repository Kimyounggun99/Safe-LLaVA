## 🌟 Safe-LLaVA: Removing Biometric Information from Vision Language Models
This repository reproduces and extends the [LLaVA project](https://github.com/haotian-liu/LLaVA) by systematically removing biometric information (e.g., gender, race, age) from the training data.
We introduce Safe-LLaVA, a privacy-conscious version of LLaVA, and propose a new evaluation protocol called BIAS (Biometric Information Awareness and Safety Benchmark).

For more details on environment setup and advanced usage, please refer to the original [LLaVA GitHub page](https://github.com/haotian-liu/LLaVA).

### 🚀 Getting Started
#### Clone the repository
```bash
git clone https://github.com/Kimyounggun99/Safe-LLaVA.git
cd Safe-LLaVA
```

Setup environment
Follow the instructions below to set up the environment:
```bash
conda create -n safe-llava python=3.10 -y
conda activate safe-llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### 📂 Data Preparation
#### 1. Download image datasets
Download the image datasets required for visual instruction tuning:
```bash
- BLIP_LAION_CC_SBU_558K (images)
- COCO (train2017)
- GQA (images)
- OCR-VQA (images)
- TextVQA (train_images)
- Visual Genome (VG_100K, VG_100K_2)
```

#### 2. Organize dataset directory
After downloading, organize the datasets into the following directory structure:
```bash
./YourPath/Safe-LLaVA/playground/data
├── LLaVA-Pretrain
│   └── images
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```
📥 You can download the images for pretraining from [this link](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and for finetuning from [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and [VisualGenome](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip).

#### 3. Download cleaned datasets
To train Safe-LLaVA, you also need the cleaned datasets:
```bash
- Pretraining dataset: Safe_blip_laion_cc_sbu_558k.json
- Visual instruction tuning dataset: Safe_llava_v1_5_mix665k.json
```
📥 Download the cleaned datasets from [this link](https://ucf-my.sharepoint.com/my?id=%2Fpersonal%2Fyo171134%5Fucf%5Fedu%2FDocuments%2FSafe%2DLLaVA&login_hint=yo171134%40ucf%2Eedu). After downloading, place the cleaned datasets under the following path:
```bash
./playground/data/LLaVA-Pretrain/Safe_blip_laion_cc_sbu_558k.json
./playground/data/Safe_llava_v1_5_mix665k.json
```
### 🏋️‍♂️ Training
Once your environment and datasets are ready, you can start training Safe-LLaVA.

#### 1. Pretraining
Run the following command to start the pretraining stage:
```bash
bash ./scripts/v1_5/pretrain.sh
```
#### 2. Visual Instruction Tuning
After pretraining, proceed to the visual instruction tuning stage:
```bash
bash ./scripts/v1_5/finetune.sh
```
### 📢 Acknowledgement
This project builds upon the incredible work of [LLaVA](https://github.com/haotian-liu/LLaVA). We deeply appreciate the original authors for making their code and models publicly available.

### TODO: 🏋️‍♂️ Testing
