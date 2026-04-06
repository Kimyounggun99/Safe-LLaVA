## 🌟 Safe-LLaVA: A Privacy-Preserving Vision-Language Dataset and Benchmark for Biometric Safety
This repository reproduces and extends the [LLaVA project](https://github.com/haotian-liu/LLaVA) by systematically removing biometric information (e.g., gender, race, age) from the training data.
We introduce Safe-LLaVA, a privacy-conscious version of LLaVA, and propose a new evaluation protocol called PRISM (Privacy-aware Evaluation of Responses in Sensitive Modalities).

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
Download the image datasets required for pretraining and visual instruction tuning. 📥 You can download the images for pretraining from [this link](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and for finetuning from [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and [VisualGenome](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip).
```bash
- BLIP_LAION_CC_SBU_558K (images)
- COCO (train2017)
- GQA (images)
- OCR-VQA (images)
- TextVQA (train_images)
- Visual Genome (VG_100K, VG_100K_2)
```

### 🏋️‍♂️ Training
You can skip this section if you use our [weights (0.5B / 7B)](https://huggingface.co/datasets/kyh9191/Safe-LLaVA/blob/main/README.md) 

#### 1. Organize dataset directory
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

#### 2. Download Safe-LLaVA datasets for model training
To train LLaVA-7B model on our Safe-LLaVA dataset, you also need the cleaned annotations:
```bash
- Pretraining dataset: Safe_blip_laion_cc_sbu_558k.json
- Visual instruction tuning dataset: Safe_llava_v1_5_mix665k.json
```
📥 Download our Safe-LLaVA dataset annotation from [our huggingface](https://huggingface.co/datasets/kyh9191/Safe-LLaVA/blob/main/README.md). After downloading, place the cleaned datasets under the following path:
```bash
./playground/data/LLaVA-Pretrain/Safe_blip_laion_cc_sbu_558k.json
./playground/data/Safe_llava_v1_5_mix665k.json
```


Once your environment and datasets are ready, you can start training Safe-LLaVA.

#### 3. Pretraining
Run the following command to start the pretraining stage:
```bash
bash ./scripts/v1_5/pretrain.sh
```
#### 4. Visual Instruction Tuning
After pretraining, proceed to the visual instruction tuning stage:
```bash
bash ./scripts/v1_5/finetune.sh
```




### 🏋️‍♂️ Testing on PRISM benchmark


#### 1. Download PRISM benchmark for model testing
To test models on our PRISM benchmark, you need to download prompts and images from [our huggingface](https://huggingface.co/datasets/kyh9191/Safe-LLaVA/blob/main/README.md).

After downloading, please make data structure like following:

```bash
./YourPath/Safe-LLaVA/PRISM_evaluation/tasks
├── small_PRISM_refusal_soft.jsonl
├── small_PRISM_refusal_hard.jsonl
├── small_PRISM_implicit_leakage.jsonl
├── large_PRISM_refusal_soft.jsonl
├── large_PRISM_refusal_hard.jsonl
└── large_PRISM_implicit_leakage.jsonl
```

```bash
./YourPath/Safe-LLaVA/PRISM_evaluation/images
├── small_images
└── large_images
```




#### 2. Generate model responses 

```bash
cd PRISM_evaluation
```

##### We provide responses from all models so you can skip this step if you use ours. Refer the `./PRISM_evaluation/answers` folder.


Generate responses from models and save their responses.

```bash
python main.py --size {small/large} --model {model_name} --task {refusal_soft/refusal_hard/implicit_leakage}
```

Example:

```bash
python main.py --size large --model Safe-LLaVA-7B --task refusal_soft
```

Generate responses from all models by running follwing command:

```bash
bash run_all.sh
```

#### 3. Evaluate refusal tasks

##### Please note that you have to have your `GPT Api Key` or `Gemini Api Key` for automatic evaluation. 

For refusal task evaluation with `GPT`, run following command:
```bash
python GPT_refusal_evaluation.py --size {small/large} --model {model_name} --task {refusal_soft/refusal_hard} --API_Key {Your_GPT_API_Key}
```
Example:

```bash
python GPT_refusal_evaluation.py --size large --model Safe-LLaVA-7B --task refusal_soft --API_Key API_KEY}
```

For refusal task evaluation with `Gemini`, run the following command:
```bash
python Gemini_refusal_evaluation.py --size {small/large} --model {model_name} --task {refusal_soft/refusal_hard} --API_Key {Your_Gemini_API_Key}
```

Example:

```bash
python Gemini_refusal_evaluation.py --size large --model Safe-LLaVA-7B --task refusal_hard --API_Key YOUR_API_KEY
```

#### 4. Evaluate implicit leakage protection task

For implicit leakage protection task evaluation with `GPT`, run the following command:
```bash
python GPT_implicit_leakage_evaluation.py --size {small/large} --model {model_name} --task implicit_leakage --API_Key {Your_GPT_API_Key}
```
Example:

```bash
python GPT_implicit_leakage_evaluation.py --size large --model Safe-LLaVA-7B --API_Key API_KEY
```

For implicit leakage protection task evaluation with `Gemini`, run following command:
```bash
python Gemini_implicit_leakage_evaluation.py --size {small/large} --model {model_name} --task implicit_leakage --API_Key {Your_Gemini_API_Key}
```
Example:

```bash
python Gemini_implicit_leakage_evaluation.py --size large --model Safe-LLaVA-7B --API_Key YOUR_API_KEY
```

Evaluate all models on both tasks by running follwing commands:

```bash
bash GPT_eval_all.sh
bash Gemini_eval_all.sh
```

#### 5. Compute refusal accuracys and implicit leakage protection scores

For implicit leakage protection scores, run the following command:

```bash
python Calculate_implicit_leakage_protection_score.py --evaluator {GPT/Gemini} --model-name {model_name}
```
Example:

```bash
python Calculate_implicit_leakage_protection_score.py --evaluator Gemini --model-name Safe-LLaVA-7B 
```

For refusal accuracys, run the following command:

```bash
python Calculate_refusal_accuracy.py --evaluator {GPT/Gemini} --model-name {model_name} --task-name {refusal_soft/refusal_hard}
```
Example:

```bash
python Calculate_refusal_accuracy.py --evaluator Gemini --model-name Safe-LLaVA-7B --task-name refusal_soft
```


### 🏋️‍♂️ Testing on general benchmarks
For model testing on general benchmarks, Please visit the [github](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) provided by LLaVA authors.


### 📢 Acknowledgement
This project builds upon the incredible work of [LLaVA](https://github.com/haotian-liu/LLaVA). We deeply appreciate the original authors for making their code and models publicly available.


