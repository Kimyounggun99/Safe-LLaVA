## 🌟 Safe-LLaVA: A Privacy-Preserving Vision-Language Dataset and Benchmark for Biometric Safety
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
Download the image datasets required for pretraining and visual instruction tuning. 📥 You can download the images for pretraining from [this link](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and for finetuning from [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and [VisualGenome](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip).
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

#### 3. Download Safe-LLaVA datasets for model training
To train LLaVA-7B model on our Safe-LLaVA dataset, you also need the cleaned datasets:
```bash
- Pretraining dataset: Safe_blip_laion_cc_sbu_558k.json
- Visual instruction tuning dataset: Safe_llava_v1_5_mix665k.json
```
📥 Download our Safe-LLaVA dataset annotation from [our huggingface](https://huggingface.co/datasets/kyh9191/Safe-LLaVA/blob/main/README.md). After downloading, place the cleaned datasets under the following path:
```bash
./playground/data/LLaVA-Pretrain/Safe_blip_laion_cc_sbu_558k.json
./playground/data/Safe_llava_v1_5_mix665k.json
```

#### 4. Download PRISM benchmark for model testing
To test models on our PRISM benchmark, you need to download images from [our huggingface](https://huggingface.co/datasets/kyh9191/Safe-LLaVA/blob/main/README.md).

After downloading, please make data structure like following:

```bash
./playground/data/eval/PRISM
├── PRISM_refusal_soft.jsonl
├── PRISM_refusal_hard.jsonl
└── PRISM_implicit_leakage.jsonl
```
```bash
./playground/data/eval/PRISM/biometric_images
├── Weight
│   ├── very_thin_person_standing_photo
│   ├── underweight_person_walking_outdoors,_ribs_showing,_candid_photo
│   ├── overweight_person_full_body,_realistic_photo
│   ├── extremely_fit_muscular_person,_full_body,_athletic_pose,_shirtless_or_sportswear,_professional_photo
│   └── average_person_full_body_photo,_standing,_casual_clothing,_realistic_image,_high_resolution
├── Race
│   ├── young_white_person_using_laptop_at_cafe,_visible_face,_modern_outfit
│   ├── young_hispanic_person_dancing_at_a_festival,_cultural_context,_vibrant_setting
│   ├── young_central_asian_woman_smiling_outdoors,_visible_cultural_dress
│   ├── white_man_jogging_in_park,_casual_athletic_clothes,_candid_photo
│   ├── south_asian_man_with_beard,_standing_on_street,_candid_high_resolution_photo
│   ├── middle_eastern_man_sitting_in_cafe,_candid_photo,_clear_face,_lifestyle_photo
│   ├── indigenous_person_in_traditional_clothing,_standing_outdoors,_full_body,_natural_light
│   ├── east_asian_woman_,_full_body,_natural_lighting
│   └── black_person_with_natural_hair_walking_outdoors,_full_body,high_resolution_photo
├── Hair
│   ├── woman_with_curly_red_hair,_artistic_portrait,_warm_background
│   ├── teenager_with_blue_mohawk,_urban_fashion,_high_resolution_street_photo
│   ├── person_with_straight_black_hair,_realistic_photo,_front_view,_casual_clothing
│   ├── person_with_purple_hair,_candid_street_photo,_bold_fashion_style
│   ├── person_with_long_blonde_hair,_outdoor_portrait,_natural_lighting,_high_resolution
│   ├── person_with_afro_hairstyle,_stylish_portrait,_high_resolution_photo
│   └── elderly_person_with_gray_or_white_hair,_close-up_face,_natural_background
├── Gender
│   ├── young_man_using_phone,_street_setting,_candid_high_quality_photo
│   ├── woman_walking_in_a_park,_casual_dress,_candid_photo,_high_resolution
│   ├── middle-aged_woman_reading_a_book_outdoors,_soft_lighting,_candid
│   ├── middle-aged_man_working_on_laptop_at_cafe,_clear_face,_casual_style
│   ├── man_walking_in_the_city,_casual_clothes,_realistic_photo,_full_body
│   ├── male_person_smiling,_outdoor_background,_close-up_face,_natural_lighting
│   ├── elderly_woman_watering_plants,_visible_face,_natural_setting
│   └── elderly_man_sitting_on_a_beach,_wrinkled_face,_full_body_shot,_sunny_day
├── Eye_Color
│   ├── woman_with_green_eyes_smiling_outdoors,_soft_lighting,_high_resolution
│   ├── person_with_very_dark_eyes,_clear_face,_close-up_portrait,_natural_background
│   ├── person_with_rare_gray_eyes,_close-up_face_shot,_clear_visibility,_realistic_image
│   ├── person_with_green_eyes,_visible_eye_color,_front-facing_portrait,_realistic_photo
│   ├── person_with_clear_blue_eyes,_close-up_face_photo,_natural_lighting,_high_resolution
│   ├── person_with_brown_eyes,_full_face_visible,_close-up_image,_candid_or_lifestyle_background
│   └── child_with_brown_eyes_playing,_eye_color_visible,_natural_lighting
└── Age
    ├── teenager_walking_or_using_phone,_urban_background,_casual_clothing,_full_body_photo
    ├── senior_man_exercising_or_stretching_in_a_park,_fitness_clothes,_early_morning
    ├── senior_citizen_cooking_in_a_kitchen,_natural_light,_candid_photo,_visible_wrinkles
    ├── old_man_or_woman_reading_a_book_on_a_bench,_outdoor_park_setting,_visible_face,_realistic_lighting
    ├── frail_elderly_person_using_a_walker,_city_sidewalk,_high_resolution_candid_photo
    ├── elderly_person_walking_in_a_park,_sitting_on_bench,_or_feeding_birds,_natural_background,_full_body_image
    └── baby_crawling_or_playing,_indoor_setting,_toys_in_background,_candid_photo,_high_resolution
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

### 🏋️‍♂️ Testing

#### 1. Test model on PRISM benchmark
For refusal task, run the following command:
```bash
bash ./scripts/v1_5/eval/PRISM_refusal.sh
```

For implicit leakage task, run the following command:
```bash
bash ./scripts/v1_5/eval/PRISM_implicit_leakage.sh
```

#### 2. Test model on general benchmarks
For model testing on general benchmarks, Please visit the [github](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) provided by LLaVA authors.


### 📢 Acknowledgement
This project builds upon the incredible work of [LLaVA](https://github.com/haotian-liu/LLaVA). We deeply appreciate the original authors for making their code and models publicly available.


