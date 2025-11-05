# 🧬CryoFSL: a few-shot learning framework for robust protein particle picking in cryo-EM micrographs

**Paper**: [“CryoFSL: a few-shot learning framework for robust protein particle picking in cryo-EM micrographs”](https://www.biorxiv.org/content/10.1101/2025.09.19.677446v1)  

CryoFSL is a **few-shot learning framework** designed for **automated protein particle picking in Cryo-Electron Microscopy (Cryo-EM) micrographs**.  
Unlike traditional approaches that require large-scale training, CryoFSL achieves high accuracy with as few as **5 labeled micrographs per protein** by leveraging **SAM2 (Segment Anything Model v2) with adapter-based fine-tuning**.

CryoFSL is specifically designed for the practical settings where current particle picking methods struggle:
1. **Novel or low-resource projects** where only a handful of annotated micrographs are available (e.g., early screening of a new target or small labs without extensive annotation resources)
2. **Low signal-to-noise and heterogeneous datasets** where particle contrast varies strongly across micrographs and templates or fully supervised models fail to generalize.
3. **Workflows that prioritize particle quality over quantity**, such as downstream projects requiring high-quality reconstruction from fewer and cleaner particles.
4. **Computationally constrained environments** where full model re-training is impractical.

CryoFSL predicts **protein particle coordinates** from Cryo-EM micrographs and generates outputs in **.STAR** format, compatible with downstream tools such as **RELION** and **CryoSPARC** for 3D reconstruction.


## 🧭 Overview
CryoFSL combines **foundation models** with **few-shot adaptation** for efficient protein particle picking. Figure below demonstrates the particle picking overflow for CryoFSL.

<p align="center">
  <img src="assets/main_diagram.png" alt="CryoFSL architecture"/>
</p>

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/biplabpoudel25/CryoFSL.git
cd CryoFSL
```

### 2. Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate cryofsl
```

### 3. Data organization
You can download datasets from EMPIAR, CryoPPP or use your own dataset and organize them as following:

```bash
CryoFSL/
└── data/
    ├── train/
    │   ├── images/
    │   │   ├── 061.png
    │   │   ├── 062.png
    │   │   └── 063.png
    │   └── labels/
    │       ├── 061.png
    │       ├── 062.png
    │       └── 063.png
    │
    ├── valid/
    │   ├── images/
    │   │   ├── 101.png
    │   │   ├── 102.png
    │   │   └── 103.png
    │   └── labels/
    │       ├── 101.png
    │       ├── 102.png
    │       └── 103.png
    │
    └── test/
        ├── images/
        │   ├── 201.png
        │   ├── 202.png
        │   └── 203.png
        └── labels/
            ├── 201.png
            ├── 202.png
            └── 203.png
```

Micrograph name in **images** and **labels** should be same. Images in each folder contains the micrographs in .png/.jpg/.jpeg format, and the labels contains their corresponding mask with the same filename. Please make sure that the filename of the micrographs in **images** and **labels** folders are consistent.

### 4. Train
To update the path of the training and validation images and labels, go to `configs/cod-sam-vit-l.yaml` and change them. Then, you can use default training (use YAML config):
```bash
python train.py --config configs/cod-sam-vit-l.yaml
```

If you want to update the path to the images and labels, you can pass them as an argument and train with override (e.g., dataset + epochs)
```bash
python train.py \
  --config configs/cod-sam-vit-l.yaml \
  --train_images /path/to/train/images \
  --train_labels /path/to/train/labels \
  --val_images /path/to/val/images \
  --val_labels /path/to/val/labels \
  --epochs 1000 \
  --sam_ckpt /path/to/sam_checkpoint.pt
```

Training will save the trained model and training log in `checkpoints` directory. 

### 5. Testing
Evaluate a trained model on the test dataset: 
```bash
python test.py \
  --config configs/cod-sam-vit-l.yaml \
  --model /path/to/trained/model \
  --output_dir /path/to/save/output/result \
  --test_images /path/to/test/images \
  --test_labels /path/to/test/labels
```

The test code will produce the following results in **outputs** folder: 
1. **micrographs_outputs** : Visual micrographs with selected particles circled in red.
2. **STAR file**: STAR file containing micrograph names and their coordinates. 
3. **all_metrics.txt**: Text document containing metrics like Precision, Recall, F1 and IoU for each test micrographs. 

You can directly use the generated `.star` file in CryoSPARC or RELION for downstream 3D reconstruction.

### 6. 🚀 Steps to Reproduce Results

Follow these steps to reproduce the results presented in the **CryoFSL** paper from **CryoPPP dataset**.

1️⃣ **Download Dataset (Example for EMPIAR-10028)**
```bash
wget https://calla.rnet.missouri.edu/cryoppp_lite/10028.tar.gz
tar -zxvf 10028.tar.gz -C CryoPPP_dataset/
```

After extraction: 
```bash
CryoPPP_dataset/
 └── 10028/
      ├── micrographs/
      └── ground_truth/
          └── particle_coordinates/
```

2️⃣ **Generate Masks and Resized Images**

Run:
```bash
python get_masks.py
```

This script:
* Reads each micrograph and its `.csv` particle coordinate file.
* Generates corresponding **binary masks** marking particle locations.
* Resizes both images and masks to **(1024 × 1024)** resolution.
* Saves them inside:
```bash
CryoPPP_dataset/10028/outputs/
    ├── images/
    └── masks/
```

3️⃣ Prepare Data Splits

Divide the processed dataset into **train, validation, and test** subsets following the configurations shown earlier (in **Data organization**).

4️⃣ Download SAM2 Pretrained Checkpoint

Download the pretrained **SAM2 (hiera-large)** checkpoint from the official repository:  
   👉 [facebookresearch/sam2](https://github.com/facebookresearch/sam2)  
**CryoFSL** uses SAM2 Hiera-Large pretrained weights. Download and place them in ```pretrained``` folder.

5️⃣ Train and test the model

Follow the training steps as explained above. 

**Example of training and testing on 10028 dataset is shown in the following notebook.**

👉 **Train:** [train_notebook.ipynb](tutorial/train_notebook.ipynb)  
👉 **Test:** [test_notebook.ipynb](tutorial/test_notebook.ipynb)

### 7. Outputs
<p align="center">
  <img src="assets/results_section_PAPER.png" alt="CryoFSL picking"/>
</p>
