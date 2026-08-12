# LLM4MG: Adapting Large Language Model for Multipath Generation via Synesthesia of Machines

<p align="center">
  <strong>Cross-modal channel multipath generation from RGB-D images, LiDAR point clouds, and mmWave radar point clouds</strong>
</p>
<p align="center">
  Ziwei Huang · Shiliang Lu · Lu Bai · Xuesong Cai · Xiang Cheng
</p>
<p align="center">
  <img alt="Paper" src="https://img.shields.io/badge/Paper-2026-blue">
  <img alt="Code" src="https://img.shields.io/badge/Code-Available-brightgreen">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3.0-orange">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8-green">
</p>



##  Overview

**LLM4MG** is a large language model (LLM)-based framework for generating fine-grained wireless channel multipath information from multi-modal sensory observations under the **Synesthesia of Machines (SoM)** paradigm.

![Figure_12](./pngs/Figure_12.jpeg)

Focusing on sixth-generation (**6G**) vehicle-to-infrastructure (**V2I**) communications, LLM4MG learns the complex and nonlinear mapping between the physical world and channel multipath characteristics. It takes synchronized multi-modal sensing data from the transceiver—including RGB-D images, LiDAR point clouds, and (mmWave) radar point clouds—and conducts:

- **LoS / NLoS** status classification;
- **multipath power** generation;
- **multipath delay** generation.

The framework adapts **LLaMA 3.2** to a wireless channel generation task through multi-modal feature alignment, propagation-aware prompt engineering, and low-rank adaptation (**LoRA**).

---



## Downloads

The following resources are required to run inference. Both Google Drive and Baidu Netdisk mirrors are provided.

| Resource | Description | Google Drive | Baidu Netdisk |
| :--- | :--- | :--- | :--- |
| **Testset** | `data/testset` — labeled test data (RGB-D / LiDAR / Radar, 2072 snapshots) and inference_dataset.py | 🔗 [Link](https://drive.google.com/drive/folders/1816h7RSUj0lAjxvilNkCx9S6w3rI5HA1?usp=sharing) | 🔗 [Link](https://pan.baidu.com/s/152yq5nUy5sBUQyIGSSQcKg?pwd=star) |
| **Pretrained weights** | `weights/LLM4MG.pth` — trained model checkpoint | 🔗 [Link](https://drive.google.com/drive/folders/1dLqM1fY0118cAhnNBUXXWocIALExpI_u?usp=sharing) | 🔗 [Link](https://pan.baidu.com/s/11ddgZJ_DHVvtZuPG3CQmBQ?pwd=star) |
| **LLaMA 3.2-1B** | `Llama-3.2-1B/` — Base LLM checkpoint and tokenizer. Automatically downloaded if unavailable locally. If download fails, download it manually from Hugging Face or the link below and update the path in the script. | 🔗 [Link](https://drive.google.com/file/d/1vG5wxgkifz--roicrS6I7l_ymp95Ll4t/view?usp=sharing) | 🔗 [Link]( https://pan.baidu.com/s/1wUfmvOv-KN7SmrJdwCH2JA?pwd=star) |

> After downloading, place the folders according to the [directory structure](#-directory-structure) below.

---



## Directory Structure

```
LLM4MG/
├── data/
│   └── testset/                  # Test dataset (download above)
├── Llama-3.2-1B/                 # LLaMA 3.2-1B checkpoint & tokenizer
├── weights/
│   └── LLM4MG.pth                # Pretrained weights (download above)
├── models/                       # Network definitions (ViT, PointNet2, RadarBEVNet, 
├── data_utils/                   # Dataset & SoM data synthesis
├── dependency/                   # detectron2, mmdetection, rcbevdet-master
├── sample/                       # Example single-sample data for quick testing
├── inference.py                  # Single-sample inference (custom paths / dataset 
├── inference_dataset.py          # Batch inference on the testset(download above)
├── utils.py                      # Data loading & preprocessing helpers
└── requirements.txt
```

---





## Quick Start

**1. Create a conda environment**

```bash
conda create -n LLM4MG python=3.9
conda activate LLM4MG
```

**2. Install PyTorch**

Make sure **CUDA Toolkit 11.8** has been installed.

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

**3. Install dependencies of LLaMA 3.2 and ViT**

```bash
pip install -r requirements.txt
```

**4. Install dependencies of RCBEVDet3D**

```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
mim install mmdet==2.28.2
mim install mmcls==0.25.0
pip install spconv-cu118
```

**5. Compile some operators manually**

```bash
python -m pip install -U setuptools wheel ninja
cd dependency

# detectron2
python -m pip install -v -e ./detectron2 \
    --no-build-isolation \
    --no-deps

# RCBEVDet3D custom ops
cd rcbevdet-master
cd mmdet3d/ops/csrc
python setup.py build_ext --inplace
cd ../deformattn
python setup.py build install
```

---



## Inference

#### Single-sample inference

Infer one sample from **your own muilti-modal  sensor data paths**:

```bash
python inference.py \
    --rgb_bs   sample/bs/RGB/rgb_image_80.png \
    --depth_bs sample/bs/Depth/depth_image_80.png \
    --lidar_bs sample/bs/Lidar/point_cloud_80.ply \
    --radar_bs sample/bs/Radar/RSF_1_radarpoint_snapshot_80.mat \
    --rgb_vh   sample/veh/RGB/front_rgb_image_80.png \
    --depth_vh sample/veh/Depth/front_depth_image_80.png \
    --lidar_vh sample/veh/Lidar/point_cloud_80.ply \
    --radar_vh sample/veh/Radar/Car_0_radarpoint_snapshot_80.mat \
    --dis 49.75 --phi 81.76 --theta -0.81 \
    --pretrained_dir weights/LLM4MG.pth
```

#### Batch inference

Run inference over the whole testset and save results as a `.mat` file:

```bash
python inference_dataset.py
```

---



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{huang2025llm4mg,
  title={LLM4MG: Adapting large language model for multipath generation via synesthesia of machines},
  author={Huang, Ziwei and Lu, Shiliang and Bai, Lu and Cai, Xuesong and Cheng, Xiang},
  journal={npj Wireless Technology},
  year={2026}
}
```

---



## License

This project is licensed under the Apache License, Version 2.0.