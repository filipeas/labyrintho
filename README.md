<p align="center">
  <img src="src/assets/logo.png" alt="Labyrintho Logo" width="200" />
</p>

<h1 align="center">Labyrintho</h1>

<p align="center">
  Labyrintho is a UI to make segmentation in seismic facies
</p>
<p align="center">
  <strong>Version 0.0.1</strong>
</p>

## About this project

**Labyrintho** is a user interface (UI) developed for seismic facies segmentation. This tool is a experiment for task of seismic segmentation, and in the future, can be auxiliate geophisics and students in this area.

## How to run
1. Clone this repository:
   ```bash
   git clone https://github.com/filipeas/labyrintho.git

2. Execute ``` ./setup.sh ``` to create a venv and install all dependences.

<!-- 3. Configure labyrintho's env and run ``` python main.py ```. -->

4. We're currently using [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything) as a base for segmentation. This project have the objetive realize segmentation with prompts, so the program have tool for apply prompts. Because of this, we need download and set checkpoints of SAM into **src/models/segment_anything/checkpoints**. Download some of the versions of ViT (in repository of the SAM) and paste into **src/models/segment_anything/checkpoints**. Then, open **main.py** and configure vit_model and checkpoint in method **__init__** in **MainApp()** class.
- For exemple, if you [download vit_b](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints), paste the file into **src/models/segment_anything/checkpoints** and open **main.py**, go to **__init__** in **MainApp()** and change flag **model_type** to vit_b and **checkpoint** to path where have the correct checkpoint.

### Requirements
- numpy
- matplotlib
- PyQt5
- opencv-python
- tifffile
- torch
- torchvision
- [minerva](https://github.com/discovery-unicamp/Minerva)

## Utils
1. For generate images with pattern colors:
    ```bash
    python src/utils/paint_mask.py /path/to/folder/ gt.png no

    python src/utils/paint_mask.py /path/to/folder/ gt_iou.png yes 
