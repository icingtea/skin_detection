# Universal Skin Detection Without Color Information

This project implements the algorithm proposed in *Universal Skin Detection Without Color Information* (WACV 2017), which detects human skin in grayscale images using facial priors and texture-based features, without relying on color.

# 📑 Features
- Custom dataset derived from the HELEN dataset.
- Skin region identification using:
  - Face-based intensity priors.
  - Distance-based facial landmarks.
  - Local Binary Patterns (LBP), lacunarity, and grayscale statistics.
- Region growing algorithm for full skin segmentation.

# ⚙️ Run Locally
1. Clone the repository and install dependencies:

```bash
git clone https://github.com/icingtea/skin_detection.git
cd skin_detection
pip install -r requirements.txt
```
2. Run `main.ipynb`

# 🌐 BibTex
```bibtex
@INPROCEEDINGS{7926593,
   author={Sarkar, Abhijit and Abbott, A. Lynn and Doerzaph, Zachary},
   booktitle={2017 IEEE Winter Conference on Applications of Computer Vision (WACV)}, 
   title={Universal Skin Detection Without Color Information}, 
   year={2017},
   volume={},
   number={},
   pages={20-28},
   keywords={Skin;Image color analysis;Face;Gray-scale;Image segmentation;Videos;Lighting},
   doi={10.1109/WACV.2017.10}
}
```