
# FaceCNN — Kaggle facial age & gender prediction

This repository contains code, trained models, and data layout used for a facial Age and Gender prediction task (Kaggle-style). The project trains convolutional neural networks on the provided face images and produces predictions for age and gender.

Key artifacts in this repo
- `25-t3-nppe1.ipynb` — primary notebook used for experiments and inference.
- `face_dataset/` — dataset directory (contains `train/`, `test/`, `train.csv`, `test.csv`, `sample_submission.csv`).
- `best_face_model.pth`, `best_finetuned_model.pth` — saved PyTorch model checkpoints.

Evaluation metrics
- Gender: F1 (macro) — higher is better.
- Age: score = 1 - min(rmse(age), 30) / 30 (clipped RMSE scaled to [0,1]).
- Final score used: harmonic mean of the gender and age scores.

Quick overview

1. Explore and reproduce experiments using `25-t3-nppe1.ipynb`.
2. Use the pretrained checkpoints for quick inference (see notebook for examples).
3. If you want to retrain, prepare dependencies and run the training cells in the notebook or extract code into a training script.

Getting started (recommended)

1. Create and activate a Python virtual environment (Python 3.8+ recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install common dependencies (adjust versions as needed):

```powershell
pip install --upgrade pip
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow tqdm opencv-python albumentations
```

Notes: this list is a reasonable starting point; the notebook may rely on extra packages (check imports in `25-t3-nppe1.ipynb`).

Reproducing results

- Open `25-t3-nppe1.ipynb` in Jupyter or VS Code and run the cells sequentially. The notebook includes data loading, augmentation, model architecture, training loops, evaluation, and example inference.
- To run inference using a saved checkpoint (example snippet — also in the notebook):

```python
import torch
from PIL import Image
from torchvision import transforms

# load model
# model = ... define model architecture ...
# model.load_state_dict(torch.load('best_finetuned_model.pth'))
# model.eval()

# sample preprocessing
transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# img = Image.open('face_dataset/test/00000.jpg').convert('RGB')
# x = transform(img).unsqueeze(0)
# out = model(x)

```

Dataset layout

- `face_dataset/train/` — training images.
- `face_dataset/test/` — test images for inference.
- `face_dataset/train.csv` — training labels (image name + target columns).
- `face_dataset/test.csv` — test image list.
- `face_dataset/sample_submission.csv` — sample submission format.

Tips and notes

- If you retrain, set a reproducible seed and log hyperparameters.
- Use mixed precision (AMP) and a good scheduler to speed training and stabilize results.
- Monitor both gender F1 and age RMSE during training; tune augmentations and loss weighting accordingly.


