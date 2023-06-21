# Gender Age Prediction

### Introduction

The aim of this project is to build a multi-task learning model for gender and age prediction from face images (UTKFace dataset from Kaggle). To achieve multi-task learning a hydranet model is built in PyTorch consisting of a pre-trained CNN backbone and two task specific heads. One for age regression and one for gender classification.

### Technologies

- Typical data science tools - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Deep learning frameworks - `PyTorch`, `PyTorch Lightning`.
- Model interpretation tools - `LIME`.
