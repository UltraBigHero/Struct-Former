# Struct-Former: Structure-Aware Transformer for Emotion Recognition in Lyrics

This repository contains the implementation of Struct-Former, a structure-aware transformer model for emotion recognition in song lyrics. The model leverages both lyrical content and structural information (paragraphs, lines) in a unified hyperbolic space to better capture hierarchical relationships between emotions.

## Overview

Struct-Former extends traditional transformer-based emotion classification by incorporating:

1. **Structural awareness**: Utilizing paragraph and line position information
2. **Hyperbolic embeddings**: Representing hierarchical relationships in non-Euclidean space
3. **Multi-task learning**: Joint supervision at both fine-grained and coarse-grained emotion levels

The model is evaluated on the DALI dataset (a time-aligned lyrics dataset) augmented with emotion annotations.

## Repository Structure

- `extract_all_dali_lyrics.py`: Extracts and processes the DALI dataset into a structured format
- `dali_structure_extractor.py`: Utility for extracting hierarchical paragraph-line-word structure from DALI annotations
- `emotion_annotator_hard_labels.py`: Annotates lyrics with emotion labels using a pre-trained emotion classifier
- `structure_model.py`: Core implementation of the Struct-Former model architecture and training utilities
- `visualization.py`: Visualization tools for model analysis (heatmaps, UMAP projections, norm trends)
- `main.ipynb`: Jupyter notebook containing the main experimental pipeline

## Model Architecture

The Struct-Former model builds upon a pre-trained BERT architecture with several key extensions:

1. **Structure-Aware Layer**: Incorporates paragraph and line position embeddings 
2. **Projection Mechanism**: Maps structural information to the BERT hidden space
3. **Hyperbolic Geometry**: Models hierarchical relationships in Poincaré ball space
4. **Coarse-Fine Supervision**: Multi-task learning with both coarse and fine-grained emotion labels

### Model Variants

The codebase supports multiple model configurations for ablation studies:

- **StructFormer-Hyper**: Full model with structure, projection, hierarchy, and hyperbolic geometry
- **StructFormer-NoStruct**: Removes structural information
- **StructFormer-NoHyper**: Removes hyperbolic geometry (Euclidean space only)
- **StructFormer-NoProj**: Removes structure projection
- **StructFormer-NoCoarse**: Removes coarse-label supervision
- **BERT**: Baseline BERT without structure, hyperbolic geometry, or coarse supervision
- **Multi-task BERT**: BERT with coarse-fine label supervision but no structure
- **HAN**: Hierarchical Attention Network baseline

## Emotion Categories

The model is trained on a hierarchical emotion taxonomy:

### Coarse Labels (7)
- joy, sadness, anger, fear, surprise, love, neutral

### Fine Labels (26)
- joy, amusement, pride, excitement, relief, optimism
- sadness, grief, disappointment, remorse
- anger, annoyance, disapproval
- fear, embarrassment, nervousness
- surprise, realization, confusion
- love, gratitude, desire
- neutral, curiosity, approval, admiration

## Dataset Preparation

1. **Extract DALI dataset**:
```python
python extract_all_dali_lyrics.py
```

2. **Annotate with emotion labels**:
```python
python emotion_annotator_hard_labels.py
```

## Training and Evaluation

Run the main experiment pipeline in the Jupyter notebook:
```
jupyter notebook main.ipynb
```

The notebook contains:
1. Dataset loading and preparation
2. Model definition and training
3. Evaluation metrics
4. Visualization of results

## Evaluation Metrics

- **Macro F1**: Classification performance across all fine-grained emotion categories
- **Weighted F1**: Classification performance weighted by class frequency
- **Hierarchical Macro F1**: Average F1 score computed within each coarse emotion category
- **Silhouette Score**: Measures embedding space quality
- **Sentiment Weighted F1**: Performance on positive, negative, and neutral sentiment categories

## Visualization Tools

The repository includes tools for visualizing:
- Confusion matrices between coarse and fine labels
- UMAP projections of embeddings
- Structure embedding norm trends

## Dependencies

### Required Packages
Install all required dependencies using:
```bash
pip install -r requirements.txt
```

Content of `requirements.txt`:
```
# Core dependencies
torch>=1.9.0
transformers>=4.11.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.62.0

# Structure-aware model requirements
geoopt>=0.5.0  # For hyperbolic geometry operations

# Data processing
umap-learn>=0.5.1  # For visualization

# Optional dependencies
jupyterlab>=3.0.0  # For running notebook experiments
```

### Environment Setup
The code has been tested with Python 3.8+. We recommend using a virtual environment:
```bash
# Create a virtual environment
python -m venv structformer-env

# Activate the environment (Linux/Mac)
source structformer-env/bin/activate
# OR (Windows)
# structformer-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


