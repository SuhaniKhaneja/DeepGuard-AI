# DeepGuard AI

Deepfakes are increasingly difficult to distinguish from authentic media, creating real risks around misinformation, identity fraud, and erosion of trust in visual content. DeepGuard AI is an image-based deepfake detection system that uses transfer learning to classify uploaded images as real or manipulated, returning a confidence score alongside a visual explanation of the model's decision. The application is deployed as an interactive web tool, making it accessible without any local setup.

## Key Features

- Image-based deepfake detection (REAL / FAKE classification)
- DenseNet121-based inference via transfer learning
- Confidence score and probability breakdown per prediction
- Adjustable confidence threshold
- Grad-CAM explainability — visual heatmap of the image regions driving each prediction
- Interactive Streamlit interface with a live model performance dashboard
- Graceful demo-mode fallback when model weights are unavailable

## Demo

**Live Demo:** https://deepguard-ai-suhani.streamlit.app/
**GitHub Repository:** https://github.com/SuhaniKhaneja/DeepGuard-AI

## Model

A DenseNet121 backbone was fine-tuned via transfer learning for binary classification (real vs. fake). DenseNet121 was selected for its strong feature reuse through dense connectivity, which helps capture the subtle, low-level artifacts characteristic of manipulated images.

**Approach:**
- Transfer learning on top of an ImageNet-pretrained DenseNet121 backbone
- Binary classification with a sigmoid output layer
- Input images resized to 224×224 and normalized to [0, 1]
- Output: a single probability score, thresholded to produce a REAL/FAKE label with an associated confidence value

## Tech Stack

**Frontend**
- Streamlit (UI, layout, interactive components)
- Custom theming via `config.toml`

**Backend**
- Python (inference pipeline, image preprocessing, request handling)

**Machine Learning**
- TensorFlow / Keras (model architecture, training, inference)
- OpenCV (image processing, Grad-CAM heatmap generation)
- Matplotlib (visualization rendering)
- NumPy, Pandas, scikit-learn

**Deployment**
- Streamlit Community Cloud

## Project Structure

```
DeepGuard-AI/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── models/
│   └── densenet121_deepfake.h5
└── assets/
    └── style.css
```

## Installation

```bash
git clone https://github.com/SuhaniKhaneja/DeepGuard-AI.git
cd DeepGuard-AI
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Upload a JPG, PNG, or WEBP image, select a model, and adjust the confidence threshold to view the prediction, confidence breakdown, and Grad-CAM visualization.

If model weight files are not present under `models/`, the app runs in demo mode and returns a placeholder prediction instead of failing.

## Results

| Model | Accuracy | ROC-AUC | Precision | Recall | Parameters | Inference Speed |
|---|---|---|---|---|---|---|
| DenseNet121 | 96.21% | 0.9887 | 96.38% | 96.04% | 7.0M | 38ms |

DenseNet121 achieves strong accuracy and ROC-AUC on the evaluation set, with an inference time of ~38ms per image.

## Future Improvements

- Video-based deepfake detection (frame sampling and temporal consistency checks)
- Batch inference support for processing multiple images at once
- REST API for programmatic access outside the Streamlit UI
- Docker support for containerized deployment
- Model quantization for faster, lighter-weight inference
- Benchmark additional architectures (e.g., EfficientNet, Vision Transformers)

## Screenshots

**Analysis view**
`assets/screenshots/analysis.png`

**Model comparison dashboard**
`assets/screenshots/comparison.png`

## License

This project is licensed under the MIT License. See `LICENSE` for details.
