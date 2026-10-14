# Manufacturing Defect Detection with AI and Grad-CAM

Production-style binary inspection for casting impellers. The system classifies
`def_front` (defective) versus `ok_front` (good), handles class imbalance with
balanced class weights, and explains predictions with Grad-CAM.

## Project structure

```text
data/                 Dataset setup instructions
notebooks/            Guided EDA, training, Grad-CAM, and evaluation notebooks
src/                  Reusable training, evaluation, inference, and XAI code
app/                  Streamlit factory-demo application
models/               Saved checkpoints (generated locally)
reports/              Metrics, plots, and Grad-CAM examples (generated locally)
config.yaml           Reproducible experiment settings
```

## Setup

## Setup

### 1. Create a virtual environment

```powershell
py -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### Dataset

Download the Kaggle casting dataset and arrange it as described in [data/README.md](data/README.md). The project supports the `casting_512x512/` dataset layout.

## Train and evaluate

## Train and Evaluate

Run these commands from the project root after activating your virtual environment.

### Train using EfficientNet

```powershell
python src/train.py --model efficientnet
```

### Evaluate the model

```powershell
python src/evaluate.py
```

### Run inference on an image

```powershell
python src/inference.py path\to\image.jpeg
```

Replace `path\to\image.jpeg` with the path to your test image.

To train the custom CNN from scratch, use:

```powershell
python src/train.py --model custom
```

Training generates `models/best_model.keras`. Evaluation saves the classification report and confusion matrix under `reports/`.

**Note:** Actual accuracy depends on the dataset, train-test split, hardware, and training process. Always report the measured test results rather than assuming a target accuracy.

## Streamlit demo

## Streamlit Demo

Run the web application from the project root.

### 1. Install application dependencies

```powershell
pip install -r app/requirements_app.txt
```

### 2. Launch the application

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL shown in the terminal.

### Features

- **AI-Based Defect Detection:** Classifies casting images as defective (`def_front`) or good (`ok_front`).
- **Image Upload:** Upload a casting image for inspection.
- **Camera Inspection:** Capture a single image using a browser-supported webcam or production camera.
- **Grad-CAM Explainability:** View a heatmap and overlay highlighting image regions that influenced the prediction.
- **Confidence Score:** View the model's prediction confidence.
- **Inspection History:** Save inspection results and review them later.
- **Reports and Exports:** Access inspection records and export reports as CSV or JSON.
- **Light and Dark Themes:** Switch between the available interface themes.

Inspection history is stored locally in `reports/inspection_history.sqlite3`. Camera access requires browser permission; deployed applications require HTTPS for camera access.
## Notebooks

The notebooks are executable walkthroughs for EDA, custom CNN training,
transfer learning, Grad-CAM, and final evaluation. They call the same `src/`
modules used by the CLI and app, so experiments remain reproducible.

## Expected outputs and future work

The official dataset is commonly reported near 99% accuracy with transfer
learning, but performance must be measured on a held-out test set and monitored
for production drift. Future improvements include real-time camera inspection,
multi-class defect taxonomy, calibrated uncertainty, edge deployment, and
operator feedback loops.
