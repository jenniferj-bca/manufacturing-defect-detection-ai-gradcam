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

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the Kaggle dataset and arrange it as described in [data/README.md](data/README.md).
The current workspace's `casting_512x512/` layout is supported directly.

## Train and evaluate

From the project root:

```powershell
python src/train.py --model efficientnet
python src/evaluate.py
python src/inference.py path\to\image.jpeg
```

Use `--model custom` for the CNN trained from scratch. Training creates
`models/best_model.keras`; evaluation writes the classification report and
confusion matrix under `reports/`. The target is >99% test accuracy, but the
actual result depends on the dataset variant, split, hardware, and training time;
always use the generated report rather than assuming the target.

## Streamlit demo

```powershell
pip install -r app/requirements_app.txt
streamlit run app/streamlit_app.py
```

Upload an image or select **Webcam / production camera** to capture a casting
part from a camera available to the browser. Each capture is inspected through
the same prediction and Grad-CAM flow; this is single-frame capture, not
continuous video inference. Browser camera permission is required, and deployed
apps must use HTTPS. Use the **Light theme** toggle on the sign-in page or in
the dashboard sidebar to switch between light and dark themes. View the
predicted class, confidence, raw heatmap, and an overlay showing regions that contributed to the decision. Select **Save
inspection to history** to retain a result in the local
`reports/inspection_history.sqlite3` database. Saved inspections remain available
after restarting the app and power the History and Reports views, including CSV
and JSON exports.

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
