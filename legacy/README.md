# Original ML assignment (Pima diabetes classifier)

The project this agent was built from, kept unchanged and runnable. It trains a Gaussian Naive Bayes and an MLP classifier on the Pima Indians diabetes data (Glucose, Insulin, BMI, Age), evaluates them, saves EDA plots, and serves predictions through a Flask API with a simple form page.

The two trained pipelines in `models/` are what the agent serves as version **v1** of the Pima model set.

## Run

All paths are relative to this folder, so run everything from here:

```bash
cd legacy
pip install -r requirements.txt

python main.py                 # train + evaluate, writes models/ and plots/
python predict.py --glucose 148 --insulin 0 --bmi 33.6 --age 50 --model mlp
python app.py                  # Flask API on http://localhost:5000 (POST /predict, GET /health)
```

Open `frontend/index.html` for the form page that calls the Flask API.

## Files

| File | Purpose |
|---|---|
| `main.py` | Full pipeline: load → explore → visualize → preprocess → train → evaluate → save |
| `data_loader.py`, `visualization.py` | Loading, EDA and plots (`plots/`) |
| `preprocessing.py` | Zero-value median imputation, IQR outlier capping, interaction features |
| `model_training.py`, `model_evaluation.py` | Naive Bayes + MLP training, metrics, ROC/PR, cross-validation |
| `model_persistence.py` | Pickle save/load and `models/manifest.json` |
| `predict.py` | Command-line prediction |
| `app.py`, `frontend/` | Flask API and form page |

Test-set results (stored in `models/manifest.json`): MLP accuracy 77.9%, F1 0.696, ROC-AUC 0.821 · Naive Bayes accuracy 71.4%, F1 0.593, ROC-AUC 0.776.
