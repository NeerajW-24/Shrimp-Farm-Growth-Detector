# 🦐 Shrimp Growth Prediction Studio

A Streamlit web app that predicts shrimp weight from pond water-quality parameters using six trained machine-learning models (Random Forest, XGBoost, ANN, SVR, Decision Tree, Linear Regression). Each prediction is compared with the closest real record in the dataset.

## What's in this folder

| File / folder | Purpose |
|---|---|
| `app.py` | The whole application |
| `model/` | The 6 trained models the app loads |
| `shrimp dataset.xlsx` | The dataset (used for value ranges and comparisons) |
| `requirements.txt` | Exact library versions the app was tested with |
| `.python-version` | Python version (3.13) |
| `.streamlit/config.toml` | Theme settings |
| `run_app.bat` / `run_app.sh` | One-click launchers (Windows / Mac-Linux) |
| `Copy_of_shrimp.ipynb`, `modellsss/` | Training notebook and older models (not used by the app) |

> Keep `app.py`, `model/`, and `shrimp dataset.xlsx` together in the same folder, since the app finds them by relative path.

## Run it on your own computer

**Requirement:** Python 3.13 (3.11 or 3.12 should also work). Download it from python.org and tick "Add Python to PATH" during install.

**Easiest way**
- Windows: double-click `run_app.bat`
- Mac/Linux: run `./run_app.sh` in a terminal

The first run takes a few minutes to install libraries (TensorFlow is large). After that it starts in seconds and opens at <http://localhost:8501>.

**Manual way** (Windows PowerShell shown)
```powershell
cd "path\to\this\folder"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Mac/Linux: use `source .venv/bin/activate` instead of the activate line above.

Stop the app with `Ctrl + C` in the terminal.

## Host it online for free (Streamlit Community Cloud)

1. Put the folder on GitHub: create a repository and upload everything except `.venv/` (the included `.gitignore` already skips it).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose the repository, branch `main`, main file `app.py`.
4. Open **Advanced settings** → set **Python version to 3.13** (or the highest available that matches) → **Deploy**.
5. After a few minutes you get a public link like `https://your-app.streamlit.app` that anyone can open.

Notes:
- The free tier has about 1 GB of memory. This app fits, but TensorFlow makes the first start slow.
- Apps that get no visitors for several days go to sleep. Visiting the link and clicking "Wake up" brings it back.
- Other hosts (Hugging Face Spaces with the Streamlit SDK, Render, Railway) also work. Use `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command.

## How to use the app

1. **Choose a model**: leave the default (Random Forest) if unsure.
2. **Select parameters**: keep "Select all" on, or untick it to adjust only some. Unselected ones use typical values. You can also tick "Select an exact row from the dataset" to use a real record.
3. **Enter values**: drag a slider or type a number.
4. **Read the result**: the predicted weight, the closest real record, and the % difference appear immediately. Turn on "Show detailed analytics" to compare all models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on tensorflow | Use Python 3.11 to 3.13, 64-bit. Not 3.14 or 32-bit Python |
| Model "could not be loaded" / version warnings | Install with `requirements.txt` exactly; don't upgrade the libraries (the models were saved with specific versions) |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Page looks plain / fonts differ | Fonts load from Google Fonts, so an internet connection is needed. The app still works without one |
| `Dataset not found` | `shrimp dataset.xlsx` must sit next to `app.py` |
