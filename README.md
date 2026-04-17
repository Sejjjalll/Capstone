# ⚡ Power Outage Risk Dashboard — Deployment Guide

Your dashboard converted from Google Colab to a permanent Flask web app.

---

## 📁 Files in this package

| File | Purpose |
|------|---------|
| `app.py` | Main Flask server (your dashboard logic + HTML) |
| `requirements.txt` | Python dependencies |
| `Procfile` | Tells Render/Railway how to start the app |
| `dashboard_clean_dataset.parquet` | **YOUR DATA FILE** — add this yourself (see Step 1) |

---

## Step 1 — Add your data file

Copy your cleaned dataset file into this folder:

```
dashboard_clean_dataset.parquet   ← preferred (faster)
   OR
dashboard_clean_dataset.csv       ← also works
```

This file is produced by the first cell of your Colab notebook
and saved to `dashboard_clean_output/dashboard_clean_dataset.parquet`.

Download it from Colab:
```python
from google.colab import files
files.download("dashboard_clean_output/dashboard_clean_dataset.parquet")
```

---

## Step 2 — Upload to GitHub (required for Render)

1. Go to https://github.com and create a **New Repository** (name it e.g. `power-outage-dashboard`)
2. Upload all files from this folder into the repo
3. Also upload `dashboard_clean_dataset.parquet` (must be < 25 MB for GitHub; if larger, see note below)

> **Large file note:** If your parquet file is > 25 MB, use [Git LFS](https://git-lfs.com) or
> host the file on Google Drive / S3 and set the `DATA_PATH` environment variable to its URL.
> For simplicity, most dashboard_clean_dataset.parquet files are well under 25 MB.

---

## Step 3 — Deploy on Render (FREE, always-on)

1. Go to **https://render.com** and sign up / log in (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select your repository
4. Fill in the form:
   - **Name:** `power-outage-dashboard` (or anything you like)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Choose **Free** plan
6. Click **"Create Web Service"**

Render will build and deploy. In ~2 minutes you'll have a permanent URL like:
```
https://power-outage-dashboard.onrender.com
```

That URL works **24/7** — no Colab required!

---

## Alternative: Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Open in your browser
http://localhost:8000
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Data file not found" | Make sure `dashboard_clean_dataset.parquet` is in the same folder as `app.py` |
| Render free tier goes to sleep | Free tier sleeps after 15 min of inactivity; upgrade to Starter ($7/mo) for always-on |
| File too large for GitHub | Use Git LFS, or set `DATA_PATH` env var to a remote URL |
| Port error locally | Set `PORT=8000` environment variable or change in `app.py` |
