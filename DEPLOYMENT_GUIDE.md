# Deployment Guide for MNIST XAI Streamlit App

## 🎯 Best Option: Streamlit Cloud (FREE & Easy)

### Why Streamlit Cloud?
- ✅ **FREE** for public apps
- ✅ **Designed for Streamlit** apps
- ✅ **Easy deployment** (5 minutes)
- ✅ **Automatic updates** from GitHub
- ✅ **No server management**
- ✅ **HTTPS by default**

### Prerequisites
1. GitHub account (free)
2. Streamlit Cloud account (free - sign up with GitHub)

### Step-by-Step Deployment

#### Step 1: Prepare Your Repository

1. **Create `.streamlit/config.toml`** (optional but recommended):

```bash
mkdir .streamlit
```

Create file: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 5
enableXsrfProtection = true
enableCORS = false
```

2. **Create `.gitignore`**:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/

# Model files (too large for git)
*.pkl
*.h5
*.pth

# Data
*.tfrecord
tensorflow_datasets/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Results
xai_results/
mnist_xai_results/
*.png
*.jpg
```

3. **Create `packages.txt`** (for system dependencies):

```txt
libgl1-mesa-glx
libglib2.0-0
```

4. **Update `requirements.txt`** to remove local-only dependencies:

```txt
# Core ML
jax>=0.4.20
jaxlib>=0.4.20
flax>=0.7.5
optax>=0.1.7

# Data
tensorflow>=2.13.0
tensorflow-datasets>=4.9.0
numpy>=1.24.0
pandas>=2.2.0

# Visualization
matplotlib>=3.7.0
Pillow>=9.5.0

# Streamlit
streamlit==1.31.0
streamlit-drawable-canvas>=0.9.0

# Utilities
tqdm>=4.65.0
```

#### Step 2: Prepare for GitHub

1. **Initialize Git** (if not already done):

```bash
cd c:\Uni\advanced_machine_learning
git init
```

2. **Add files**:

```bash
git add mnist_cnn_model.py
git add mnist_xai_visualizations.py
git add mnist_shape_analysis.py
git add streamlit_mnist_app.py
git add requirements.txt
git add packages.txt
git add .streamlit/config.toml
git add README*.md
```

3. **Commit**:

```bash
git commit -m "Initial commit: MNIST XAI Streamlit app"
```

4. **Create GitHub repository**:
   - Go to https://github.com/new
   - Name: `mnist-xai-app`
   - Public (for free Streamlit Cloud)
   - Don't initialize with README (you have one)

5. **Push to GitHub**:

```bash
git remote add origin https://github.com/YOUR_USERNAME/mnist-xai-app.git
git branch -M main
git push -u origin main
```

#### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in deployment settings**:
   - **Repository**: `YOUR_USERNAME/mnist-xai-app`
   - **Branch**: `main`
   - **Main file path**: `streamlit_mnist_app.py`
   - **App URL**: Choose a custom URL (e.g., `mnist-xai`)

5. **Click "Deploy"**

6. **Wait 3-5 minutes** for deployment

7. **Your app will be live at**: `https://YOUR_USERNAME-mnist-xai.streamlit.app`

### Important: Model File

**Problem**: The model file (`mnist_model.pkl`) is too large for GitHub (>100MB typically).

**Solutions**:

#### Option A: Train Model on First Run

Add this to `streamlit_mnist_app.py` at the top:

```python
# Add before load_model()
def ensure_model_exists():
    """Train model if it doesn't exist"""
    model_path = 'mnist_model.pkl'

    if not Path(model_path).exists():
        st.info("Model not found. Training now (this takes 5-10 minutes)...")
        st.warning("This only happens once!")

        # Import training function
        from train_mnist import train_model

        # Train with fewer epochs for faster deployment
        with st.spinner("Training model..."):
            state, history, model = train_model(
                num_epochs=10,  # Reduced for faster training
                batch_size=128,
                save_path=model_path
            )

        st.success("Model trained successfully!")
        st.experimental_rerun()
```

#### Option B: Use Git LFS (Git Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"

# Add and commit
git add .gitattributes
git add mnist_model.pkl
git commit -m "Add model with Git LFS"
git push
```

#### Option C: Host Model Separately

Upload model to:
- Google Drive
- Dropbox
- AWS S3
- Hugging Face Hub

Then download in app:

```python
import requests

def download_model():
    model_url = "YOUR_DIRECT_DOWNLOAD_LINK"
    model_path = "mnist_model.pkl"

    if not Path(model_path).exists():
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
```

### Cost: FREE! ✨

- Streamlit Cloud Community: **FREE**
  - 1 app
  - Public repositories
  - Shared resources
  - Perfect for demos!

---

## Option 2: Firebase + Cloud Run (More Complex)

Since Firebase Hosting doesn't support Python apps directly, you'd need:

### Architecture

```
Firebase Hosting (Frontend)
    ↓
Cloud Run (Backend - Streamlit)
    ↓
Your App
```

### Steps

1. **Convert to Docker**:

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY mnist_cnn_model.py .
COPY mnist_xai_visualizations.py .
COPY mnist_shape_analysis.py .
COPY streamlit_mnist_app.py .

# Copy model (or download it)
COPY mnist_model.pkl .

# Expose port
EXPOSE 8080

# Run streamlit
CMD streamlit run streamlit_mnist_app.py --server.port=8080 --server.address=0.0.0.0
```

2. **Deploy to Cloud Run**:

```bash
# Build image
gcloud builds submit --tag gcr.io/nnxdemo/mnist-xai-app

# Deploy to Cloud Run
gcloud run deploy mnist-xai-app \
  --image gcr.io/nnxdemo/mnist-xai-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

3. **Connect Firebase Hosting** (optional):

Create `firebase.json`:

```json
{
  "hosting": {
    "public": "public",
    "rewrites": [{
      "source": "**",
      "run": {
        "serviceId": "mnist-xai-app",
        "region": "us-central1"
      }
    }]
  }
}
```

**Cost**: Cloud Run has free tier, but may incur costs with traffic.

---

## Option 3: Other Cloud Platforms

### Hugging Face Spaces (FREE & Easy)

Similar to Streamlit Cloud:

1. Create account at https://huggingface.co
2. Create new Space
3. Choose Streamlit
4. Upload files
5. Done!

**URL**: `https://huggingface.co/spaces/YOUR_USERNAME/mnist-xai`

### Heroku (Previously Free, Now Paid)

```bash
# Install Heroku CLI
heroku login

# Create app
heroku create mnist-xai-app

# Create Procfile
echo "web: streamlit run streamlit_mnist_app.py" > Procfile

# Deploy
git push heroku main
```

**Cost**: ~$5-7/month for basic dyno

### Render (Has Free Tier)

1. Go to https://render.com
2. New Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run streamlit_mnist_app.py`

**Cost**: Free tier available

---

## 🎯 Recommendation

**For Your Use Case**: Use **Streamlit Cloud**

### Why?

1. ✅ **FREE** (perfect for demo)
2. ✅ **Easiest** (5-minute setup)
3. ✅ **Designed for Streamlit** (no configuration needed)
4. ✅ **Auto-updates** from GitHub
5. ✅ **Share-able URL** for presentations

### Firebase?

Firebase is great for:
- Static websites (HTML/CSS/JS)
- React/Vue/Angular apps
- Mobile apps

But **not ideal** for Python/Streamlit apps.

If you want to use your Firebase project, consider:
- **Option 2** (Cloud Run + Firebase Hosting)
- Host a static landing page on Firebase
- Link to Streamlit Cloud app

---

## Quick Start: Streamlit Cloud

```bash
# 1. Create .streamlit/config.toml (settings)
# 2. Update .gitignore (exclude large files)
# 3. Push to GitHub
# 4. Go to share.streamlit.io
# 5. Deploy!
```

**Your app will be live in 5 minutes!** 🚀

---

## Need Help?

1. **Streamlit Cloud docs**: https://docs.streamlit.io/streamlit-community-cloud
2. **Firebase + Cloud Run**: https://firebase.google.com/docs/hosting/cloud-run
3. **This project's docs**: See README_MAIN.md

---

**Recommended: Start with Streamlit Cloud, then consider Firebase/Cloud Run if you need more control.** ✨
