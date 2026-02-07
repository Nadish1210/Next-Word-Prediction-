# 🚀 Streamlit Cloud Deployment Guide

## 📋 Files Needed for Streamlit Cloud

### ✅ **Essential Files**

1. **app.py** - Main Streamlit application (standalone version without FastAPI backend)
2. **requirements.txt** - Python dependencies
3. **next_word_model.h5** - Trained model file
4. **tokenizer.pkl** - Tokenizer file

### 📂 **GitHub Repository Structure**

```
your-repo-name/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── next_word_model.h5        # Model (generated after training)
├── tokenizer.pkl             # Tokenizer (generated after training)
└── README.md                 # Project description
```

---

## 🔧 **Step-by-Step Deployment**

### **Step 1: Train Your Model Locally**

Before deploying, you need to generate the model files:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py
```

This creates:
- `next_word_model.h5` (your trained model)
- `tokenizer.pkl` (your tokenizer)

### **Step 2: Prepare Your GitHub Repository**

1. **Create a new GitHub repository**
   - Go to https://github.com/new
   - Name it (e.g., "next-word-prediction")
   - Make it Public
   - Click "Create repository"

2. **Upload files to GitHub**

   **Option A: Using GitHub Web Interface**
   - Click "uploading an existing file"
   - Drag and drop these files:
     - `app.py`
     - `requirements.txt` (rename from requirements_streamlit.txt)
     - `next_word_model.h5`
     - `tokenizer.pkl`
     - `README.md`
   - Click "Commit changes"

   **Option B: Using Git Command Line**
   ```bash
   # Initialize git
   git init
   git add app.py requirements.txt next_word_model.h5 tokenizer.pkl README.md
   git commit -m "Initial commit"
   
   # Connect to GitHub
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### **Step 3: Deploy on Streamlit Cloud**

1. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Click "Sign in with GitHub"
   - Authorize Streamlit

2. **Create New App**
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit will install dependencies
   - Load your model
   - Start the app
   - Takes 2-5 minutes

4. **Your App is Live! 🎉**
   - You'll get a URL like: `https://your-app-name.streamlit.app`
   - Share this link with anyone!

---

## ⚠️ **Important Notes**

### **File Size Limits**

GitHub has file size limits:
- **Max 100 MB per file**
- If `next_word_model.h5` is larger, use Git LFS (Large File Storage)

**Using Git LFS:**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.pkl"

# Add and commit
git add .gitattributes
git add next_word_model.h5 tokenizer.pkl
git commit -m "Add model files"
git push
```

### **requirements.txt for Streamlit Cloud**

Use this minimal `requirements.txt`:

```
streamlit==1.28.0
tensorflow==2.15.0
numpy==1.24.3
scikit-learn==1.3.2
```

**Don't include:**
- FastAPI (not needed in standalone version)
- uvicorn (not needed)
- requests (not needed for standalone)

---

## 📝 **Differences: Full System vs Streamlit Cloud**

### **Original System (Local Development)**
```
Frontend (Streamlit) ←→ Backend (FastAPI) ←→ Model
     Port 8501              Port 8000
```

### **Streamlit Cloud (Deployment)**
```
App (Streamlit + Model embedded)
```

The `app.py` file combines everything into one application!

---

## 🔍 **Verification Steps**

After deployment, test your app:

1. ✅ App loads without errors
2. ✅ Model status shows "✅ Model Loaded"
3. ✅ Can make predictions
4. ✅ Can generate sentences
5. ✅ Download button works

---

## 🐛 **Common Issues & Solutions**

### Issue 1: "Module not found"
**Solution:** Check `requirements.txt` has all dependencies

### Issue 2: "File not found: next_word_model.h5"
**Solution:** Make sure you uploaded the model files to GitHub

### Issue 3: "Out of memory"
**Solution:** Streamlit Cloud has memory limits. Optimize model:
```python
# In train_model.py, reduce model size:
model = Sequential([
    Embedding(vocab_size, 50, input_length=seq_length),  # Reduced from 100
    LSTM(75),  # Reduced from 150
    Dense(vocab_size, activation='softmax')
])
```

### Issue 4: App is slow
**Solution:** 
- Use `@st.cache_resource` for model loading (already done in app.py)
- Reduce model complexity
- Use smaller vocabulary

---

## 🎯 **Quick Deployment Checklist**

- [ ] Train model locally (`python train_model.py`)
- [ ] Verify model files exist (next_word_model.h5, tokenizer.pkl)
- [ ] Create GitHub repository
- [ ] Upload files:
  - [ ] app.py
  - [ ] requirements.txt (streamlit version)
  - [ ] next_word_model.h5
  - [ ] tokenizer.pkl
  - [ ] README.md
- [ ] Sign in to Streamlit Cloud
- [ ] Deploy app
- [ ] Test all features
- [ ] Share URL!

---

## 📊 **Alternative: Deploy Backend Separately**

If you want to keep the FastAPI backend:

### **Option 1: Deploy Backend on Render/Railway**
1. Deploy `backend.py` on Render.com or Railway.app
2. Get the backend URL (e.g., https://your-api.onrender.com)
3. Update `frontend.py` to use this URL
4. Deploy frontend on Streamlit Cloud

### **Option 2: Use Streamlit Sharing + Heroku**
1. Backend on Heroku (supports FastAPI)
2. Frontend on Streamlit Cloud
3. Connect them via environment variables

---

## 💡 **Pro Tips**

1. **Use Environment Variables** for configuration
   ```python
   import os
   API_URL = os.getenv("API_URL", "http://localhost:8000")
   ```

2. **Add .gitignore** to exclude unnecessary files:
   ```
   __pycache__/
   *.pyc
   venv/
   .env
   ```

3. **Update README.md** with:
   - Demo link
   - Features
   - How to use
   - Technology stack

4. **Monitor Usage**
   - Streamlit Cloud provides usage stats
   - Check memory and CPU usage
   - Optimize if needed

---

## 🌟 **Your Deployment URL**

After deployment, your app will be available at:
```
https://[your-username]-[repo-name]-[random-string].streamlit.app
```

You can customize this in Streamlit Cloud settings!

---

## 📞 **Support Resources**

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Forum: https://discuss.streamlit.io/
- GitHub LFS: https://git-lfs.github.com/

---

## ✅ **Summary**

**For Streamlit Cloud, you need:**
1. `app.py` (standalone version)
2. `requirements.txt` (minimal dependencies)
3. `next_word_model.h5` (your trained model)
4. `tokenizer.pkl` (your tokenizer)

**Upload to GitHub → Deploy on Streamlit Cloud → Done! 🚀**

---

Good luck with your deployment! 🎉
