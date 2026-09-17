# ⚡ Setup
```
uv pip install omegaconf albumentations==0.5.2 webdataset scikit-image easydict scikit-learn pandas lightning kornia
uv pip install -r third_party/inpaint_anything/lama/requirements.txt
```

# 🧠 Grounded SAM 2
```
cd third_party/grounded_sam_2
uv pip install -e .
uv pip install --no-build-isolation -e grounding_dino
cd ..
```



# 🧪 Debug (optional)
mkdir -p debug_images  # images will be saved here