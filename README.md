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

# Download some pretrained weights
Note: Some of the checkpoints cannot be downloaded directly, you may need to download them manually from the links provided in the script.
```
# read this file first before running it
bash scripts/download_pretrained_weights.sh
```



# 🧪 Debug (optional)
mkdir -p debug_images  # images will be saved here