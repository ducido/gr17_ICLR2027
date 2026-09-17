mkdir pretrained
cd pretrained/

source .venv/bin/activate

# grounded sam 2
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    wget "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

# sed (optional)
# PLEASE download 'sed_model_large.pth' from the google drive: https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link

# yolo world (optional)
# wget https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth

# inpaint_anything
# PLEASE download 'big-lama' from the google drive: https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing

# install huggingface-cli
uv pip install huggingface_hub

current_dir=$(pwd)
cd ~/.cache/huggingface/hub
huggingface_cache_dir=$(pwd)
cd $current_dir


# grounding-dino
huggingface-cli download "IDEA-Research/grounding-dino-base"
if [ ! -d "grounding-dino-base" ]; then
    ln -s ${huggingface_cache_dir}"/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb" ${current_dir}"/grounding-dino-base"
fi


cd ..