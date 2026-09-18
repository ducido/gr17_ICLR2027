
module load gcc/13.2.0
module load cuda/12.6.2

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# python -c "import torchcodec; print('torchcodec OK')"




python scripts/inference_service.py \
    --client
