


module load gcc/13.2.0
module load cuda/12.6.2

python scripts/inference_service.py \
    --embodiment_tag="new_embodiment" \
    --model_path="CP/GR00T-N1.7-ALOHA-RightArm-Multitask" \
    --action_horizon=16 \
    --bestofN_option=sdn \
    --server
