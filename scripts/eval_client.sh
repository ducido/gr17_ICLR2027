

module load gcc/13.2.0
module load cuda/12.6.2

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"



python scripts/eval/eval_policy_w_sdn.py \
    --dataset_path="/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/gr17_ICLR2027/CP/aloha_placing_kitchen_lerobot" \
    --save_plot_path="/projects/extern/kisski/kisski-spath/dir.project/VLA_Imit/gr17_ICLR2027/debug_images/" \
    --action_horizon=16 \
    --plot