# task : sim_transfer_cube_scripted  sim_insertion_scripted  sim_stack_cube_scripted  sim_storage_cube_scripted
policy_class="TAMAC"
task_name="sim_transfer_cube_scripted"
log_name="sim_transfer_cube_scripted"

python3 imitate_episodes_our.py \
    --task_name $task_name \
    --ckpt_dir "ckpt/$policy_class/$task_name/$log_name" \
    --policy_class $policy_class --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 1e-5 \
    --seed 0 | tee "log/$policy_class/$task_name/$log_name.log"

