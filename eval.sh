# Transfer Cube task
policy_class="TAMAC"
task_name="sim_storage_cube_scripted"
log_name="3.16-full"


python3 imitate_episodes_our.py \
--task_name $task_name \
--ckpt_dir "ckpt/$policy_class/$task_name/$log_name" \
--policy_class $policy_class --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 --eval | tee "log/$policy_class/$task_name/$log_name-eval.log"

python3 imitate_episodes_our.py \
--task_name $task_name \
--ckpt_dir "ckpt/$policy_class/$task_name/$log_name" \
--policy_class $policy_class --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 --temporal_agg --eval | tee "log/$policy_class/$task_name/$log_name-eval-temporal.log"
