# task : sim_transfer_cube_scripted  sim_insertion_scripted  sim_stack_cube_scripted  sim_storage_cube_scripted

task="sim_insertion_scripted_rev"
dataset_dir="./dataset/1-$task"
python3 record_sim_episodes_our.py --task_name $task --dataset_dir $dataset_dir --num_episodes 100
python3 visualize_episodes.py --dataset_dir $dataset_dir --episode_idx -1

