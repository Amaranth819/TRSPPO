# python -u train.py --agent ppo \
#   --domain_name walker --task_name walk \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --num_train_steps 3000000 \
#   --work_dir ./logs/walker_walk/ppo/1 \
#   --save_tb --save_video


# python -u train.py --agent trsppo \
#   --domain_name walker --task_name walk \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --tr_policy_coef 0.1 --num_train_steps 3000000 \
#   --work_dir ./logs/walker_walk/trsppo/1 \
#   --save_tb --save_video



# python -u train.py --agent ppo \
#   --domain_name walker --task_name run \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --num_train_steps 3000000 \
#   --work_dir ./logs/walker_run/ppo/1 \
#   --save_tb --save_video


# python -u train.py --agent trsppo \
#   --domain_name walker --task_name run \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --tr_policy_coef 0.1 --num_train_steps 3000000 \
#   --work_dir ./logs/walker_run/trsppo/1 \
#   --save_tb --save_video




# python -u train.py --agent ppo \
#   --domain_name hopper --task_name hop \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 2048 \
#   --actor_lr 1e-4 --critic_lr 1e-4 \
#   --num_train_steps 3000000 \
#   --work_dir ./logs/hopper_hop/ppo/1 \
#   --save_tb --save_video


# python -u train.py --agent trsppo \
#   --domain_name hopper --task_name hop \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --tr_policy_coef 0.1 --num_train_steps 3000000 \
#   --work_dir ./logs/hopper_hop/trsppo/1 \
#   --save_tb --save_video



# python -u train.py --agent ppo \
#   --domain_name cartpole --task_name swingup \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --num_train_steps 500000 \
#   --work_dir ./logs/cartpole_swingup/ppo/1 \
#   --save_tb --save_video



# python -u train.py --agent trsppo \
#   --domain_name cartpole --task_name swingup \
#   --action_repeat 4 --seed 1 \
#   --hidden_dim 64 --batch_size 64 --rollout_steps 1000 \
#   --actor_lr 3e-4 --critic_lr 3e-4 \
#   --tr_policy_coef 0.1 --num_train_steps 500000 \
#   --work_dir ./logs/cartpole_swingup/trsppo/1 \
#   --save_tb --save_video