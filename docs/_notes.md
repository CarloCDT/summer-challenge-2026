
Champions: checkpoints/20260910-224257_iter200.pt

checkpoints/20260911-082014_iter100.pt

1. Train the teacher
python3 -u -m training.train_ppo --config training/configs/ppo_28ch_level1Pro.yaml
python3 -u -m training.train_ppo --config training/configs/ppo_28ch_level2Pro.yaml
python3 -u -m training.train_ppo --config training/configs/ppo_28ch_level2ProMax.yaml
python3 -u -m training.train_ppo --config training/configs/ppo_28ch_vs_silver.yaml

2. Train the student

python3 -u -m training.distill --config training/configs/distill.yaml


3. Bake

python3 bake_agent.py checkpoints/20260911-144318-distill_iter50.pt -o submission.py --verify 5


4. Check it before pasting

python3 test_submission.py submission.py --opponent level2ProMax --episodes 50


I have created an empty repo in /home/carlo/claude_env, add the non-lite environment there, a readme of how to sue it and a notebook on how to play it against the bosses we have created. the idea is this will be shared for others to train their models.
Maybe also include the heatmap of the channels in the notebook


python3 bake_debug_agent.py checkpoints/20260911-144318-distill_iter50.pt \
    -o debug_submission.py --quant int4 --calibrate 24 --calibrate-opponent level2Silver

    https://www.codingame.com/ide/challenge/summer-challenge-2026-back-track-king