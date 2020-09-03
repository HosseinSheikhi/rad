import numpy as np
import json
import matplotlib.pyplot as plt


def pars_json(address):
    episode_reward = []
    episode = []
    batch_reward = []
    critic_loss = []
    actor_loss = []
    step = []
    with open(address) as f:
        data = json.load(f)

    for item in data:
        episode_reward.append(item['episode_reward'])
        episode.append(item['episode'])
        batch_reward.append(item['batch_reward'])
        critic_loss.append(item['critic_loss'])
        actor_loss.append(item['actor_loss'])
        step.append(item['step'])

    return step, episode_reward


fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.savefig
# step_original, episode_reward_original = pars_json('log/original/cheetah_run/cutout_color1/train.json')
# step_ch, episode_reward_ch = pars_json('log/prioritized/cheetah_run/cutout_color1/train.json')
# plt.plot(step_original, episode_reward_original, label='original')
# plt.plot(step_ch, episode_reward_ch, label='priority')

# step_original, episode_reward_original = pars_json('log/original/cartpole_swing/cutout_color1/train.json')
# step_original2, episode_reward_original2 = pars_json('log/original/cartpole_swing/cutout_color2/train.json')
# step_p1, episode_reward_p1 = pars_json('log/prioritized/cartpole_swing/final_criteria/cutout_color1/train.json')
# step_p2, episode_reward_p2 = pars_json('log/prioritized/cartpole_swing/final_criteria/cutout_color2/train.json')
# plt.plot(step_original, episode_reward_original, label='original1')
# plt.plot(step_original2, episode_reward_original2, label='original2')
# plt.plot(step_p1, episode_reward_p1, label='prioritized1')
# plt.plot(step_p2, episode_reward_p2, label='prioritized2')

step_original, episode_reward_original = pars_json('log/original/cartpole_swing/crop/train.json')
step_p1, episode_reward_p1 = pars_json('log/prioritized/cartpole_swing/final_criteria/crop2_s22/train.json')
plt.plot(step_original, episode_reward_original, label='original1')
plt.plot(step_p1, episode_reward_p1, label='prioritized1')

plt.title("cartpole swing")
plt.xlabel('step')
plt.ylabel('reward')
plt.legend(loc="upper left")
plt.savefig('/home/hossein/Desktop/plt3.png')
fig.show()
