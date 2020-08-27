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
        step.append(item['actor_loss'])

    return episode, episode_reward


episode1, episode_reward1 = pars_json('log/original/cutout_color1/train.json')
episode2, episode_reward2 = pars_json('log/original/cutout_color2/train.json')
episode_p, episode_reward_p = pars_json('log/prioritized/cutout_color1/train.json')
episode_crop, episode_reward_crop = pars_json('log/original/crop/train.json')
fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(episode1, episode_reward1)
plt.plot(episode2, episode_reward2)
plt.plot(episode_p, episode_reward_p)
plt.plot(episode_crop, episode_reward_crop)


fig.show()
