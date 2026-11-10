

import torch





def compute_group_normalized_rewards ( reward_fn, 
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std,
    ): 

    rewards = []

    for (response, ground_truth) in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        rewards.append(reward_dict["reward"])


    raw_rewards = torch.tensor(rewards, dtype = torch.float32)

    # we need to group the rewards 
    grouped_raw_rewards = raw_rewards.view(-1, group_size)

    grouped_mean_rewards = torch.mean(grouped_raw_rewards, dim = -1, keepdim = True)

    advantage_rewards = grouped_raw_rewards - grouped_mean_rewards

    # in this case we divide advatage_rewards by std
    if normalize_by_std:
        grouped_std_rewards = torch.std(grouped_raw_rewards, dim = -1, keepdim = True, unbiased=True)
        advantage_rewards = advantage_rewards / (grouped_std_rewards + advantage_eps)
    
    return advantage_rewards.reshape(-1), raw_rewards, {}







    advantage = list(map(lambda x: (x - rewards.mean()) / (rewards.std() + advantage_eps),  rewards)
                     

    return advantage, rewards, {}


def main(): 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")

    args = parser.parse_args()




if __name__ == "__main__":
    main()