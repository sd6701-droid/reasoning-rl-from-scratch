

import torch
from typing import Any, Literal




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




def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages,
        policy_log_probs
    ):

    loss =  -1 * raw_rewards_or_advantages * policy_log_probs

    return loss


def compute_grpo_clip_loss (
        advantages,
        policy_log_probs,
        old_log_probs,    # reference policy
        cliprange
    ) :

    distribution_shift = torch.exp(policy_log_probs - old_log_probs) 
    
    unclip_term = distribution_shift * advantages
    clip_term = torch.clip(distribution_shift, min = 1 - cliprange, max = 1 + cliprange) * advantages
    loss = -1 * torch.minimum(unclip_term, clip_term)

    return loss, {}
                

def compute_policy_gradient_loss (
        policy_log_probs,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
        raw_rewards, 
        advantages,
        old_log_probs,
        clip_range
    ):

    if loss_type == "no_baseline":
        loss, metadata = compute_naive_policy_gradient_loss(
            raw_rewards, 
            policy_log_probs
        )

    elif loss_type == "reinforce_with_baseline":
        loss, metadata = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs
        )
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss (
            advantages,
            policy_log_probs,
            old_log_probs,    # reference policy
            cliprange=clip_range
        )

    return loss, {}


def main(): 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")

    args = parser.parse_args()




if __name__ == "__main__":
    main()