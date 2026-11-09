

from transformers import AutoModelForCausalLM, AutoTokenizer
from student.evaluate import load_prompt
from datasets import load_dataset, load_from_disk
from sft import tokenize_prompt_and_output, compute_entropy, get_response_log_probs
import wandb
import torch

def wandb_init():
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def train_loop(args):
    torch.manual_seed(args.seed)

    wandb_init()




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--intellect-path", default="/gpfs/scratch/sd6701/personal/salt/grpo/data-distrib/intellect_math/test")
    parser.add_argument("--lr", default="0.0002")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    prompt_template = load_prompt("intellect")

    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)

    prompts, gts = [], []
    for ex in dataset: 
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))
    

    print("prompts", prompts)
    print("gts", gts)
    tokenize_prompt_and_output(prompt_strs = prompts, output_strs = gts, tokenizer=tokenizer)











if __name__ == "__main__" :
    main()




