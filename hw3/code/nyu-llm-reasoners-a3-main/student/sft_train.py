

from transformers import AutoModelForCausalLM, AutoTokenizer
from student.evaluate import load_prompt
from datasets import load_dataset, load_from_disk
from sft import tokenize_prompt_and_output, compute_entropy, get_response_log_probs, sft_microbatch_train_step
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    prompt_template = load_prompt("intellect")

    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    
    # loading the data from the dataset right ??
    dataset = load_from_disk(args.intellect_path)


    print("First Examples", dataset[0])

    # create a dataloader

    prompts, gts = [], []
    for ex in dataset: 
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))
    

    # print("prompts", prompts)
    # print("gts", gts)
    tokenized_data = tokenize_prompt_and_output(prompt_strs = prompts, output_strs = gts, tokenizer=tokenizer)

    train_dataset = TensorDataset(
                    tokenized_data["input_ids"],
                    tokenized_data["labels"],
                    tokenized_data["response_mask"]
                )



    dataloader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle = True,
            )

    # test forward pass

    batch = next(iter(dataloader))
    input_ids, labels, response_mask = batch

    print("Input:", input_ids.shape)
    print("Labels:", labels.shape)
    print("Response mask:", response_mask.shape)

    input_ids = input_ids.cuda()
    labels = labels.cuda()
    response_mask = response_mask.cuda()

    
    # create a model that would give us logits 

    model_name = "Qwen/Qwen2.5-Math-1.5B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16
    ).to("cuda")

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 2e-4
    )



    #convert logits to target tokens 

    response_logits = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)

    policy_log_probs = response_logits["log_probs"]

    loss = sft_microbatch_train_step(
            policy_log_probs, 
            response_mask, 
            gradient_accumulation_steps = 1, 
            normalize_constant= 1.0
        )
    
    # update model parameters
    optimizer.step()
    # set model parameters to zero / clear gradients
    optimizer.zero_grad()
    
    

    




    









if __name__ == "__main__" :
    main()




