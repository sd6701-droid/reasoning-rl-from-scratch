

from transformers import AutoModelForCausalLM, AutoTokenizer
from student.evaluate import load_prompt
from datasets import load_dataset, load_from_disk
from sft import tokenize_prompt_and_output, compute_entropy, get_response_log_probs, sft_microbatch_train_step
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset

def wandb_init(args):
    wandb.init(
        project=args.wandb_project, 
        name=args.wandb_run_name, 
        config=vars(args)
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")




def train_loop(args):
    torch.manual_seed(args.seed)

    wandb_init(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
                batch_size=args.batch_size,
                shuffle = True,
            )

    # test forward pass

    model_name = args.model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16
    ).to("cuda")

    # create a model that would give us logits 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr
    )

    global_step = 0
    
    model.train()

    for epoch in range(0, args.epochs):
        # at the start of epoch we need to start 
        for batch_idx, batch in enumerate(dataloader): 
            input_ids, labels, response_mask = batch

            print("Input:", input_ids.shape)
            print("Labels:", labels.shape)
            print("Response mask:", response_mask.shape)

            input_ids = input_ids.cuda()
            labels = labels.cuda()
            response_mask = response_mask.cuda()

        
            

            #convert logits to target tokens 

            response_logits = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)

            policy_log_probs = response_logits["log_probs"]

            loss = sft_microbatch_train_step(
                    policy_log_probs, 
                    response_mask, 
                    gradient_accumulation_steps = args.gradient_accumulation_steps, 
                    normalize_constant= args.normalize_constant
                )
            
            should_update = (
                ((batch_idx + 1) % args.gradient_accumulation_steps == 0)
                or (batch_idx + 1) == len(dataloader)
            )
            if should_update:
                # update model parameters
                optimizer.step()
                # set model parameters to zero / clear gradients
                optimizer.zero_grad()

                global_step += 1
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train_step": global_step,
                    "train/epoch": epoch + 1
                })
            
        
    





def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--intellect-path", default="/gpfs/scratch/sd6701/personal/salt/grpo/data-distrib/intellect_math/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wandb_project", type=str, default="sft-intellect")
    parser.add_argument("--wandb_run_name", type=str, default="qwen-math-sft")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--normalize_constant", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./checkpoints") 

    args = parser.parse_args()

    train_loop(args)
    
    




    









if __name__ == "__main__" :
    main()




