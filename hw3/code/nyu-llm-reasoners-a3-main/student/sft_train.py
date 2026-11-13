

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

    # the loop is almost ready we need to check the results 
    prompts, gts = [], []
    for ex in dataset: 
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        # this is for reasoning trace
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"),"")
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
    
    optimizer.zero_grad()


    for epoch in range(0, args.epochs):
        # at the start of epoch we need to start 

        group_loss = 0.0

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

            group_start = (
                batch_idx // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps

            group_size = min(
                args.gradient_accumulation_steps,
                len(dataloader) - group_start
            )

            loss, metrics = sft_microbatch_train_step(
                    policy_log_probs, 
                    response_mask, 
                    gradient_accumulation_steps = group_size, 
                    normalize_constant= args.normalize_constant
                )
            
            should_update = (
                ((batch_idx + 1) % args.gradient_accumulation_steps == 0)
                or (batch_idx + 1) == len(dataloader)
            )

            group_loss += loss.item()

            if should_update:
                # update model parameters
                optimizer.step()
                # set model parameters to zero / clear gradients
                optimizer.zero_grad()

                global_step += 1

                wandb.log({
                    "train/loss": group_loss,
                    "train_step": global_step,
                    "train/epoch": epoch + 1
                })

                group_loss = 0.0
            
        
    





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







# remaining steps for evaluation 

# Right now nothing in sft_train.py generates tokens, and that's expected. The real question is how the weights you train end up in whatever does the generating.

# Training doesn't do rollouts. In SFT you use teacher forcing. You feed the full prompt plus the target response in one forward pass, and get_response_log_probs scores how likely the model finds each token that's already there. The model never picks a token itself during training.

# Eval does need rollouts. To get accuracy, the model has to generate a response one token at a time. Each new token is sampled from its own output and appended before the next step. Then the reward function grades the finished text. Your student/evaluate.py already does this. Look at line 21: llm.generate(...) with vLLM, greedy decoding (temperature=0.0) and max_tokens=2048. Then question_only_reward_fn grades each output.

# The gap: two separate copies of the model. The vLLM LLM(...) object in evaluate.py loads weights from a path or model name when it's created (line 80). Your training loop updates a different object, the HF AutoModelForCausalLM, in GPU memory. vLLM has no way to see those updates. So the design question is how the trained weights reach vLLM. At a high level there are three options:

# 1. Offline: save, then evaluate. Save a checkpoint at the end (or every N steps) and point evaluate.py --model at it. This is the simplest. What else besides the model weights does vLLM need in that directory to load it?
# 2. In-loop: sync weights into a running vLLM instance. Create vLLM once, and before each eval copy the current policy's weights into it. Check the handout for this. Assignments like this usually describe the pattern, often with vLLM on a separate GPU. Think about what gpu_memory_utilization=0.85 would leave for training if both shared one GPU.
# 3. HF generate directly on the training model. It needs no syncing but is much slower. You'd also need to think about .eval(), no_grad, and padding direction for batched generation.

# Questions to answer before you pick one:
# - What does the handout ask for: accuracy curves during training (which pushes you toward option 2) or only a final number?
# - How many GPUs do you have?
# - Your eval prompts in evaluate.py (lines 96–99) are built the same way as your training prompts. But for MATH (line 112) it uses prompt_template. Which format did your model learn, and which will it see at eval?
# - If you train on reasoning traces, are 2048 new tokens enough for a full trace plus the final answer? What happens to the reward if generation gets cut off?

# A good first sanity check is option 1 on the untrained base model: save it through your own save path, then run evaluate.py on it. If accuracy matches running evaluate.py on Qwen/Qwen2.5-Math-1.5B directly, your save/load path works before training changes anything.

# Which approach does the handout point you toward, and how many GPUs are you running on?
