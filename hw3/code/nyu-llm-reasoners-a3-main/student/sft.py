


from transformers import AutoModelForCausalLM, AutoTokenizer
from student.evaluate import load_prompt
from datasets import load_dataset, load_from_disk
import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    # we need to tokenise the prompt and output strings 
    #  construct mask  if 1 --> reponse token, 0 --> other tokens

    input_ids_list = []
    label_ids_list = []
    response_mask_list = []
    labels = []
    for prompt, response in zip(prompt_strs, output_strs):
         # tokenize input 
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]


        # tokenize the output/responses
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        print("Prompt IDs:", len(prompt_ids),  prompt_ids)
        print("Output IDs:", len(response_ids), response_ids)

        input_ids = prompt_ids + response_ids

        labels_ids = input_ids[1:]

        response_mask = ([0] * (len(prompt_ids) - 1) + [1] * len(response_ids)) 

        input_ids_list.append(input_ids)
        label_ids_list.append(labels_ids)
        response_mask_list.append(response_mask)

    max_length_input_list = max(len(x) for x in input_ids_list)

    pad_token = tokenizer.pad_token_id
    for i in range(len(input_ids_list)):
        rem_pad = max_length_input_list - len(input_ids_list[i])
        input_ids_list[i] = (input_ids_list[i] + [pad_token] * rem_pad)[:-1]
        label_ids_list[i] = label_ids_list[i] + [pad_token] * rem_pad
        response_mask_list[i] = response_mask_list[i] + [False] * rem_pad

    return { 
        "input_ids" : torch.tensor(input_ids_list, dtype=torch.long), 
        "labels": torch.tensor(label_ids_list, dtype=torch.long), 
        "response_mask": torch.tensor(response_mask_list, dtype=torch.long) 
    } 


def compute_entropy (logits) :

    print("logits", logits)
    ans = []
    for b in range(len(logits)):
        seq_entropy = []
        for s in range(len(logits[b])):
           
           token_logits = logits[b, s, :]
           probs = torch.softmax(token_logits, dim = -1)
           log_probs = torch.log_softmax(token_logits, dim = -1)

           entropy = - (probs * log_probs).sum(dim = -1)
           seq_entropy.append(entropy)
        ans.append(torch.stack(seq_entropy))
    
    return torch.stack(ans)


def get_response_log_probs (model, input_ids, labels, return_token_entropy) :

    
   

    output = model(input_ids).logits

    # log_softmax ovwer V

    log_probs = torch.log_softmax(output, dim = -1)

    labels = labels.unsqueeze(-1)

    # this one of the important part 

# --- Selecting the log-prob of each label token (gather) ---
#
# log_probs has shape (batch, seq_len, vocab_size): for every position,
# a full log-distribution over the vocabulary (last axis = vocab).
#
# Example: log_probs shape (1, 2, 4)
#   position 0: [-2.0, -1.0, -3.0, -0.5]
#   position 1: [-4.0, -2.0, -0.1, -3.0]
#   labels = [[3, 1]]  -> shape (1, 2)
#
# Goal: at each position, pick the entry whose vocab index == label.
#   position 0, label 3 -> -0.5
#   position 1, label 1 -> -2.0
#   expected result: [[-0.5, -2.0]], shape (1, 2) == l
#
# Key ideas:
#   - The gather dimension says WHERE to select (the vocab axis).
#   - The label values say WHICH vocab indices to sele
#   - gather requires index to have the same number of dims as the input,
#     so labels needs an extra trailing size-1 dim bef
#     (values unchanged, only shape (B, S) -> (B, S, 1)).
#   - gather's output has the index's shape (B, S, 1),
#     trailing dim afterward to get (B, S).
#   - index must be an int64 (long) tensor.

    selcted_log_probs = torch.gather(log_probs, dim = -1, index = labels)

    selcted_log_probs = selcted_log_probs.squeeze(-1)

    if return_token_entropy:
        token_entropy = compute_entropy(output)

    
    if return_token_entropy:
        return {
            "log_probs": selcted_log_probs,
            "token_entropy": token_entropy,
        }

    return {
        "log_probs": selcted_log_probs,
    }



def masked_normalize (tensor, mask, normalize_constant, dim=None):


    total_sum = torch.sum(tensor * mask, dim = dim) / normalize_constant


    return total_sum



def sft_microbatch_train_step (policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant):

    masked_pertoken_loss = -1 * policy_log_probs

    # 1) Compute per-token NLL as the negative policy log-probabilities.
    # 2) Mask out prompt and padding tokens so only response tokens contribute.
    # 3) Sum and normalize the masked losses to obtain the scalar SFT loss.
    
    # Reduction choices:
#
# 1) Sum over sequence, then sum over batch (dim=None):
#    - Produces the total NLL of all response tokens.
#    - Loss and gradient grow with batch size and response length.
#    - Changing the microbatch size changes the effective learning rate.
#
# 2) Sum over sequence, then mean over batch:
#    - Produces the average per-example NLL.
#    - It is insensitive to batch size, but longer responses contribute more
#      than shorter responses.
#
# 3) Mean over all response tokens:
#    - Produces the average per-token NLL.
#    - Every response token contributes equally.
#    - Note: masked_normalize divides by a fixed constant rather than the
#      number of masked tokens, so it does not compute this directly.

    loss = masked_normalize(
            masked_pertoken_loss,
            response_mask,
            normalize_constant,
            dim=None,
        )

    loss = loss / gradient_accumulation_steps

    loss.backward()

    print(gradient_accumulation_steps, normalize_constant, loss.item())
    
    return loss, {}









def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--intellect-path", default="/gpfs/scratch/sd6701/personal/salt/grpo/data-distrib/intellect_math/test")

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

    tokenize_prompt_and_output(prompt_strs = prompts, output_strs = gts, tokenizer=tokenizer)




if __name__ == "__main__":
    main()
