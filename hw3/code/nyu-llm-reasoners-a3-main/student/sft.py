


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
