"""Minimal evaluation script for MATH and Intellect test sets."""

from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    print('path of intellect-dataset', path)
    return path.read_text()


def evaluate(llm, prompts, ground_truths):
    """Run evaluation and return accuracy."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)
    print("outputs", outputs)
    print("ground-truths", ground_truths)
    correct = 0
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        print(text, "----->", ground_truths[i])
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]

    return correct / len(outputs)


def evaluate_math_dataset(llm, prompts, ground_truths):
    """Run evaluation and return accuracy."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)
    print("outputs", outputs)
    print("ground-truths", ground_truths)

    correct = 0
    format_reward_error = 0
    answer_reward_error = 0
    cr = 0
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        print(text, "----->", ground_truths[i])
        reward = question_only_reward_fn(text, ground_truths[i])
        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        reward = reward["reward"]

        if format_reward == 1.0:
            if answer_reward == 1.0:
               cr += 1 
            else:
                answer_reward_error += 1
        else :
            format_reward_error += 1


        correct += reward["reward"]

    print(len(outputs), "correct-answer", correct, "format_reward_error-answer", format_reward_error, "answer_reward_error-answer", answer_reward_error)
    return correct / len(outputs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="/gpfs/scratch/sd6701/personal/salt/grpo/data-distrib/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    print("prompt_template", prompt_template)
    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("llm", llm)

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts)
    print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")

    acc = evaluate_math_dataset(llm, prompts, gts)
    #calcuate this in case of math dataset 
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("Hello --- world")
    main()
