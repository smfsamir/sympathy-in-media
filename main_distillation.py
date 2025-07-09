import ipdb
import loguru
import json
import os
import pathlib
import click

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from packages.prompts.task_1_ner_distill_prompt import TASK_1_PROMPT
from trl import SFTConfig, SFTTrainer

logger = loguru.logger
# message = ["Language modeling is "]
# inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# # optional verifying cuda
# # inputs = {k: v.to('cuda') for k,v in inputs.items()}
# # olmo = olmo.to('cuda')
# response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


@click.command()
def create_distillation_examples_task1():
    prompts = []
    completions = []
    with open("data/distillation_data/distill_examples.jsonl", "w") as f:
        for annotated_example_file in os.listdir("data/evaluation_dataset"):
            annotation_object = json.load(open(os.path.join("data/evaluation_dataset", annotated_example_file)))
            article_paragraphs = annotation_object['article']
            prompt = TASK_1_PROMPT + "\n".join(article_paragraphs) + "\n\n"
            response = json.dumps(annotation_object['task1'])
            prompts.append(prompt)
            completions.append(response)
            f.write(f"{{'prompt': {prompt}, 'completion': '{response}'}}\n")
    dataset = Dataset.from_dict({
        'prompt': prompts,
        'completion': completions
    }) 
    dataset.to_json("data/distillation_data/distill_examples.json") 
    logger.info("Distillation examples created successfully.")
        # Add more examples or prompts as needed
        # f.write("Another example prompt here\n")

@click.command()
def distill_task1_olmo():
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    dataset = load_dataset("json", data_files={'train': "data/distillation_data/distill_examples.json"}, split='train')
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    training_args = SFTConfig()
    trainer = SFTTrainer(
        "allenai/OLMo-1B-hf",
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # optional verifying cuda
    # inputs = {k: v.to('cuda') for k,v in inputs.items()}
    # olmo = olmo.to('cuda')
    # response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)



@click.group()
def main():
    pass

main.add_command(create_distillation_examples_task1)
main.add_command(distill_task1_olmo)
# main.add_command(create_distillation_examples_task1)

if __name__ == "__main__":
    main()