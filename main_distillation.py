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

def compute_metrics(eval_preds):
    # This function can be customized to compute specific metrics
    # For now, we will just return a dummy metric
    ipdb.set_trace()
    predictions, labels = eval_preds
    predictions = predictions.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

@click.command()
def distill_task1_olmo():
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    dataset = load_dataset("json", data_files={'train': "data/distillation_data/distill_examples.json"}, split='train')
    train_testvalid = dataset.train_test_split(test_size=0.5)
    train_dataset = train_testvalid['train']
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    eval_dataset = test_valid['train']
    test_dataset = test_valid['test']

    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    training_args = SFTConfig(
        output_dir="/h/smfsamir/hf_cache/olmo-1b-hf_task1_distillation",
        logging_steps=10,
        num_train_epochs=5,
        per_device_train_batch_size=2
    )
    trainer = SFTTrainer(
        "allenai/OLMo-1B-hf",
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

@click.group()
def main():
    pass

main.add_command(create_distillation_examples_task1)
main.add_command(distill_task1_olmo)
# main.add_command(create_distillation_examples_task1)

if __name__ == "__main__":
    main()