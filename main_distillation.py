import torch
import ipdb
from functools import partial
import loguru
import json
import os
import pathlib
import click

from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from packages.prompts.task_1_ner_distill_prompt import TASK_1_PROMPT
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

logger = loguru.logger
# message = ["Language modeling is "]
# inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# # optional verifying cuda
# # inputs = {k: v.to('cuda') for k,v in inputs.items()}
# # olmo = olmo.to('cuda')
# response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

OLMO_TOKENIZER = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
FLAN_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
META_TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

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
    dataset.to_json("data/distillation_data/eval_distill_examples.json") 
    logger.info("Distillation examples created successfully.")
        # Add more examples or prompts as needed
        # f.write("Another example prompt here\n")

@click.command()
def create_training_dataset():
    prompts = []
    completions = []
    training_annotations = 'training_data.json'
    annotation_object = json.load(open(os.path.join("data", training_annotations)))
    eval_fnames = os.listdir("data/evaluation_dataset")

    paragraph_num_tokens = []
    flan_num_tokens = []
    num_words = []

    for fname in os.listdir("data/articles"):
        if fname in eval_fnames:
            continue
        else:
            person_annotations = annotation_object[fname]
        article_paragraphs = '\n'.join([f"{i+1}. {paragraph}" for i, paragraph in enumerate(json.load(open(os.path.join("data/articles", fname))))])
        paragraph_num_tokens.append(len(OLMO_TOKENIZER(article_paragraphs)['input_ids']))
        flan_num_tokens.append(len(FLAN_TOKENIZER(article_paragraphs)['input_ids']))
        num_words.append(len(article_paragraphs.split()))
        prompt = f"{TASK_1_PROMPT}" + article_paragraphs
        response = f"### Answer: {json.dumps(person_annotations['task1'])}"
        prompts.append(prompt)
        completions.append(response)
    dataset = Dataset.from_dict({
        'prompt': prompts,
        'completion': completions
    }) 
    prompt_length = len(OLMO_TOKENIZER.encode(TASK_1_PROMPT))
    dataset.to_json("data/distillation_data/train_distill_examples.json") 
    # with open("data/distillation_data/train_distill_examples.json", "r") as f:
    #     object = json.load(f)
    dataset = load_dataset("json", data_files={'train': "data/distillation_data/train_distill_examples.json"}, split='train')
    logger.info("Distillation examples created successfully.")
    pass

def compute_metrics(eval_preds):
    arr = eval_preds.label_ids
    arr = arr[0]
    arr = arr[arr != -100]
    label_text = OLMO_TOKENIZER.decode(arr, skip_special_tokens=True)
    logger.info(f"Ground Truth: {label_text}")

    predictions = eval_preds.predictions[0]
    mask = ~(predictions == -100).all(axis=1)
    predictions = predictions[mask]
    predicted_string = OLMO_TOKENIZER.decode(predictions.argmax(axis=1), skip_special_tokens=True)
    logger.info(f"Prediction: {predicted_string}")
    return {"accuracy": 0}

def preprocess_function(tokenizer, sample):
    model_inputs = tokenizer(sample['prompt']) # don't pad in preprocessing
    label_str = f"{sample['completion']}"
    # json stringifying the label_str
    labels = tokenizer((label_str))
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    #     ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

@click.command()
def distill_task1_olmo():
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    eval_dataset = load_dataset("json", data_files={'test': "data/distillation_data/distill_examples.json"}, split='test')
    train_dataset = load_dataset("json", data_files={'train': "data/distillation_data/train_distill_examples.json"}, split='train')
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    # tokenizer.pad_token = tokenizer.eos_token

    # collator = DataCollatorForCompletionOnlyLM(instruction_template="### Instruction:", 
    #                                            response_template="### Answer:", 
    #                                            tokenizer=tokenizer, 
    #                                            mlm=False
    training_args = SFTConfig(
        max_length=3300,
        output_dir="/h/smfsamir/hf_cache/olmo-1b-hf_task1_distillation",
        eval_steps = 10,
        eval_strategy="steps"
    )
    #                                            ) 
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        args=training_args,
        # data_collator=collator,
        compute_metrics=compute_metrics # TODO: double check that you're using the right tokenizer in the compute_metrics function
    )
    trainer.train()

@click.group()
def main():
    pass

main.add_command(create_distillation_examples_task1)
main.add_command(distill_task1_olmo)
main.add_command(create_training_dataset)
# main.add_command(create_distillation_examples_task1)

if __name__ == "__main__":
    main()