import random
import pathlib
import torch
import ipdb
from functools import partial
import loguru
import json
import os
import click
from dotenv import dotenv_values


from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForTokenClassification
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from packages.prompts.task_1_ner_distill_prompt import TASK_1_PROMPT

config = dotenv_values(".env")
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
def create_training_dataset_rolling():
    training_annotations = 'training_data.json'
    annotation_object = json.load(open(os.path.join("data", training_annotations)))

    def contained_in(entity, entities):
        """
        Check if the entity is contained in the list of entities.
        """
        for e in entities:
            if entity in e:
                return True
        return False

    paragraphs = []
    outputs = []
    subjects = []
    outlets = []
    current_mentioned_entities = []
    total_num_people = 0
    for fname in os.listdir("data/articles"):
        first_mentioned_entities = set([])
        victim_aligned_entities = annotation_object[fname]['task1']['Victim-aligned']
        police_aligned_entities = annotation_object[fname]['task1']['Police-aligned']
        total_people_in_article = len(victim_aligned_entities) + len(police_aligned_entities)
        task_2_corefs = annotation_object[fname]['task2'] # Dict[str, List[str]]
        perspectives_in_article = 0
        for i, paragraph in enumerate(json.load(open(os.path.join("data/articles", fname)))):
            num = i + 1
            if f"paragraph {num}" not in task_2_corefs:
                output = "No new entities in this paragraph."
            else:
                paragraph_entities = task_2_corefs[f"paragraph {num}"]
                new_entities = set(paragraph_entities) - first_mentioned_entities
                if len(new_entities) == 0:
                    output = "No new entities in this paragraph."
                else:
                    output = []
                    for entity in new_entities:
                        if contained_in(entity, victim_aligned_entities):
                            output.append(f"{entity} (victim-aligned)")
                            perspectives_in_article += 1
                        elif contained_in(entity, police_aligned_entities):
                            output.append(f"{entity} (police-aligned)")
                            perspectives_in_article += 1
                        else:
                            raise ValueError(f"Entity {entity} not found in either victim-aligned or police-aligned entities.")
                        total_num_people += 1
                    first_mentioned_entities.update(new_entities)
                    output = ", ".join(output)

            paragraphs.append(paragraph)
            current_mentioned_entities.append(tuple(first_mentioned_entities))
            outputs.append(output)
            subjects.append(fname.split("_")[1])
            outlets.append(pathlib.Path(fname.split("_")[2]).stem)
        if perspectives_in_article < total_people_in_article - 1: # subtract victims
            logger.warning(f"Article {fname} has fewer perspectives ({perspectives_in_article}) than total people ({total_people_in_article - 1}).")
    logger.info(f"Total number of people identified: {total_num_people}")
    dataset = Dataset.from_dict({
        'paragraph': paragraphs,
        'output': outputs,
        'subject': subjects,
        'outlet': outlets,
        'current_mentioned_entities': current_mentioned_entities
    })
    dataset.to_json("data/distillation_data/rolling_training_dataset.json")
    logger.info("Rolling training dataset created successfully.")

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

# def compute_metrics_flan(eval_preds):


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

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs['tokenizer']

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ):
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # Perform decoding and loss calculations here
            model = self.model
            tokenizer = self.tokenizer
            for i, _data in enumerate(eval_dataloader):
                example_text = tokenizer.batch_decode(_data['input_ids'], skip_special_tokens=True)[0]
                input_example = example_text[:example_text.rfind('\n\n')] + "### Answer:"
                tokenized_input = tokenizer(input_example, return_tensors="pt").to(model.device)
                prediction = model.generate(
                    **tokenized_input, 
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.95,
                    max_new_tokens=1000
                )
                prediction_text = tokenizer.batch_decode(prediction, skip_special_tokens=True)[0]
                prediction_text_answer = prediction_text[prediction_text.rfind("### Answer:") + len("### Answer:"):].strip()
                logger.info(f"ANSWER: {prediction_text_answer}")
                break
                if i == 0:
                    logger.info(f"Eval batch {_data}")
            metrics = {'wer': 0}
            return metrics

def preprocess_text(samples):
    batch = OLMO_TOKENIZER([samples['prompt'][i] + samples['completion'][i] for i in range(len(samples['prompt']))], 
                           padding=True,
                           return_tensors="pt")
    batch['labels'] = batch['input_ids'].clone()
    return batch

def get_model(model_name):
    if model_name == 'olmo':
        model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    elif model_name == 'meta':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    elif model_name == 'olmo-7b':
        model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    return model

def get_tokenizer(model_name):
    cache_dir = os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    if model_name == 'olmo':
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", cache_dir=cache_dir)
    elif model_name == 'meta':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@click.command()
@click.option('--learning_rate', type=float, default=2e-5)
@click.option('--num_training_steps', type=int, default=100)
@click.option('--warmup_steps', type=int, default=100)
@click.option('--weight_decay', type=float, default=0.01)
@click.option('--model_name', type=click.Choice(['olmo', 'flan', 'meta']), default='olmo')
def distill_task1_olmo(learning_rate, num_training_steps, warmup_steps, weight_decay, model_name):

    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    SCRATCH_DIR = config['SCRATCH_DIR']
    eval_dataset = load_dataset("json", data_files={'test': "data/distillation_data/distill_examples.json"}, split='test')
    train_dataset = load_dataset("json", data_files={'train': "data/distillation_data/train_distill_examples.json"}, split='train')
    train_dataset = train_dataset.map(
        preprocess_text, 
        batched=True,
    )
    eval_dataset = eval_dataset.map(
        preprocess_text, 
        batched=True,
    )

    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    training_arguments = TrainingArguments(
        output_dir=os.path.join(SCRATCH_DIR, "sympathy_task_1"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=num_training_steps,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=100,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps
    )
    collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator, 
        tokenizer=tokenizer
    )
    trainer.train()


def preprocess_flan_fn(sample):
    prompt = f"Here's an article about {sample['subject']}, who was killed by police as reported by {sample['outlet']}. Identify entities (people, organizations) who are expressing a perspective about the incident. Here are the roles we've identified so far: {sample['current_mentioned_entities']}. Identify NEW people/agencies providing a perspective, if any, in this paragraph:\n\n{sample['paragraph']}"
    return {'prompt': prompt, 'response': sample['output']}

def tokenize_batch_flan_fn(samples):
    model_inputs = FLAN_TOKENIZER(samples['prompt'], padding=True, truncation=True, return_tensors="pt")
    labels = FLAN_TOKENIZER(samples['response'], padding=True, truncation=True, return_tensors="pt")['input_ids']
    model_inputs['labels'] = labels
    return model_inputs

@click.command()
def distill_flant5():

    FLAN_TOKENIZER = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )

    dataset = load_dataset("json", data_files={'train': "data/distillation_data/rolling_training_dataset.json"}, split='train')
    # rolling_training_dataset.json
    # split train_dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    train_dataset = train_dataset.map(
        preprocess_flan_fn, 
        remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
    ).map(
        tokenize_batch_flan_fn, 
        batched=True,
    )
    eval_dataset = eval_dataset.map(
        preprocess_flan_fn, 
        remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
    ).map(
        tokenize_batch_flan_fn, 
        batched=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config['SCRATCH_DIR'], "sympathy_task_1_flan"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=1000,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100
    )
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        FLAN_TOKENIZER,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator, 
        tokenizer=FLAN_TOKENIZER
    )
    trainer.train()


@click.group()
def main():
    pass
import random

# Define hyperparameter ranges
param_ranges = {
    "learning_rate": (1e-5, 1e-3),   # float range (log-uniform recommended)
    "weight_decay": (0.0, 0.3),      # float range
    "training_steps": (100, 200)       # int range
}

def random_hyperparams(n_trials=10, seed=None):
    if seed is not None:
        random.seed(seed)

    configs = []
    for _ in range(n_trials):
        training_steps = random.randint(*param_ranges["training_steps"])
        config = {
            # log-uniform sampling for learning rate
            "learning_rate": 10 ** random.uniform(-5, -3),  
            "weight_decay": random.uniform(*param_ranges["weight_decay"]),
            "training_steps": training_steps,
            "warmup_steps": random.randint(0, training_steps - 1)  # strictly less
        }
        configs.append(config)
    return configs

@click.command()
def compute_required_memory():
    model = get_model("olmo-7b") 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Memory for weights (float16): {num_params * 2 / 1024**3:.2f} GB")
    pass

@click.command()
def assess_baseline_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    eval_dataset = load_dataset("json", data_files={'test': "data/distillation_data/distill_examples.json"}, split='test')

    ipdb.set_trace()
    eval_dataset = eval_dataset.map(
        lambda samples: tokenizer(samples['prompt'], padding=True, truncation=True, return_tensors="pt"), 
        batched=True,
    )
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

@click.command()
def assess_ft_flan_model():
    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(
        cache_dir=os.path.join(config['SCRATCH_DIR'], "sympathy_task_1_flan", "checkpoint-1000")
    )
    dataset = load_dataset("json", data_files={'train': "data/distillation_data/rolling_training_dataset.json"}, split='train')
    # rolling_training_dataset.json
    # split train_dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    eval_dataset = eval_dataset.map(
        preprocess_flan_fn, 
        remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
    ).map(
        tokenize_batch_flan_fn, 
        batched=True,
    )

    predictions = flan_t5.generate(
        input_ids=eval_dataset['input_ids'][:8], 
        attention_mask=eval_dataset['attention_mask'], 
        max_new_tokens=300
    )
    predicted_texts = FLAN_TOKENIZER.batch_decode(predictions, skip_special_tokens=True)
    ipdb.set_trace()

    pass

main.add_command(create_distillation_examples_task1)
main.add_command(distill_task1_olmo)
main.add_command(distill_flant5)
main.add_command(create_training_dataset)
main.add_command(create_training_dataset_rolling)
main.add_command(compute_required_memory)
main.add_command(assess_baseline_ner_model)
main.add_command(create_training_dataset_rolling)
main.add_command(assess_ft_flan_model)
# main.add_command(create_distillation_examples_task1)

if __name__ == "__main__":
# Example: generate 5 random configs
    main()
    # trials = random_hyperparams(n_trials=5, seed=42)
    # # learning_rates = " ".join([f'{trial['learning_rate']:.6f}' for trial in trials])
    # learning_rates = " ".join([f"{trial['learning_rate']:.6f}" for trial in trials])
    # training_steps = " ".join([str(trial['training_steps']) for trial in trials])
    # warmup_steps = " ".join([str(trial['warmup_steps']) for trial in trials])
    # weight_decays = " ".join([str(trial['weight_decay']) for trial in trials])
    # print(f"LRS=({learning_rates})")
    # print(f"TRAINING_STEPS=({training_steps})")
    # print(f"WARMUP_STEPS=({warmup_steps})")
    # print(f"WEIGHT_DECAYS=({weight_decays})")

    # # main()