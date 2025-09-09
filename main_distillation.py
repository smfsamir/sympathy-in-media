import wandb
import pandas as pd
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
from typing import List, Dict


from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForTokenClassification
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from packages.prompts.task_1_ner_distill_prompt import TASK_1_PROMPT
from packages.parsing_utils import CorefEntityMetadata, get_manual_annotation_occurrences, load_article_paragraphs, load_repaired_articles, load_training_data_annotations_for_person
from packages.flan_utils import compute_metrics_tokenized_batch, generate_singleton_prediction, evaluate_entity_identified_single, generate_predictions, generate_predictions_tokenized_batch, convert_text_to_entity_present_label

config = dotenv_values(".env")
logger = loguru.logger
# message = ["Language modeling is "]
# inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# # optional verifying cuda
# # inputs = {k: v.to('cuda') for k,v in inputs.items()}
# # olmo = olmo.to('cuda')
# response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


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

class CustomSeq2SeqTrainer(Seq2SeqTrainer):

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
            all_eval_entity_present_gt_labels = []
            all_eval_entity_present_prediction_labels = []
            all_is_correct_labels = []
            all_cer_metrics = []
            for i, _data in enumerate(eval_dataloader): # should be batch size set by trainer.
                batch = generate_predictions_tokenized_batch(model, tokenizer, _data)
                batch_metrics = compute_metrics_tokenized_batch(batch)
                batch_entity_present_gt_labels = convert_text_to_entity_present_label(batch['label_text']) # TODO: implement this function
                batch_entity_present_predicted_labels = convert_text_to_entity_present_label(batch['predicted_text']) # TODO: implement this function
                all_eval_entity_present_gt_labels.extend(batch_entity_present_gt_labels)
                all_eval_entity_present_prediction_labels.extend(batch_entity_present_predicted_labels)
                all_cer_metrics.extend(batch_metrics['cer']) # TODO: double check these keys
                all_is_correct_labels.extend(batch_metrics['is_correct'])
                print(batch_metrics)
                ipdb.set_trace()
                #######
                example_input_text = tokenizer.batch_decode(_data['input_ids'], 
                                                      skip_special_tokens=True)[0]
                example_output_text = tokenizer.batch_decode(_data['labels'], 
                                                       skip_special_tokens=True)[0]
                if "the entity name is" not in example_output_text.lower():
                    continue # we're looking for a target JSON output
                logger.info(f"Target text: {example_output_text}")
                prediction_logits = model.generate(
                    input_ids=_data['input_ids'].to(self.args.device), 
                    attention_mask=_data['attention_mask'].to(self.args.device), 
                    max_new_tokens=300
                )                                    
                predicted_text = tokenizer.batch_decode(prediction_logits, skip_special_tokens=True)[0]
                if "There is no valid entity providing a perspective here." in predicted_text:
                    logger.info(f"Got a no entity response")
                # elif "{" in predicted_text and "}" in predicted_text:
                elif "the entity name is" in predicted_text.lower():
                    logger.info(f"Got a valid entity response: {predicted_text}")
                else:
                    logger.warning(f"Predicted text not in expected format: {predicted_text}")
                break
            f1_metric = f1_score(all_eval_entity_present_gt_labels, all_eval_entity_present_prediction_labels, pos_label='valid entity')
            accuracy_metric = sum(all_is_correct_labels) / len(all_is_correct_labels) # TODO
            cer_metric = np.median(cer_metrics)
            metrics = {'cer': cer_metric, 'accuracy': accuracy_metric, 'f1': f1_metric}
            return metrics

def tokenize_batch_flan_fn(tokenizer, samples):
    model_inputs = tokenizer(samples['prompt'], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(samples['completion'], padding=True, truncation=True, return_tensors="pt")['input_ids']
    model_inputs['labels'] = labels
    return model_inputs

def compute_metrics_flan():
    pass

@click.command()
def distill_flant5():

    FLAN_TOKENIZER = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )

    dataset = load_dataset("json", data_files={'train': "data/distillation_data/coref_training_dataset.json"}, split='train')

    subjects_unique = set(dataset['victim_name'])
    train_subjects = set(random.sample(subjects_unique, int(len(subjects_unique) * 0.6)))
    dev_subjects = subjects_unique - train_subjects

    logger.info(f"Train subjects: {train_subjects}")
    logger.info(f"Dev subjects: {dev_subjects}")
    # rolling_training_dataset.json
    # split train_dataset into train and validation sets
    train_dataset =  dataset.filter(lambda example: example['victim_name'] in train_subjects)
    eval_dataset = dataset.filter(lambda example: example['victim_name'] in dev_subjects)

    # check the fraction of valid entities in the train and eval sets
    logger.info(f"Train valid entities proportion: {sum(train_dataset['valid_entity'])} / {len(train_dataset)}")
    logger.info(f"Eval valid entities proportion: {sum(eval_dataset['valid_entity'])} / {len(eval_dataset)}")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    new_tokens = ["{", "}"]
    new_tokens = set(new_tokens) - set(FLAN_TOKENIZER.vocab.keys())
    new_tokens = list(new_tokens)
    FLAN_TOKENIZER.add_tokens(new_tokens)
    model.resize_token_embeddings(len(FLAN_TOKENIZER))
    train_dataset = train_dataset.map(
        partial(tokenize_batch_flan_fn, FLAN_TOKENIZER), 
        batched=True
    )
        # preprocess_flan_fn, 
        # remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
    eval_dataset = eval_dataset.map(
        partial(tokenize_batch_flan_fn, FLAN_TOKENIZER), 
        batched=True
    )
    wandb.init(project="sympathy")
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config['SCRATCH_DIR'], "sympathy_distillation"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=1000,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        report_to="wandb"
    )
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        FLAN_TOKENIZER,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator, 
        tokenizer=FLAN_TOKENIZER
    )
    trainer.train(resume_from_checkpoint=False)

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
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )
    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            config['SCRATCH_DIR'], 
            "sympathy_distillation", 
            "checkpoint-1000")
    ).to('cuda')

    dataset = load_dataset("json", 
                           data_files={'train': "data/distillation_data/coref_training_dataset.json"}, 
                           split='train')

    # rolling_training_dataset.json
    # split train_dataset into train and validation sets
    # eval_subjects = ['Danny Lafrance-Godmer', 'Jeremy Nuvviaq', 'Dale Culver', 'Jason Gary Roy', 'Bradley Thomas Clattenburg', 'Charles Qirngnirq', 'Riley Fairholm', 'Radford James Good Dagger', 'Elgyn Muskego', 'Illutak Anautak', 'Christopher Arkell', 'David Charles Sandaker', 'Abisay Cruz', 'William David McCaffrey', 'John Robert Buehler'] 
    eval_subjects = ['Chris Bloomfield', 
                     'Erixon Kabera', 
                     'Buck E Evans', 
                     'Babak Saidi', 
                     'David Meadows', 
                     'Pierre Charron', 
                     'Bony Jean-Pierre', 
                     'Jermaine Carby\t', 
                     'Eugene Ethan Marcano', 
                     'Dillon Warren Breed']

    eval_dataset = dataset.filter(lambda example: example['victim_name'] in eval_subjects) 
    
    def evaluate_entity_identified_batch(example): # not batched
        no_entity_str = "there is no valid entity providing a perspective here."
        entity_present_str = "the entity name is"
        ground_truth = example['completion'].lower()
        prediction = example['predicted_text'].lower()
        if entity_present_str in ground_truth:
            if entity_present_str in prediction:
                is_correct = True
            else:
                is_correct = False
        elif no_entity_str in ground_truth:
            if no_entity_str in prediction:
                is_correct = True
            else:
                is_correct = False
        else:
            logger.warning(f"Ground truth not in expected format: {ground_truth}")
            raise ValueError(f"Ground truth not in expected format: {ground_truth}")
        example['entity_identified_correct'] = is_correct
        return example

    eval_dataset = eval_dataset.map(
        partial(tokenize_batch_flan_fn, tokenizer), 
        batched=True
    )
    assert 'input_ids' in eval_dataset.column_names
    assert 'labels' in eval_dataset.column_names
    eval_dataset = eval_dataset.map(partial(generate_predictions, flan_t5, tokenizer),
                                    batched=True, 
                                    batch_size=8)
    eval_dataset = eval_dataset.map(evaluate_entity_identified_batch)
    ipdb.set_trace()

    trainer = CustomSeq2SeqTrainer(
        model=flan_t5,
        args=Seq2SeqTrainingArguments(
            output_dir=os.path.join(config['SCRATCH_DIR'], "sympathy_distillation"),
            per_device_eval_batch_size=8,
            report_to="none"
        ),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.evaluate()
    # for subject in eval_subjects:
    #     eval_subset = dataset.filter(lambda example: example['subject'] == subject) # is this still in the right order?
        # eval_subset = eval_subset.map(
        #     preprocess_flan_fn, 
        #     remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
        # ).map(
        #     tokenize_batch_flan_fn, 
        #     batched=True,
        # )
        # predictions = flan_t5.generate(
        #     input_ids=torch.tensor(eval_subset['input_ids']).to('cuda'), 
        #     attention_mask=torch.tensor(eval_subset['attention_mask']).to('cuda'), 
        #     max_new_tokens=300
        # )
        # predicted_texts = FLAN_TOKENIZER.batch_decode(predictions, skip_special_tokens=True)
        # ipdb.set_trace()

    pass

def construct_length_limited_prompt(victim_name: str, 
                                    coref_entity_obj: CorefEntityMetadata,
                                    all_paragraphs: List[str], 
                                    coref_auto_indices: List[int], 
                                    annotation_indices: List[int]) -> Dict:
    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    coref_auto_paragraphs = "\n".join([f"{i+1}. {all_paragraphs[index - 1]}" for i, index in enumerate(coref_auto_indices)])
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    index_to_auto_paragraph_index = {index: i+1 for i, index in enumerate(coref_auto_indices)}
    intersection_indices = list(sorted(set(annotation_indices).intersection(set(coref_auto_indices))))

    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    subset_indices = [index_to_auto_paragraph_index[index] for index in intersection_indices]
    #### Constructing the input
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 

    if coref_entity_obj.valid_entity: 
        # if len(intersection_indices) == 0: # TODO: this is more likely to happen, since you're not doing the split...
        # output_str = json.dumps({'entity_name': coref_entity_obj.entity_name, 'police_aligned': coref_entity_obj.police_aligned,'perspective_paragraphs': subset_indices})
        # output_str = json.dumps({'entity_name': coref_entity_obj.entity_name, 'police_aligned': coref_entity_obj.police_aligned,'perspective_paragraphs': subset_indices})
        # TODO: try a natural language output
        output_str = f"The entity name is {coref_entity_obj.entity_name}. They are {'aligned with the police' if coref_entity_obj.police_aligned else 'not aligned with the police'}. The paragraphs that reflect their perspectives are: {', '.join(map(str, subset_indices)) if len(subset_indices) > 0 else 'none'}."
    else:
        output_str = f"There is no valid entity providing a perspective here."
    return {'prompt': task_instruction_str, 'completion': output_str, 'entity_name': coref_entity_obj.entity_name, 'valid_entity': coref_entity_obj.valid_entity, victim_name: victim_name}

def _create_training_instance(victim_name: str, 
                              all_paragraphs: List[str], 
                              coref_metadata_obj: CorefEntityMetadata, 
                              training_annotation_entity: Dict) -> List[Dict]:
    # TODO: need to return multiple instances, containing at most 5 paragraphs.

    MAX_PARAGRAPHS = 5
    # TODO: get the intersection of the paragraphs
    ## TODO: watch out for multiple paragraphs
    unique_auto_indices = list(sorted(list(set(coref_metadata_obj.auto_paragraph_indices))))
    num_prompts_required = len(unique_auto_indices) // MAX_PARAGRAPHS + (1 if len(unique_auto_indices) % MAX_PARAGRAPHS > 0 else 0)
    training_instances = []
    all_perspective_paragraphs_empty = True
    for i in range(num_prompts_required):
        coref_indices = unique_auto_indices[i*MAX_PARAGRAPHS:(i+1)*MAX_PARAGRAPHS]
        training_instances.append(construct_length_limited_prompt(
            victim_name=victim_name,
            coref_entity_obj=coref_metadata_obj,
            all_paragraphs=all_paragraphs,
            coref_auto_indices=coref_indices,
            annotation_indices=get_manual_annotation_occurrences(training_annotation_entity, coref_metadata_obj.entity_name) if coref_metadata_obj.valid_entity else []
        ))
    # NOTE: uncomment this after figuring out how to load the set of perspective paragraphs in the natural language format
    #     if coref_metadata_obj.valid_entity:
    #         output_dict = json.loads(training_instances[-1]['completion'])
    #         if len(output_dict['perspective_paragraphs']) > 0:
    #             all_perspective_paragraphs_empty = False

    # if coref_metadata_obj.valid_entity and all_perspective_paragraphs_empty:
    #     logger.warning("Valid entity but no perspective paragraphs found")
    #     ipdb.set_trace()
    return training_instances

    
@click.command()
def create_coref_training_dataset():
    def _remove_repaired_suffix(article_name: str) -> str:
        assert '_repaired' in article_name, f"Article name {article_name} does not contain '_repaired'"
        return article_name.replace('_repaired', '')
    # write a function to create a coreference resolution training dataset.

    # TODO: note that some people have multiple articles from one outlet (Charles Qirnirq)
    all_articles = os.listdir("data/linked_coref_annotations")
    imperfect_articles = pd.read_csv('data/imperfect_articles.csv')['article'].tolist()
    repaired_articles = load_repaired_articles()
    perfect_articles = set(all_articles) - set(imperfect_articles)
    training_set = []
    victim_names = []
    for article in perfect_articles.union(repaired_articles):
        # load the coref object and the annotation object
        article_index = article.split('_')[0]
        person_name = article.split('_')[1]
        outlet = article.split('_')[2]
        annotations = load_training_data_annotations_for_person(person_name, outlet, identifier=article_index)
        # coref_metadata_objects = [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/coref_metadata", article)))]
        if '_repaired' in article:
            paragraphs = load_article_paragraphs(_remove_repaired_suffix(article))
        elif article in perfect_articles:
            paragraphs = load_article_paragraphs(article)

        if article in perfect_articles:
            coref_metadata_objects = [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/linked_coref_annotations", article)))]
        # TODO: load the repaired articles here, separately.
        elif article in repaired_articles:
            coref_metadata_objects= [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/repaired_coref_annotations", article)))]
        else:
            raise ValueError(f"Article {article} not found in perfect or repaired articles.")
        for coref_obj in coref_metadata_objects:
            training_instances = _create_training_instance(
                victim_name=person_name,
                all_paragraphs=paragraphs,
                coref_metadata_obj=coref_obj, # just do the first one for now
                training_annotation_entity=annotations
            )
            training_set.extend(training_instances)
            victim_names.extend([person_name] * len(training_instances))
    dataset = Dataset.from_dict({
        'prompt': [instance['prompt'] for instance in training_set],
        'completion': [instance['completion'] for instance in training_set],
        'entity_name': [instance['entity_name'] for instance in training_set],
        'valid_entity': [instance['valid_entity'] for instance in training_set],
        'victim_name': victim_names
    })
    # log the number of valid entities relative to the total
    logger.info(f"Number of valid entities: {sum(dataset['valid_entity'])} / {len(dataset)}")
    dataset.to_json("data/distillation_data/coref_training_dataset.json")

main.add_command(distill_flant5)
# main.add_command(create_training_dataset_rolling)
main.add_command(compute_required_memory)
main.add_command(assess_baseline_ner_model)
# main.add_command(create_training_dataset_rolling)
main.add_command(assess_ft_flan_model)
main.add_command(create_coref_training_dataset)
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
