from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import click 
import loguru

logger = loguru.logger
from packages.coref_utils import compute_fastcoref_annotation
from packages.parsing_utils import CorefEntityInferenceMetadata, get_paragraph_occurrences, load_article_paragraphs

@click.command()
def inspect_distributions_unsupervised():
    years = []
    outlets = []
    for article in os.listdir('unsupervised_articles'):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            obj = json.load(f)
            date = obj['incident_date']
            year = date.split('-')[0]
            years.append(year)
            outlet = obj['publisher']
            outlets.append(outlet)

    # sort by year
    years.sort()
    sns.histplot(years)
    # rotate xtick labels
    plt.xticks(rotation=45)
    plt.savefig('unsupervised_article_years_distribution.png')

    # only include the top 30 outlets, and put the rest in other
    outlet_counts = {}
    for outlet in outlets:
        if outlet not in outlet_counts:
            outlet_counts[outlet] = 0
        outlet_counts[outlet] += 1

    # create a new figure
    plt.figure()
    sorted_outlet_counts = sorted(outlet_counts.items(), key=lambda x: x[1], reverse=True)
    top_outlets = sorted_outlet_counts[:30]
    top_outlet_names = [x[0] for x in top_outlets]
    top_outlet_values = [x[1] for x in top_outlets]
    other_count = sum([x[1] for x in sorted_outlet_counts[30:]])
    top_outlet_names.append('Other')
    top_outlet_values.append(other_count)
    sns.barplot(x=top_outlet_names, y=top_outlet_values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('unsupervised_article_outlet_distribution.png')


@click.command()
def compute_fastcoref_annotations():
    from fastcoref import FCoref
    model = FCoref(device='cuda:0')

    all_articles = os.listdir('unsupervised_articles')
    fcoref_annotations = []
    for article in all_articles:
        with open(f'unsupervised_articles/{article}', 'r') as f:
            paragraphs = json.load(f)['article']
        fcoref_annotations.append(
            compute_fastcoref_annotation(model, paragraphs, article) 
        )
    with open('data/unsupervised_fcoref_annotations.json', 'w') as f:
        json.dump(fcoref_annotations, f, indent=4)

@click.command()
def compute_article_basis_coref_objects():
    with open('data/unsupervised_fcoref_annotations.json', 'r') as f:
        fcoref_annotations = json.load(f)
    for article_annotations in tqdm(fcoref_annotations):
        coref_inference_objects = []
        article = article_annotations['article']
        cluster_strings = article_annotations['clusters']
        cluster_occurrence_indices = article_annotations['cluster_indices']
        for i in range(len(cluster_strings)):
            coref_paragraph_occurrences = get_paragraph_occurrences(
                    article_annotations['paragraph_boundaries'],
                    cluster_occurrence_indices[i]
                ) 
            coref_inference_objects.append(
                CorefEntityInferenceMetadata(
                    cluster_strings=cluster_strings[i],
                    cluster_indices=cluster_occurrence_indices[i],
                    auto_paragraph_indices=coref_paragraph_occurrences
                )
            )
            # save to file
        with open(f'data/unsupervised_coref_annotations/{article}', 'w') as f:
            json.dump([obj.__dict__ for obj in coref_inference_objects], f, indent=4)

def construct_length_limited_inference_prompt(victim_name: str, 
                                    coref_entity_obj: CorefEntityInferenceMetadata,
                                    all_paragraphs: List[str], 
                                    coref_auto_indices: List[int], 
                                    ) -> Dict:
    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    coref_auto_paragraphs = "\n".join([f"{i+1}. {all_paragraphs[index - 1]}" for i, index in enumerate(coref_auto_indices)])
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    index_to_auto_paragraph_index = {index: i+1 for i, index in enumerate(coref_auto_indices)}

    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    return {'prompt': task_instruction_str} 

def _create_inference_instance(victim_name: str, 
                              all_paragraphs: List[str], 
                              coref_metadata_obj: CorefEntityInferenceMetadata
                              ) -> List[Dict]:
    # TODO: need to return multiple instances, containing at most 5 paragraphs.

    MAX_PARAGRAPHS = 5
    # TODO: get the intersection of the paragraphs
    ## TODO: watch out for multiple paragraphs
    unique_auto_indices = list(sorted(list(set(coref_metadata_obj.auto_paragraph_indices))))
    num_prompts_required = len(unique_auto_indices) // MAX_PARAGRAPHS + (1 if len(unique_auto_indices) % MAX_PARAGRAPHS > 0 else 0)
    training_instances = []
    for i in range(num_prompts_required):
        coref_indices = unique_auto_indices[i*MAX_PARAGRAPHS:(i+1)*MAX_PARAGRAPHS]
        training_instances.append(construct_length_limited_inference_prompt(
            victim_name=victim_name,
            coref_entity_obj=coref_metadata_obj,
            all_paragraphs=all_paragraphs,
            coref_auto_indices=coref_indices
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
def create_coref_inference_dataset():
    from datasets import Dataset
    inference_articles = os.listdir("data/unsupervised_coref_annotations")
    outlets = []
    victim_names = []
    inference_instances = []
    article_indices = []
    for article in tqdm(inference_articles):
        article_index = article.split('_')[0]
        person_name = article.split('_')[1]
        outlet = article.split('_')[2]

        paragraphs = load_article_paragraphs(article, path="unsupervised_articles")
        coref_metadata_objects = [CorefEntityInferenceMetadata(**obj) for obj in json.load(open(os.path.join("data/unsupervised_coref_annotations", article)))]
        for coref_metadata_obj in coref_metadata_objects:
            inference_instances.extend(
                _create_inference_instance(
                    victim_name=person_name,
                    all_paragraphs=paragraphs,
                    coref_metadata_obj=coref_metadata_obj,
                    training_annotation_entity=None
                )
            )
            outlets.append(outlet)
            victim_names.append(person_name)
            article_indices.append(article_index)
    dataset = Dataset.from_dict({
        'victim_name': victim_names,
        'outlet': outlets,
        'prompt': [instance['prompt'] for instance in inference_instances],
        'article_index': article_indices
    })
    dataset.to_json("data/distillation_data/coref_inference_dataset.json")
    return dataset
        
@click.group()
def main():
    pass

main.add_command(inspect_distributions_unsupervised)
main.add_command(compute_fastcoref_annotations) # run this first
main.add_command(compute_article_basis_coref_objects)

if __name__ == '__main__':
    main()