from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import click 
import loguru

logger = loguru.logger
from packages.coref_utils import compute_fastcoref_annotation
from packages.parsing_utils import CorefEntityInferenceMetadata, get_paragraph_occurrences

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
        
@click.group()
def main():
    pass

main.add_command(inspect_distributions_unsupervised)
main.add_command(compute_fastcoref_annotations) # run this first
main.add_command(compute_article_basis_coref_objects)

if __name__ == '__main__':
    main()