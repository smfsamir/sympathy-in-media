import json
import os
import click 
import loguru

logger = loguru.logger
from packages.coref_utils import compute_fastcoref_annotation

@click.command()
def inspect_distributions_unsupervised():
    pass

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

@click.group()
def main():
    pass

main.add_command(inspect_distributions_unsupervised)
main.add_command(compute_fastcoref_annotations)

if __name__ == '__main__':
    main()