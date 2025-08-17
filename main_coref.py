import json
import click
import ipdb
import loguru
from fastcoref import FCoref

logger = loguru.logger

def compute_paragraph_boundaries(paragraphs):
    paragraph_boundaries =[]
    start = 0
    end = 0
    for i, paragraph in enumerate(paragraphs):
        if i == len(paragraphs) - 1:
            end = start + len(paragraph)
        else: 
            end = start + len(paragraph) + 1
        paragraph_boundaries.append((start, end))
        start = end
    return paragraph_boundaries

@click.command()
def compute_fastcoref_annotations():
    model = FCoref(device='cuda:0')
    articles = ["78_Radford James Good Dagger_Global News.json"] # TODO: fill in the 10 articles.
    fcoref_annotations = []
    for article in articles:
        paragraphs = json.load(f'data/articles/{article}')
        full_text = " ".join(paragraphs)
        paragraph_boundaries = compute_paragraph_boundaries(paragraphs)
        preds = model.predict(texts=[full_text])[0]
        clusters = preds.get_clusters()
        cluster_indices = preds.get_clusters(as_strings=False)
        fcoref_annotations.append({
            'article': article,
            'cluster_indices': cluster_indices,
            'clusters': clusters,
            'full_text': full_text,
            'paragraph_boundaries': paragraph_boundaries
        })
    with open('data/fcoref_annotations.json', 'w') as f:
        json.dump(fcoref_annotations, f, indent=4)

# model = FCoref(device='cuda:0')
@click.group()
def main():
    pass

main.add_command(compute_fastcoref_annotations)

if __name__ == "__main__":
    main()