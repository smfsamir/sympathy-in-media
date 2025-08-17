import json
import click
import ipdb
import loguru

logger = loguru.logger

def compute_paragraph_boundaries(paragraphs):
    from fastcoref import FCoref
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
    from fastcoref import FCoref
    model = FCoref(device='cuda:0')
    articles = ["78_Radford James Good Dagger_Global News.json", 
                "99_Hudson Daryl Willis_Surrey Now-Leader.json",
                "28_Alex Wettlaufer_Global News.json",
                "44_Gerald Rattu_Durham Region.json",
                "80_Joey Knapaysweet_TimminsToday.com.json",
                "73_Sterling Ross Cardinal_CBC.json",
                "15_Bill Saunders_CTV.json",
                "97_Maurizio Angelo Facchin_Burnaby Now.json",
                "89_Jason Gary Roy_Calgary Herald.json"] # TODO: fill in the 10 articles.
    fcoref_annotations = []
    for article in articles:
        with open(f'data/articles/{article}', 'r') as f:
            paragraphs = json.load(f)
        full_text = " ".join(paragraphs)
        paragraph_boundaries = compute_paragraph_boundaries(paragraphs)
        preds = model.predict(texts=[full_text])[0]
        clusters = preds.get_clusters()
        cluster_indices = preds.get_clusters(as_strings=False)
        # TODO: have to put the target paragraphs into the clusters.
        fcoref_annotations.append({
            'article': article,
            'cluster_indices': cluster_indices,
            'clusters': clusters,
            'full_text': full_text,
            'paragraph_boundaries': paragraph_boundaries
        })
    with open('data/fcoref_annotations.json', 'w') as f:
        json.dump(fcoref_annotations, f, indent=4)

@click.command()
def inspect_annotations():
    with open('data/fcoref_annotations.json', 'r') as f:
        coref_annotations = json.load(f)
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    for article_annotations in coref_annotations:
        article = article_annotations['article']
        manual_annotations = training_data_annotations[article]
        entities = manual_annotations['task1']['Victim-aligned'] + manual_annotations['task1']['Police-aligned']
        assert entities[0].endswith(' (victim)')
        annotated_entities = entities[1:]
        annotated_entities = [entity.lower() for entity in annotated_entities]
        # remove the parenthetical element from the entity
        annotated_entities = [entity.split(' (')[0] for entity in annotated_entities]
        print(annotated_entities)
        clusters = article_annotations['clusters']
        total_matches = 0
        found_entities = []
        for entity in annotated_entities:
            found = False
            for cluster in clusters:
                for automatic_entity in cluster:
                    automatic_entity = automatic_entity.lower()
                    if entity in automatic_entity:
                        total_matches += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.warning(f"Entity {entity} not found in article {article}.")
            else:
                found_entities.append(entity)
            # for automatic_entity in cluster:
            #     automatic_entity = automatic_entity.lower()
            #     if automatic_entity in annotated_entities:
            #         total_matches += 1
        logger.info(f"{total_matches}/{len(annotated_entities)} entities matched for article {article}.")

# model = FCoref(device='cuda:0')
@click.group()
def main():
    pass

main.add_command(compute_fastcoref_annotations)
main.add_command(inspect_annotations)

if __name__ == "__main__":
    main()