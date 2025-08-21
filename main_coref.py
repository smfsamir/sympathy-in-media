import numpy as np
import os
import json
import click
from typing import List, Tuple
import ipdb
import loguru
from dataclasses import dataclass

@dataclass
class CorefEntityMetadata:
    cluster_strings: List[str] # this will be provided as input
    cluster_indices: List[Tuple[int, int]] # this will be provided as input
    auto_paragraph_indices: List[int] # this will be provided as input
    valid_entity: bool # this needs to be predicted at inference time
    police_aligned: str # yes/no, or NA if not valid_entity. This needs to be predicted at inference time
    entity_name: str # the name of the entity, NA if not valid_entity. Otherwise, the name should be present in paragraph_indices. This needs to be predicted at inference time

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
    from fastcoref import FCoref
    model = FCoref(device='cuda:0')
    all_articles = os.listdir('data/articles')
    # get a 50-25-25 random split of the articles, for a train/valid/test set
    train_indices = np.random.choice(len(all_articles), size=int(0.5*len(all_articles)), replace=False)
    remaining_indices = list(set(range(len(all_articles))) - set(train_indices))
    valid_indices = np.random.choice(remaining_indices, size=int(0.25*len(all_articles)), replace=False)
    test_indices = list(set(remaining_indices) - set(valid_indices))

    train_articles = [all_articles[i] for i in train_indices]
    valid_articles = [all_articles[i] for i in valid_indices]
    test_articles = [all_articles[i] for i in test_indices]
    logger.info(f"Train articles: {train_articles}")
    logger.info(f"Valid articles: {valid_articles}")
    logger.info(f"Test articles: {test_articles}")

    # articles = ["78_Radford James Good Dagger_Global News.json", 
    #             "99_Hudson Daryl Willis_Surrey Now-Leader.json",
    #             "28_Alex Wettlaufer_Global News.json",
    #             "44_Gerald Rattu_Durham Region.json",
    #             "80_Joey Knapaysweet_TimminsToday.com.json",
    #             "73_Sterling Ross Cardinal_CBC.json",
    #             "15_Bill Saunders_CTV.json",
    #             "97_Maurizio Angelo Facchin_Burnaby Now.json",
    #             "89_Jason Gary Roy_Calgary Herald.json"] # TODO: fill in the 10 articles.

    fcoref_annotations = []
    for article in train_articles:
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

def check_if_entity_name_in_paragraphs(paragraphs: List[str], 
                                       paragraph_indices: List[int],
                                       entity_name) -> bool:
    """
    Check if the complete entity name is present in any of the paragraphs.
    """
    for index in paragraph_indices:
        paragraph = paragraphs[index - 1]
        if entity_name.lower() in paragraph.lower():
            return True
    return False


def get_paragraph_occurrences(paragraph_boundaries: List[Tuple],
                             entity_occurrences: List[Tuple]):
    occurrences = []
    for i, occurrence in enumerate(entity_occurrences):
        start, end = occurrence
        for j, (para_start, para_end) in enumerate(paragraph_boundaries):
            if start >= para_start and end <= para_end:
                occurrences.append(j + 1)  # +1 to make it 1-indexed
                break
    return occurrences

def get_manual_annotation_occurrences(manual_annotation_obj, entity):
    occurrences = []
    for paragraph_key, entity_list in manual_annotation_obj['task2'].items():
        if entity in entity_list:
            occurrences.append(paragraph_key)
    return occurrences

def write_annotation_report(article: str,
                            matched_entities: List[CorefEntityMetadata], 
                            unmatched_entities: List[CorefEntityMetadata],
                            unfound_entities: List[str] = []):
    article = article.split('.')[0]  # remove the .json extension
    if unfound_entities == []:
        article_fname =f"data/linked_coref_annotations/{article}.json"
    else:
        article_fname = f"data/linked_coref_annotations/{article}_unfound={len(unfound_entities)}.json" 
    with open(f"{article_fname}", 'w') as f:
        # write all the metadata as a json list.
        # do the matched entities first, and then the unmatched entities.
        all_metadata = [entity.__dict__ for entity in matched_entities] + \
            [entity.__dict__ for entity in unmatched_entities]
        json.dump(all_metadata, f, indent=4)
        
def produce_annotation_report(article, 
                              coref_annotation_obj,
                              manual_annotation_obj):
    print(f"===={article}=====")
    entities = manual_annotation_obj['task1']['Victim-aligned'] + \
        manual_annotation_obj['task1']['Police-aligned']
    assert entities[0].endswith(' (victim)')
    annotated_entities = entities[1:]
    annotated_entities = [entity.lower() for entity in annotated_entities]
    # remove the parenthetical element from the entity
    annotated_entities = [entity.split(' (')[0] for entity in annotated_entities]
    clusters = coref_annotation_obj['clusters']
    total_matches = 0
    found_entities = []

    # load the paragraphs from data/articles
    with open(f'data/articles/{article}', 'r') as f:
        paragraphs = json.load(f)

    valid_cluster_indices = []
    metadata_valid_entities = []
    metadata_invalid_entities = []
    for entity in annotated_entities:
        found_cluster_index = -1
        print(f"----Searching for entity: {entity}-----")
        found = False
        for i, cluster in enumerate(clusters):
            for automatic_entity in cluster:
                automatic_entity = automatic_entity.lower()
                if entity in automatic_entity:
                    total_matches += 1
                    found = True
                    found_cluster_index = i
                    break
            if found:
                break

        if not found:
            logger.warning(f"Entity {entity} not found in article {article}.")
        else:
            assert found_cluster_index != -1
            coref_paragraph_occurrences = get_paragraph_occurrences(
                coref_annotation_obj['paragraph_boundaries'],
                coref_annotation_obj['cluster_indices'][found_cluster_index]
            ) 
            annotation_indices = get_manual_annotation_occurrences(
                manual_annotation_obj, entity)
            found_entities.append(entity)
            print("Coref indices:", coref_paragraph_occurrences)
            print("Manual annotation indices:", annotation_indices)
        
            # construct the metadata object
            is_police_aligned = any([iter_entity.startswith(entity) for iter_entity in manual_annotation_obj['task1']['Police-aligned']])
            metadata_valid_entity = CorefEntityMetadata(
                cluster_strings=clusters[found_cluster_index],
                cluster_indices=coref_annotation_obj['cluster_indices'][found_cluster_index],
                auto_paragraph_indices=coref_paragraph_occurrences,
                valid_entity=True,
                police_aligned='yes' if is_police_aligned else 'no',
                entity_name=entity
            )
            metadata_valid_entities.append(metadata_valid_entity)
            valid_cluster_indices.append(found_cluster_index)
            # assert that the entity is present in the text in the paragraph indices
            assert check_if_entity_name_in_paragraphs(
                paragraphs, 
                coref_paragraph_occurrences, 
                entity
            ), f"Entity {entity} not found in paragraphs for article {article}."
    logger.info(f"{total_matches}/{len(annotated_entities)} entities matched for article {article}.")

    # iterate over the cluster indices that were not validated, and create metadata entries for them
    unfound_entities = set(annotated_entities) - set(found_entities)

    for i, cluster in enumerate(coref_annotation_obj['clusters']):
        if i not in valid_cluster_indices:
            # create a metadata object for this cluster
            coref_paragraph_occurrences = get_paragraph_occurrences(
                coref_annotation_obj['paragraph_boundaries'],
                coref_annotation_obj['cluster_indices'][i]
            )
            if len(unfound_entities) == 0:
                overlap_f1s = [0.0] * len(coref_paragraph_occurrences)
            else:
                overlap_f1s = []
            for entity in unfound_entities:
                manual_paragraph_occurrences = get_manual_annotation_occurrences(
                    manual_annotation_obj,
                    entity)  
                # compute the f1 overlap with the manual annotation occurrences
                overlap = set(coref_paragraph_occurrences) & set(manual_paragraph_occurrences)
                if len(overlap) > 0:
                    overlap_f1 = len(overlap) / (len(coref_paragraph_occurrences) + len(manual_paragraph_occurrences) - len(overlap))
                    overlap_f1s.append(overlap_f1)
                else:
                    overlap_f1s.append(0.0)
            metadata_invalid_entity = CorefEntityMetadata(
                cluster_strings=cluster,
                cluster_indices=coref_annotation_obj['cluster_indices'][i],
                auto_paragraph_indices=coref_paragraph_occurrences,
                valid_entity=False,
                police_aligned='NA',
                entity_name='NA'
            )
            # TODO: sort by max f1-overlap with the missing entities.
            metadata_invalid_entities.append((metadata_invalid_entity, max(overlap_f1s)))
    # sort the invalid entities by the max overlap f1 score
    metadata_invalid_entities.sort(key=lambda x: x[1], reverse=True)
    metadata_invalid_entities = [entity[0] for entity in metadata_invalid_entities]
    write_annotation_report(article, 
                            metadata_valid_entities, 
                            metadata_invalid_entities, 
                            list(unfound_entities))
    

@click.command()
def inspect_annotations():
    with open('data/fcoref_annotations.json', 'r') as f:
        coref_annotations = json.load(f)
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    for article_annotations in coref_annotations:
        article = article_annotations['article']
        manual_annotations = training_data_annotations[article]
        produce_annotation_report(article, 
                                  article_annotations, 
                                  manual_annotations)
# model = FCoref(device='cuda:0')
@click.group()
def main():
    pass

# @click.command()
# def annotate_missing_entities():

    
main.add_command(compute_fastcoref_annotations)
main.add_command(inspect_annotations)   

if __name__ == "__main__":
    main()