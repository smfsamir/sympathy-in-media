import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CorefEntityMetadata:
    cluster_strings: List[str] # this will be provided as input
    cluster_indices: List[Tuple[int, int]] # this will be provided as input
    auto_paragraph_indices: List[int] # this will be provided as input
    valid_entity: bool # this needs to be predicted at inference time
    police_aligned: str # yes/no, or NA if not valid_entity. This needs to be predicted at inference time
    entity_name: str # the name of the entity, NA if not valid_entity. Otherwise, the name should be present in paragraph_indices. This needs to be predicted at inference time

def get_manual_annotation_occurrences(manual_annotation_obj, entity) -> List[int]:
    occurrences = []
    for paragraph_key, entity_list in manual_annotation_obj['task2'].items():
        if entity in entity_list:
            occurrences.append(int(paragraph_key.split(' ')[1]))
    return occurrences


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

def load_training_data_annotations_for_person(person_name: str, outlet: str,
                                              identifier: Optional[int] = -1):
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    training_data_annotation = None
    matched_annotations = []
    for k, v in training_data_annotations.items():
        if person_name in k and outlet in k:
            if identifier != -1 and f"{identifier}" not in k:
                continue
            training_data_annotation = v
            matched_annotations.append(training_data_annotation)
    assert training_data_annotation is not None, f"Could not find training data annotation for {person_name} and {outlet}."
    assert len(matched_annotations) == 1, f"Found multiple training data annotations for {person_name} and {outlet}."
    return training_data_annotation

def load_article_paragraphs(article: str) -> List[str]:
    with open(f'data/articles/{article}', 'r') as f:
        paragraphs = json.load(f)
    return paragraphs

def retrieve_entity_from_coref_objs(coref_objs: List[CorefEntityMetadata], 
                                    entity_name) -> CorefEntityMetadata:
    for coref_obj in coref_objs:
        if coref_obj.entity_name.lower() == entity_name.lower():
            return coref_obj
    raise ValueError(f"Could not find coref object for entity {entity_name}.")