import loguru
import ipdb
from typing import Dict

logger = loguru.logger
def generate_predictions(model, tokenizer, batch) -> Dict:
    inputs = tokenizer(batch['prompt'], return_tensors='pt', padding=True).to('cuda')
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=300
    )
    batch['predicted_text'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

def generate_singleton_prediction(model, tokenizer, example) -> Dict:
    inputs = tokenizer(example['prompt'], return_tensors='pt').to('cuda')
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=300
    )
    ipdb.set_trace()
    example['predicted_text'] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return example

def evaluate_entity_identified(example) -> Dict: # not batched
    no_entity_str = "there is no valid entity providing a perspective here."
    entity_present_str = "the entity name is"
    ground_truth = example['completion'].lower()
    prediction = example['predicted_text'].lower()
    if entity_present_str in ground_truth:
        example['valid_entity'] = 1
        if entity_present_str in prediction:
            is_correct = 1
        else:
            is_correct = 0
    elif no_entity_str in ground_truth:
        if no_entity_str in prediction:
            is_correct = 1
        else:
            is_correct = 0
    else:
        logger.warning(f"Ground truth not in expected format: {ground_truth}")
        raise ValueError(f"Ground truth not in expected format: {ground_truth}")
    return is_correct