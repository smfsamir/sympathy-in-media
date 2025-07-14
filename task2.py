import spacy
import neuralcoref

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

SPEECH_VERBS = {"say", "tell", "report", "claim", "state", "add", "explain", "note", "argue", "declare"}
SUBJECT_DEPS = {"nsubj", "nsubjpass"}
PRONOUNS = {"he","she","they","his","her","its","we","I"}

def get_paragraph_perspectives(paragraphs, known_entities):
    ents = set(known_entities)
    para_map = {}

    for idx, para in enumerate(paragraphs):

        doc = nlp(para)
        resolved = nlp(doc._.coref_resolved)
        speakers = set()

        for sent in resolved.sents:
            if not any(tok.lemma_.lower() in SPEECH_VERBS and tok.pos_=="VERB"
                       for tok in sent):
                continue

            for verb in sent:
                if verb.lemma_.lower() not in SPEECH_VERBS or verb.pos_!="VERB":
                    continue
                for child in verb.children:
                    if child.dep_ not in SUBJECT_DEPS:
                        continue

                    # direct NER match
                    if child.ent_type_ in {"PERSON","ORG"}:
                        subj = child.text.lower()

                    # pronoun to coref cluster
                    elif child.lower_ in PRONOUNS:
                        subj = None
                        for cl in resolved._.coref_clusters:
                            if any(child in span for span in cl.mentions):
                                subj = cl.main.text.lower()
                                break
                        if subj is None:
                            continue


                    # fallback is any noun that matches a known_entity substring
                    else:
                        subj = child.text.lower()

                    # now see if subj maps to known_entities
                    for known in ents:
                        if known in subj or subj in known:
                            speakers.add(known)

        if speakers:
            para_map[f"paragraph {idx+1}"] = sorted(speakers)

    return para_map
