def collect_aligned_entities(doc, vic_tokens, off_tokens):
    vic_set, off_set = set(), set()

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            span_tokens = set(ent)
            if span_tokens & vic_tokens:
                vic_set.add(ent.text.lower())
            elif span_tokens & off_tokens:
                off_set.add(ent.text.lower())

    # Generic nouns within the token sets (e.g., "victim")
    def maybe_add_generics(tok_set, bucket):
        for t in tok_set:
            if (t.pos_ in ("NOUN", "PROPN")) and (t.ent_type_ == ""):
                s = "".join(ch for ch in t.text if ch.isalnum() or ch == "-").lower()
                if s:
                    bucket.add(s)

    maybe_add_generics(vic_tokens, vic_set)
    maybe_add_generics(off_tokens, off_set)

    return sorted(vic_set), sorted(off_set)


def process_article(sent_list, ff, name, gender="ignore_gender", race="ignore_race"):
    raw_text = "\n".join(sent_list)
    text = ff.coref_preprocess(raw_text)
    doc = ff.nlp(text)
    off_tokens, vic_tokens = ff.partition_tokens(doc, name, gender, race, verbose=False)

    vic_ents, off_ents = collect_aligned_entities(doc, vic_tokens, off_tokens)

    victim_tok_strings  = [t.text.lower() for t in vic_tokens]
    police_tok_strings  = [t.text.lower() for t in off_tokens]

    return {
         "task1": {
            "Victim-aligned": victim_tok_strings,
            "Police-aligned": police_tok_strings
        }

    # this is a slight extension of the zy2021 to get better results
    # return {
    #     "task1": {
    #         "Victim-aligned": vic_ents,
    #         "Police-aligned": off_ents
    #     }
    # }

    }
