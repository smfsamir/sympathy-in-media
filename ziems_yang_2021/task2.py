from collections import defaultdict

SAY_VERBS = {
    "say",
    "tell",
    "explain",
    "report",
    "answer",
    "claim",
    "declare",
    "reply",
    "state",
    "confirm",
}
SUBJECT_DEPS = {"nsubj", "nsubjpass"}


def build_text_and_spans(paragraphs):
    text = ""
    spans = []  # (start,end)
    cursor = 0
    for p in paragraphs:
        if text:
            text += "\n\n"
            cursor += 2
        start = cursor
        text += p
        cursor += len(p)
        spans.append((start, cursor))
    return text, spans


def make_name_token_sets(doc, names):
    # map each lowercased name to set of tokens (by simple token lowercase overlap and coref)
    name2tok = {n.lower(): set() for n in names}

    # literal matches
    for name in name2tok:
        words = set(name.split())
        for tok in doc:
            if tok.text.lower() in words:
                name2tok[name].add(tok)

    # coref expansion: if any token in a cluster is in the set, take the whole cluster
    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
        for cl in doc._.coref_clusters:
            cluster_toks = {t for span in cl for t in span}
            for name in name2tok:
                if cluster_toks & name2tok[name]:
                    name2tok[name] |= cluster_toks
    return name2tok


def token_to_name(tok, name2tok):
    # pick the first name whose token set contains this token
    for name, toks in name2tok.items():
        if tok in toks:
            return name
    return None


def paragraph_index(char_idx, spans):
    for i, (s, e) in enumerate(spans, start=1):
        if s <= char_idx < e:
            return i
    return None


def find_perspectives(doc, spans, name2tok):
    para2names = defaultdict(set)

    for tok in doc:  # replicate zy2021 trigger
        try:
            cond1 = tok.head.head.lower_ == "according"
        except:
            cond1 = False
        cond2 = tok.head.lemma_ in SAY_VERBS and tok.dep_ in SUBJECT_DEPS

        if not (cond1 or cond2):
            continue

        # Map subject token -> entity name
        name = token_to_name(tok, name2tok)
        if not name:
            continue  # skip if we can't map to a provided entity

        char_idx = tok.head.idx  # they return head idx; good enough
        para_id = paragraph_index(char_idx, spans)
        if para_id is not None:
            para2names[f"paragraph {para_id}"].add(name)

    return {p: sorted(list(v)) for p, v in para2names.items()}


def process_article(paragraphs, ff, all_names, preprocess=True):
    raw_text, spans = build_text_and_spans(paragraphs)
    text_for_nlp = ff.coref_preprocess(raw_text) if preprocess else raw_text
    doc = ff.nlp(text_for_nlp)
    name2tok = make_name_token_sets(doc, all_names)
    task2 = find_perspectives(doc, spans, name2tok)
    return {"task2": task2}
