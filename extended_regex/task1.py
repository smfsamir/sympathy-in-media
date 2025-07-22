import re
import spacy
import neuralcoref
from unidecode import unidecode
import pandas as pd

nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)

PEOPLE_TERMS_FILE = "./resources/textbook_analysis/people_terms.csv"
PB_WORDS_FILE = "./regex/pb.txt"
VA_WORDS_FILE = "./regex/va.txt"

def build_context_regex(terms):
    escaped = [re.escape(term) for term in terms]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def preprocess_article_with_unidecode(article_list):
    cleaned = []
    for paragraph in article_list:
        paragraph = paragraph.strip()
        if not paragraph or paragraph[-1] in ".!?":
            cleaned.append(paragraph)
        else:
            cleaned.append(paragraph + ".")
    full_text = " ".join(cleaned)
    full_text = unidecode(full_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text

def extract_and_classify_entities(text, victim_name):
    doc = nlp(text)

    victim_aligned = set()
    police_aligned = set()

    escaped = re.escape(victim_name)

    POLICE_PATTERNS = [
        re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+police\b'),
        re.compile(r'\bbureau of investigators\b', re.IGNORECASE),
        re.compile(r'\bbei\b', re.IGNORECASE),
        re.compile(r'\bspecial investigations unit\b', re.IGNORECASE),
        re.compile(r'\bsiu\b', re.IGNORECASE),
        re.compile(r'\brcmp\b', re.IGNORECASE),
        re.compile(r'\bopp\b', re.IGNORECASE),
        re.compile(r'\bcsis\b', re.IGNORECASE),
        re.compile(r'\bindependent investigation unit(?: of manitoba)?\b', re.IGNORECASE),
        re.compile(r'\bPolice Nationale(?: d\'[A-Z][a-z]+)?\b', re.IGNORECASE),
        re.compile(r'\b(?:officer|detective|chief|const|lieutenant|constable|superintendent|spokesperson|spokesman|spokeswoman|sergeant|acting|staff|supt)\b', re.IGNORECASE),
    ]

    STRICT_POLICE_ORGS = POLICE_PATTERNS[:8] + [re.compile(r'\bjustice ministry\b', re.IGNORECASE)]

    VA_PATTERNS = [
        re.compile(rf"\b{escaped}\b", re.IGNORECASE),
        re.compile(rf"\b{escaped}(?:[’']s)\s+mother\b", re.IGNORECASE),
        re.compile(r"\bold\s+brewery\s+mission\b", re.IGNORECASE),
        re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'s (daughter|son|neighbour)\b", re.IGNORECASE),
    ]

    TITLE_WORDS = {
        "officer", "detective", "chief", "const", "const.", "lieutenant", "constable",
        "coroner", "superintendent", "spokesperson", "spokesman", "spokeswoman",
        "sergeant", "acting", "staff", "supt", "chief supt"
    }

    filtered_spans = []
    for span in list(doc.ents) + list(doc.noun_chunks):
        text = span.text.strip().lower()

        if text in TITLE_WORDS:
            continue
        if span[-1].tag_ == "POS":
            continue
        if any(part in text.split() for part in ['st', 'ave', 'rd', 'rue', 'blvd', 'avenue']):
            continue
        if any(ent.label_ in {'FAC', 'LOC', 'DATE', 'TIME', 'ORDINAL', 'CARDINAL', 'NORP'} for ent in span.ents):
            continue
        if span[0].pos_ in {"DET", "SCONJ", "PRON", "ADP", "CCONJ"} or span[0].dep_ == 'advmod':
            continue

        filtered_spans.append(span)

    unique_spans = []
    seen = set()
    for span in filtered_spans:
        key = span.text.lower()
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    RCMP_RE = re.compile(r'\brcmp\b', re.IGNORECASE)
    OPP_RE = re.compile(r'\bopp\b', re.IGNORECASE)

    for span in unique_spans:
        entity_text = span.text
        lower_text = entity_text.lower()

        if RCMP_RE.search(lower_text):
            police_aligned.add("rcmp")
            continue
        if OPP_RE.search(lower_text):
            police_aligned.add("opp")
            continue
        if any(p.fullmatch(entity_text) for p in STRICT_POLICE_ORGS):
            police_aligned.add(lower_text)
            continue
        if any(p.search(entity_text) for p in POLICE_PATTERNS):
            candidates = [ent for ent in doc.ents if ent.label_ == "PERSON" and span.start <= ent.start <= ent.end <= span.end]
            police_aligned.add((candidates[0].text if candidates else entity_text).lower())
            continue
        if any(p.search(entity_text) for p in VA_PATTERNS):
            name = entity_text
            if len(span) > 3:
                candidates = [ent for ent in doc.ents if ent.label_ == "PERSON" and span.start <= ent.start <= ent.end <= span.end]
                if candidates and candidates[0][-1].tag_ != "POS":
                    name = candidates[0].text
            victim_aligned.add(name.lower())
            continue

    if not victim_aligned or victim_name.lower() not in victim_aligned:
        victim_aligned.add("victim")

    return sorted(victim_aligned), sorted(police_aligned)
