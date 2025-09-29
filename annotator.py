# annotate_simple.py
# pip install streamlit streamlit-text-label

import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Iterable, List
import streamlit as st
from streamlit_text_label import label_select, Selection

st.set_page_config(page_title="Article Annotator", layout="wide")

def order_paragraphs(article: Dict[str, str]) -> OrderedDict:
    try:
        return OrderedDict(sorted(article.items(), key=lambda kv: int(kv[0].split()[-1])))
    except Exception:
        return OrderedDict(article.items())

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        x2 = x.strip()
        if x2 and x2 not in seen:
            seen.add(x2); out.append(x2)
    return out

def iter_spans(result: Any, body: str) -> Iterable[Dict[str, Any]]:
    if result is None:
        return
    items: List[Any] = []
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        items = result.get("spans") or result.get("labels") or [result]
    else:
        items = [result]
    for sp in items:
        if isinstance(sp, dict):
            start = int(sp["start"]); end = int(sp["end"])
            labels = sp.get("labels") or ([sp.get("label")] if sp.get("label") else [])
        else:
            start = int(getattr(sp, "start")); end = int(getattr(sp, "end"))
            labels = list(getattr(sp, "labels", [])) or ([getattr(sp, "label")] if getattr(sp, "label", None) else [])
        yield {"start": start, "end": end, "text": body[start:end], "labels": labels}

def selections_from_task1(task1: Dict[str, List[str]], body: str) -> List[Selection]:
    sels, cursor = [], 0
    for lab, items in task1.items():
        for text in items:
            if not text:
                continue
            idx = body.find(text, cursor)
            if idx == -1:
                # fall back to case-insensitive match
                low_idx = body.lower().find(text.lower())
                if low_idx == -1:
                    continue
                idx = low_idx
            end = idx + len(text)
            sels.append(Selection(start=idx, end=end, text=body[idx:end], labels=[lab]))
            cursor = end
    return sels

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def changed(a, b) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) != json.dumps(b, sort_keys=True, ensure_ascii=False)

def first_or_none(seq):
    return seq[0] if (isinstance(seq, list) and len(seq) > 0) else None

def normalize_task2_sparse(task2_any, valid_keys: set, allowed_entities: set) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if isinstance(task2_any, dict):
        for k, v in task2_any.items():
            if k not in valid_keys:
                continue
            ent = first_or_none(v)
            if ent in allowed_entities:
                out[k] = [ent]
    return out

st.sidebar.header("Upload JSON")
work_dir = Path("./annotations")
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "doc" not in st.session_state:
    st.session_state.doc = None

st.session_state.autosave = True

up = st.sidebar.file_uploader("Upload .json", type=["json"])
if up:
    raw = up.read()
    staged = work_dir / up.name
    work_dir.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(raw)
    st.session_state.file_path = staged
    st.session_state.doc = json.loads(raw.decode("utf-8"))
    st.sidebar.success(f"Loaded → {staged}")

if st.sidebar.button("Clear"):
    st.session_state.clear()

if not st.session_state.doc:
    st.info("Upload a JSON with an 'article' object to begin.")
    st.stop()


doc = st.session_state.doc
doc.setdefault("task1", {"Victim-aligned": [], "Police-aligned": []})
doc.setdefault("task2", {})  # sparse storage

st.title("Task 1: classify entities")
st.caption("Pick a label, highlight text, then press Update inside the widget.")

article = order_paragraphs(doc.get("article", {}))
body = "\n\n".join(article.values())
labels = ["Victim-aligned", "Police-aligned"]

initial_selections = selections_from_task1(doc.get("task1", {}), body)
result = label_select(body=body, labels=labels, selections=initial_selections)

before_t1 = json.loads(json.dumps(doc["task1"], ensure_ascii=False))
victim, police = [], []
for sp in iter_spans(result, body):
    frag = sp["text"].strip()
    labs = sp["labels"] or []
    if not frag:
        continue
    if "Victim-aligned" in labs:
        victim.append(frag)
    if "Police-aligned" in labs:
        police.append(frag)
raw_task1 = {
    "Victim-aligned": dedupe_keep_order(victim),
    "Police-aligned": dedupe_keep_order(police),
}

st.subheader("Rename entities (saved lowercase)")

current_entities = dedupe_keep_order(raw_task1["Victim-aligned"] + raw_task1["Police-aligned"])

for ent in current_entities:
    k = f"canon::{ent}"
    if k not in st.session_state:
        st.session_state[k] = ent.lower()

cols = st.columns(2)
for i, ent in enumerate(current_entities):
    with cols[i % 2]:
        st.text_input(f"'{ent}' →", key=f"canon::{ent}", help="Saved value will be lowercase.")

canon_map = {ent: str(st.session_state.get(f"canon::{ent}", ent)).strip().lower() for ent in current_entities}
doc["task1"] = {
    "Victim-aligned": dedupe_keep_order([canon_map.get(x, x.lower()) for x in raw_task1["Victim-aligned"]]),
    "Police-aligned":  dedupe_keep_order([canon_map.get(x, x.lower()) for x in raw_task1["Police-aligned"]]),
}

c1, c2 = st.columns(2)
with c1:
    st.metric("Victim-aligned", len(doc["task1"]["Victim-aligned"]))
    st.write(", ".join(doc["task1"]["Victim-aligned"]) or "(none)")
with c2:
    st.metric("Police-aligned", len(doc["task1"]["Police-aligned"]))
    st.write(", ".join(doc["task1"]["Police-aligned"]) or "(none)")

all_entities = dedupe_keep_order(doc["task1"]["Victim-aligned"] + doc["task1"]["Police-aligned"])
valid_keys = set(article.keys())
doc["task2"] = normalize_task2_sparse(doc.get("task2", {}), valid_keys, set(all_entities))

if st.session_state.autosave and changed(before_t1, doc["task1"]) and st.session_state.file_path:
    save_json(st.session_state.file_path, doc)
    st.success("Autosaved Task 1")

st.divider()

st.header("Task 2: assign paragraph perspectives")
st.caption("Choose at most one entity for each paragraph. Leave as None if no perspective.")

if not all_entities:
    st.info("No entities from Task 1 yet.")
else:
    pill_options = ["none"] + all_entities
    before_t2 = json.loads(json.dumps(doc["task2"], ensure_ascii=False))
    new_t2: Dict[str, List[str]] = {}

    for pkey, ptext in article.items():
        st.markdown(f"### {pkey}")
        st.write(ptext)

        pill_key = f"t2::{pkey}"
        saved_choice = first_or_none(doc["task2"].get(pkey, []))
        initial = saved_choice if (saved_choice in all_entities) else "none"

        _ = st.pills(
            "Perspective",
            options=pill_options,
            selection_mode="single",
            default=initial,
            key=pill_key,
            label_visibility="collapsed",
            width="content",
        )

        current = st.session_state.get(pill_key, initial)
        if current in all_entities:
            new_t2[pkey] = [current]  # sparse keep
        st.divider()

    if changed(before_t2, new_t2):
        doc["task2"] = new_t2
        if st.session_state.file_path:
            save_json(st.session_state.file_path, doc)
            st.success("Saved Task 2")

st.divider()

st.subheader("JSON preview")
st.code(json.dumps(doc, indent=2, ensure_ascii=False), language="json")
