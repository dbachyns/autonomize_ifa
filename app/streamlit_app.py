"""Web UI for the form agent: ask questions, review forms, upload new ones.

    streamlit run app/streamlit_app.py

Three screens:
- Ask: chat about one form or all forms; the values an answer relies on are highlighted on the scans.
- Forms: each form's scan with every extracted value outlined by confidence, the values in an editable
  table (correct or confirm them), flags to acknowledge, a summary, and the review history.
- Upload: add a new scan or PDF; it is read, checked, and searchable a minute or two later.
"""

from __future__ import annotations

import base64
import io
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from form_agent.config import settings
from form_agent.ingest import load_pages
from form_agent.layout import source_path
from form_agent.review import apply_review
from form_agent.schema import flag_id
from form_agent.store import build_index, save_record
from form_agent.tools import FormTools
from form_agent.view import LEVELS, DocView, ViewItem, evidence_for_answer

st.set_page_config(page_title="Form Agent", page_icon="📄", layout="wide")

COLLECTIONS = {
    "Texas prior-auth forms": (settings.db_path, settings.cache_dir),
    "FUNSD business forms": (settings.data_dir / "funsd.db", settings.data_dir / "cache_funsd"),
}
LEVEL_STYLE = {  # emoji, outline colour on the scan
    "reviewed": ("🔵", (37, 99, 235)),
    "high": ("🟢", (22, 163, 74)),
    "medium": ("🟡", (202, 138, 4)),
    "low": ("🔴", (220, 38, 38)),
    "blank": ("⚪", (160, 160, 160)),
    "unknown": ("⚫", (90, 90, 90)),
}
SEVERITY = {"error": "🛑", "warning": "⚠️", "info": "ℹ️"}
HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))


# --- shared resources ------------------------------------------------------------------------


def collection() -> tuple[Path, Path]:
    return COLLECTIONS[st.session_state.get("collection", next(iter(COLLECTIONS)))]


def db_stamp(db: Path) -> float:
    return db.stat().st_mtime if db.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_tools(db: str, cache: str, stamp: float) -> FormTools:
    return FormTools(Path(db), Path(cache))


@st.cache_resource(show_spinner=False)
def get_agent(db: str, cache: str, stamp: float):
    from form_agent.agent import FormAgent

    return FormAgent(db_path=Path(db), cache_dir=Path(cache))


@st.cache_resource(show_spinner=False, max_entries=32)
def get_pages(path: str) -> list[Image.Image]:
    return [p.convert("RGB") for p in load_pages(Path(path))]


def tools() -> FormTools:
    db, cache = collection()
    if not db.exists():
        build_index(cache, db)
    return get_tools(str(db), str(cache), db_stamp(db))


def agent():
    db, cache = collection()
    return get_agent(str(db), str(cache), db_stamp(db))


def scan_pages(view: DocView) -> list[Image.Image] | None:
    path = source_path(view.record.source_file)
    return get_pages(str(path)) if path.exists() else None


def reindex() -> None:
    db, cache = collection()
    build_index(cache, db)


# --- drawing ---------------------------------------------------------------------------------


def draw(page: Image.Image, page_no: int, items: list[ViewItem], highlight: set[str] = frozenset(),
         show_all: bool = True) -> Image.Image:
    im = page.copy()
    d = ImageDraw.Draw(im, "RGBA")
    for item in items:
        hot = item.id in highlight
        if not (hot or show_all):
            continue
        color = LEVEL_STYLE[item.level][1]
        for p, (x0, y0, x1, y1) in item.boxes:
            if p != page_no:
                continue
            pad = 6
            if hot:
                d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=(255, 214, 0, 70), outline=(255, 140, 0), width=8)
            else:
                d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=color + (230,), width=4)
    return im


def crop_around(im: Image.Image, boxes: list[tuple[float, float, float, float]], margin: float = 0.06) -> Image.Image:
    """Full-width band around the boxes, so the reader sees each value with its label."""
    if not boxes:
        return im
    y0 = max(0, min(b[1] for b in boxes) - im.height * margin)
    y1 = min(im.height, max(b[3] for b in boxes) + im.height * margin)
    return im.crop((0, int(y0), im.width, int(y1)))


def show_evidence(view: DocView, items: list[ViewItem]) -> None:
    pages = scan_pages(view)
    if not pages:
        st.caption("The scan for this form isn't available locally.")
        return
    ids = {i.id for i in items}
    for page_no, page in enumerate(pages, 1):
        boxes = [b for i in items for p, b in i.boxes if p == page_no]
        if not boxes:
            continue
        im = draw(page, page_no, items, highlight=ids, show_all=False)
        st.image(crop_around(im, boxes), width="stretch",
                 caption=f"Page {page_no}: " + ", ".join(sorted({i.label.rstrip(':') for i in items if any(p == page_no for p, _ in i.boxes)})))


# --- sidebar ---------------------------------------------------------------------------------


def sidebar() -> str:
    with st.sidebar:
        st.title("📄 Form Agent")
        available = [name for name, (_, cache) in COLLECTIONS.items() if cache.exists()]
        st.selectbox("Collection", available, key="collection")
        page = st.radio("Go to", ["💬 Ask", "📋 Forms", "⬆️ Upload"], key="nav", label_visibility="collapsed")
        if not HAS_KEY:
            st.warning("No OPENAI_API_KEY in .env: browsing and reviewing work, asking and uploading don't.")
        with st.expander("Confidence colours", expanded=False):
            for level, text in LEVELS.items():
                st.markdown(f"{LEVEL_STYLE[level][0]} **{level}**: {text}")
            st.caption("Confidence is about *reading* the scan. Problems with the form's content "
                       "(e.g. an NPI that fails its check digit) are shown as flags: 🛑 ⚠️ ℹ️")
    return page


# --- Ask -------------------------------------------------------------------------------------


def page_ask() -> None:
    t = tools()
    docs = t.list_documents()
    st.header("Ask about the forms")
    scope_options = ["All forms"] + [d["id"] for d in docs]
    titles = {d["id"]: d["title"] for d in docs}
    left, right = st.columns([3, 1])
    scope = left.selectbox("Scope", scope_options, format_func=lambda s: s if s == "All forms" else f"{s} · {titles[s]}")
    key = f"chat::{st.session_state.collection}::{scope}"
    history = st.session_state.setdefault(key, [])
    if right.button("New conversation", width="stretch"):
        history.clear()

    if not history:
        examples = (["How many requests are urgent, and which are missing a clinical reason?",
                     "Which forms have an NPI that fails its check?",
                     "Who is the oldest patient?"] if "Texas" in st.session_state.collection else
                    ["What kinds of forms are in this collection?", "Which forms mention Winston?"])
        if scope != "All forms":
            examples = ["Summarize this form.", "Is anything on this form uncertain or wrong?"]
        st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))

    for turn in history:
        render_turn(turn)

    question = st.chat_input("Ask a question", disabled=not HAS_KEY)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Looking it up…"):
            doc = None if scope == "All forms" else scope
            if doc and question.strip().lower().rstrip(".") in ("summarize this form", "summarize", "summary"):
                result = agent().summarize(doc)
            else:
                result = agent().ask(question, doc_id=doc, history=[(h["q"], h["result"].answer) for h in history])
        history.append({"q": question, "result": result, "doc": doc})
        st.rerun()


def render_turn(turn: dict) -> None:
    result = turn["result"]
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "-"
        st.caption(f"{result.model} · {len(result.steps)} tool calls · {result.seconds:.1f}s · {cost}")
        t = tools()
        cited = list(dict.fromkeys(re.findall(r"\[([\w\-.]+)\]", result.answer)))
        known = {short for short in t.short.values()}
        cited = [c for c in cited if c in known] or ([turn["doc"]] if turn.get("doc") else [])
        if cited:
            with st.expander(f"Where this comes from on the scans ({len(cited)} form{'s' if len(cited) > 1 else ''})"):
                tabs = st.tabs(cited[:6])
                for tab, short in zip(tabs, cited[:6]):
                    with tab:
                        view = t.view(short)
                        items = evidence_for_answer(view, result.answer)
                        if items:
                            show_evidence(view, items)
                        else:
                            st.caption("No specific value from this form is quoted in the answer.")
                        st.button("Open this form", key=f"open::{id(turn)}::{short}", on_click=open_form, args=(short,))
        if result.steps:
            with st.expander("How the agent found this"):
                st.code(result.trace(), language=None)


def open_form(short: str) -> None:
    st.session_state.selected_doc = short
    st.session_state.nav = "📋 Forms"


# --- Forms -----------------------------------------------------------------------------------


def page_forms() -> None:
    t = tools()
    docs = t.list_documents()
    st.header("Forms")
    rows = []
    for d in docs:
        view = t.view(d["id"])
        levels = [i.level for i in view.items]
        open_flags = [f for f in view.flags if not view.acknowledged(f)]
        rows.append({"Form": d["id"], "Title": d["title"], "Type": d["form_type"],
                     "Needs attention": sum(needs_attention(view, i) for i in view.items),
                     "🛑 Errors": sum(f.severity == "error" for f in open_flags),
                     "⚠️ Warnings": sum(f.severity == "warning" for f in open_flags),
                     "🔴 Disagrees": levels.count("low"), "🟡 Unconfirmed": levels.count("medium"),
                     "Reviewed": "✓" if view.record.review else ""})
    st.caption("**Needs attention** = values whose reading isn't confirmed (🔴 evidence disagrees, 🟡 not confirmed) "
               "plus values with an error or warning nobody has marked as seen. Flags marked as seen aren't counted.")
    table = pd.DataFrame(rows)
    selection = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="forms_table",
                             height=min(38 * (len(rows) + 1), 320))
    picked = selection.selection.rows if selection and selection.selection else []
    if picked:
        st.session_state.selected_doc = rows[picked[0]]["Form"]
    short = st.session_state.get("selected_doc")
    if short not in t.short.values():
        st.info("Select a form above.")
        return
    form_detail(t, short)


def form_detail(t: FormTools, short: str) -> None:
    view = t.view(short)
    rec = view.record
    st.subheader(f"{short} · {rec.universal.title or 'Untitled form'}")
    run = rec.run
    st.caption(f"Type: {rec.form_type} · read by {run.reader} ({run.model}) · "
               + (f"${run.cost_usd:.4f} · " if run.cost_usd else "") + (f"{run.seconds:.0f}s" if run.seconds else ""))

    attention = [i for i in view.items if needs_attention(view, i)]
    shown_all = [i for i in view.items if i.kind != "text" or i.value]
    filter_key = f"filter::{short}"
    choice = st.segmented_control(
        "Show", ["all", "attention"], key=filter_key, default="all", label_visibility="collapsed",
        format_func=lambda c: f"All values ({len(shown_all)})" if c == "all" else f"Needs attention ({len(attention)})")
    items = attention if choice == "attention" else shown_all
    st.caption("Hover a value to find it on the scan (or a box to find its value) · click to review it on the right · "
               "outline colour = how well the reading is confirmed (see sidebar) · badge = a problem with the content "
               "(🛑 error, ⚠️ warning, ℹ️ note)")

    sel_key = f"sel::{st.session_state.collection}::{short}"
    viewer_col, panel_col = st.columns([3, 1.1], gap="medium")
    with viewer_col:
        pages = page_images(str(source_path(rec.source_file))) if source_path(rec.source_file).exists() else []
        if not pages:
            st.caption("The scan isn't available locally (FUNSD scans: run scripts/download_funsd.sh); values only.")
        review_stamp = rec.review.updated_at.isoformat() if rec.review else ""
        clicked = form_viewer(
            doc=short, signature=f"{choice}|{review_stamp}", pages=pages, height=780,
            items=[viewer_item(view, i) for i in items], selected=st.session_state.get(sel_key),
            key=f"viewer::{st.session_state.collection}::{short}", default=None)
        if clicked and clicked.get("n") != st.session_state.get(f"n::{sel_key}"):
            st.session_state[f"n::{sel_key}"] = clicked["n"]
            st.session_state[sel_key] = clicked["id"]
            st.rerun()
    with panel_col:
        item_panel(view, short, sel_key, attention)

    tab_flags, tab_summary, tab_history = st.tabs([f"All flags ({len(rec.flags)})", "Summary", "Review history"])
    with tab_flags:
        flags_panel(view, short)
    with tab_summary:
        summary_panel(short)
    with tab_history:
        history_panel(view)


_form_viewer = components.declare_component("form_viewer", path=str(Path(__file__).parent / "components" / "form_viewer"))


def form_viewer(**kwargs):
    """Scan + value list with hover highlighting (app/components/form_viewer). Returns the last clicked item."""
    return _form_viewer(**kwargs)


@st.cache_resource(show_spinner=False, max_entries=32)
def page_images(path: str, max_width: int = 1400) -> list[dict]:
    """Pages as JPEG data URLs for the viewer; box coordinates stay in full-resolution pixels."""
    out = []
    for page in get_pages(path):
        small = page.copy()
        small.thumbnail((max_width, max_width * 2))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=82)
        out.append({"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "w": page.width, "h": page.height})
    return out


def needs_attention(view: DocView, item: ViewItem) -> bool:
    """Reading not confirmed, or a content problem nobody has looked at yet."""
    if item.level in ("low", "medium", "unknown"):
        return True
    return any(f.severity in ("error", "warning") and not view.acknowledged(f) for f in item.issues)


def viewer_item(view: DocView, item: ViewItem) -> dict:
    return {
        "id": item.id, "label": short_label(item), "value": item.value or "", "level": item.level,
        "section": (item.section or "").split("—")[-1].strip() or None, "reading": item.reading,
        "boxes": [[p, *b] for p, b in item.boxes],
        "issues": [{"severity": f.severity, "message": f.message, "ack": view.acknowledged(f)} for f in item.issues],
    }


def short_label(item: ViewItem) -> str:
    """'Services Requested · row 2 · Code' -> 'Row 2 · Code' (the section header already names the table)."""
    label = item.label.rstrip(":")
    if item.kind == "cell":
        rest = " · ".join(label.split(" · ")[1:])
        label = rest[:1].upper() + rest[1:]
    return label


def item_panel(view: DocView, short: str, sel_key: str, attention: list[ViewItem]) -> None:
    ids = [i.id for i in attention]
    sel = st.session_state.get(sel_key)
    if sel not in {i.id for i in view.items}:
        sel = None

    def go(step: int) -> None:
        if not ids:
            return
        if sel in ids:
            nxt = ids[(ids.index(sel) + step) % len(ids)]
        else:
            nxt = ids[0] if step > 0 else ids[-1]
        st.session_state[sel_key] = nxt

    st.markdown(f"**{len(attention)}** value{'s' if len(attention) != 1 else ''} need attention")
    b1, b2 = st.columns(2)
    b1.button("◀ Previous", on_click=go, args=(-1,), disabled=not ids, width="stretch", key=f"prev::{short}")
    b2.button("Next ▶", on_click=go, args=(1,), disabled=not ids, width="stretch", key=f"next::{short}",
              type="primary" if sel is None else "secondary")

    if sel is None:
        st.info("Hover a value to find it on the scan. Click one, or press **Next**, to review it here.")
        green = [i.id for i in view.items if i.level == "high"]
        if green and st.button(f"✓ Confirm all {len(green)} 🟢 values", help="Mark every value whose reading local "
                               "evidence confirms as checked by you", key=f"bulk::{short}"):
            save_review(view, confirm=green, reviewer=st.session_state.get("reviewer") or None)
        return

    item = view.item(sel)
    st.divider()
    st.markdown(f"#### {short_label(item)}")
    if item.section:
        st.caption(item.section)
    st.markdown(f"{LEVEL_STYLE[item.level][0]} **Reading:** {item.reading}")
    for f in item.issues:
        seen = view.acknowledged(f)
        st.markdown(f"{SEVERITY[f.severity]} **{'Seen' if seen else 'Issue'}:** {f.message}")

    if not item.editable:
        st.caption("This value can't be edited.")
        return
    with st.form(key=f"form::{short}::{sel}", border=False):
        if item.kind == "checkbox":
            options = [o for o, _ in item.options]
            new = st.multiselect("Ticked boxes", options, default=[o for o, c in item.options if c])
        else:
            new = st.text_area("Value", value=item.value or "", height=68 if len(item.value or "") < 60 else 140)
        c1, c2 = st.columns(2)
        save = c1.form_submit_button("Save correction", type="primary", width="stretch")
        confirm = c2.form_submit_button("✓ Looks right", width="stretch")
    reviewer = st.session_state.get("reviewer") or None
    if save:
        old = [o for o, c in item.options if c] if item.kind == "checkbox" else (item.value or "")
        changed = sorted(new) != sorted(old) if item.kind == "checkbox" else new.strip() != old
        if changed:
            go(1)
            save_review(view, edits={sel: new if item.kind == "checkbox" else (new.strip() or None)}, reviewer=reviewer)
        else:
            st.info("Nothing changed. Use **✓ Looks right** to confirm the value as is.")
    if confirm:
        go(1)
        unacked = [flag_id(f) for f in item.issues if not view.acknowledged(f)]
        save_review(view, confirm=[sel], acknowledge=unacked, reviewer=reviewer)
    st.caption("✓ Looks right also marks this value's issues as seen.")
    st.text_input("Reviewer name", key="reviewer", placeholder="optional")


def save_review(view: DocView, **kwargs) -> None:
    _, cache = collection()
    save_record(cache, apply_review(view.record, **kwargs))
    reindex()
    st.toast("Review saved. The agent now answers from the reviewed values.", icon="✅")
    st.rerun()


def flags_panel(view: DocView, short: str) -> None:
    if not view.flags:
        st.success("No flags on this form.")
        return
    to_ack = []
    for f in view.flags:
        ack = view.acknowledged(f)
        cols = st.columns([12, 2])
        cols[0].markdown(f"{SEVERITY[f.severity]} **{f.rule}**: {f.message}" + ("  \n✓ *acknowledged by a reviewer*" if ack else ""))
        if not ack and cols[1].checkbox("Seen", key=f"ack::{short}::{flag_id(f)}"):
            to_ack.append(flag_id(f))
    if st.button(f"Acknowledge {len(to_ack)} flag{'s' if len(to_ack) != 1 else ''}", disabled=not to_ack,
                 key=f"ackbtn::{short}"):
        save_review(view, acknowledge=to_ack, reviewer=st.session_state.get("reviewer") or None)


def summary_panel(short: str) -> None:
    key = f"summary::{st.session_state.collection}::{short}"
    if st.button("Write summary" if key not in st.session_state else "Rewrite summary", disabled=not HAS_KEY,
                 key=f"sumbtn::{short}"):
        with st.spinner("Summarizing…"):
            st.session_state[key] = agent().summarize(short)
    if key in st.session_state:
        result = st.session_state[key]
        st.markdown(result.answer)
        st.caption(f"{result.model} · {result.seconds:.1f}s · ${result.cost_usd or 0:.4f}")


def history_panel(view: DocView) -> None:
    review = view.record.review
    if not review:
        st.caption("Not reviewed yet. Edit or confirm values in the Fields tab.")
        return
    st.markdown(f"Reviewed version based on the **{review.based_on}** reading · last saved "
                f"{review.updated_at:%Y-%m-%d %H:%M} UTC · {len(review.confirmed)} value(s) confirmed · "
                f"{len(review.acknowledged)} flag(s) acknowledged")
    if review.corrections:
        st.dataframe(pd.DataFrame([{"Field": c.label, "Was": _fmt(c.old), "Now": _fmt(c.new),
                                    "By": c.reviewer or "", "When": f"{c.at:%Y-%m-%d %H:%M}"}
                                   for c in review.corrections]), hide_index=True)


def _fmt(value) -> str:
    if isinstance(value, list):
        return ", ".join(value) or "(none)"
    return "(blank)" if value in (None, "") else str(value)


# --- Upload ----------------------------------------------------------------------------------


def page_upload() -> None:
    from form_agent.pipeline import SUPPORTED, process_file

    st.header("Upload a form")
    st.markdown("Any form works: a known type gets its strict schema and rules, anything else the generic record. "
                "Reading takes a minute or two (vision model + local checks), then the form is searchable.")
    file = st.file_uploader("Scan or PDF", type=sorted(s.lstrip(".") for s in SUPPORTED))
    reader = st.radio("Reader", ["hybrid", "openai"], horizontal=True,
                      captions=["vision model + pixel checkbox check + OCR cross-check (recommended)",
                                "vision model only"])
    if st.button("Read this form", type="primary", disabled=not (file and HAS_KEY)):
        db, cache = collection()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / file.name
            path.write_bytes(file.getvalue())
            with st.status("Processing…", expanded=True) as status:
                try:
                    record = process_file(path, reader=reader, cache_dir=cache, db_path=db, on_step=st.write)
                except Exception as exc:  # show the reason in the UI instead of a stack trace
                    status.update(label="Failed", state="error")
                    st.error(f"Could not read the form: {exc}")
                    return
                status.update(label="Done", state="complete")
        st.session_state.uploaded = record.doc_id

    if doc := st.session_state.get("uploaded"):
        t = tools()
        short = t.short.get(doc, doc)
        view = t.view(short)
        attention = sum(needs_attention(view, i) for i in view.items)
        st.success(f"**{short}** is ready: {view.record.form_type} · {sum(1 for i in view.items if i.value)} values · "
                   f"{len(view.flags)} flags · {attention} need attention")
        st.button("Open it", type="primary", on_click=open_form, args=(short,))


# --- main ------------------------------------------------------------------------------------

page = sidebar()
if page.endswith("Ask"):
    page_ask()
elif page.endswith("Forms"):
    page_forms()
else:
    page_upload()
