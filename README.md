# Intelligent Form Agent — read, extract, and explain

An agent that reads scanned forms, extracts what was filled in, and answers questions about them: about one
form, as a summary, or across a whole collection. Every answer cites the forms it came from and points out
anything a human should double-check.

```text
$ python -m form_agent ask "How many requests are urgent, and which are missing a clinical reason?"

There are 5 urgent requests, and all five are missing a clinical reason for urgency: [00c33de5],
[0a18d138], [0a9d92f7], [0adc26e6], and [0adf81aa].

[gpt-6-luna · 2 tool calls · 6.9s · $0.0007]
```

**Results** (details in [docs/evaluation.md](docs/evaluation.md)):

| | |
|---|---|
| Extraction, 11 Texas prior-auth forms vs. hand-labeled truth | **99.9%** of 695 fields correct · 100% of checkboxes · the one remaining error is flagged for review |
| Extraction, 50 FUNSD forms of types it was never tuned on | **86.0%** of labeled label→value pairs found |
| Q&A agent, 30 questions with known answers | **30/30** (13 single-form, 14 cross-form, 3 summaries) |
| Reading confidence on the Texas forms | 565 values confirmed, 5 unconfirmed, **1 red**: exactly the one real extraction error |
| Cost with `gpt-6-luna` | ~$0.01 to read a form · ~$0.001 per question |

## Quick start (about 5 minutes)

The repo ships with the extraction results for every form, so most of it runs **without an API key**.

```bash
git clone <this repo> && cd autonomize_ifa
python3.12 -m venv .venv && source .venv/bin/activate    # or: conda env create -f environment.yml && conda activate ifa
pip install -r requirements.txt
```

**No key needed:**

```bash
pytest                                        # 126 tests, ~30 s, no API calls
python -m form_agent eval tx --reader hybrid  # extraction vs. hand-labeled truth: 99.9% of 695 fields
streamlit run app/streamlit_app.py            # web UI → 📋 Forms: scans, confidence colours, review
```

**With an OpenAI key** (`cp .env.example .env`, then set `OPENAI_API_KEY`; about $0.001 per question):

```bash
python -m form_agent ask "Which urgent requests are missing a clinical reason?" --trace   # across all forms
python -m form_agent ask "Is the service provider's NPI valid?" --doc 0ab19462            # one form
python -m form_agent summarize 0a18d138                                                   # summary
python -m form_agent chat                                                                 # follow-up questions
python -m form_agent eval qa                                                              # 30 graded questions, ~$0.02
python -m form_agent extract path/to/your/forms   # read your own PDFs/images (~$0.01 each), then ask about them
```

The search index (`data/forms.db`) is built automatically on first use. In the web UI, the sidebar also has
the FUNSD collection; run `scripts/download_funsd.sh` once to see its scans.

## What it does

1. **Reads any form.** A vision LLM transcribes every labeled field (blank ones too), checkbox group, table
   and free-text note into one *universal* record, page by page. Pre-printed values (fees, fax numbers) are
   captured and tagged `printed`, so they are never confused with what the person filled in.
2. **Knows some forms better.** When it recognizes a known form type (here, the *Texas Standard Prior
   Authorization Request*), a second pass fills a strict typed schema, and type-specific rules apply.
   New form types are plugins.
3. **Checks its own work.** OpenCV measures the ink inside every checkbox and overrides the model where the
   pixels are decisive. Local OCR re-reads the page and flags values the model may have "corrected".
   Validation rules flag bad NPIs, code formats, impossible dates, double-ticked pick-one boxes, and more.
4. **Answers questions.** An LLM agent with read-only tools (`get_document`, `search`, `sql`,
   `list_documents`) answers questions about one form or across all of them. It writes summaries that end
   with a *Needs review* list. Counts come from SQL, so they are exact. Follow-up questions keep the context.
5. **Shows its evidence and takes corrections** (web UI). Every value is located on the scan and colored by
   how well local evidence confirms it. Answers highlight the values they quote on the scan. A reviewer can
   correct or confirm values and acknowledge flags. The reviewed version is saved separately, with a log,
   and the agent answers from it.

## Setup

Requirements: Python 3.12 and an OpenAI API key (only for reading new forms and for asking questions;
browsing the cached results needs no key).

```bash
git clone <this repo> && cd autonomize_ifa

# conda …
conda env create -f environment.yml && conda activate ifa
# … or venv
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

cp .env.example .env          # then put your OPENAI_API_KEY in .env
```

`.env` settings (all optional except the key): `OPENAI_MODEL` (reader, default `gpt-6-luna`),
`FORM_AGENT_MODEL` (Q&A agent, defaults to `OPENAI_MODEL`), `FORM_AGENT_READER` (default `openai`).

## Web UI

```bash
streamlit run app/streamlit_app.py
```

| Screen | What you can do |
|---|---|
| **💬 Ask** | Chat about all forms or one form, with follow-ups. Under each answer, *Where this comes from* shows the scans with the quoted values highlighted. *How the agent found this* lists its tool calls. |
| **📋 Forms** | Every form with error, warning and uncertain counts. Open one to get the scan next to its list of values. **Hover** a value to zoom the scan to it, or hover a box on the scan to find its value. The outline color shows how well the reading is confirmed: 🟢 confirmed by local OCR or checkbox ink, 🟡 unconfirmed, 🔴 evidence disagrees, 🔵 reviewed. A badge marks a problem with the content (🛑 ⚠️ ℹ️, e.g. an invalid NPI). **Needs attention** filters both the scan and the list. **Click** a value, or press **Next**, to review it on the right: *Save correction*, or *✓ Looks right*. Also on this screen: all flags, a one-click summary, and the review history. |
| **⬆️ Upload** | Drop in a PDF or image of any form. It is read, verified, validated, located on the page and indexed, then it opens in *Forms*. |

The sidebar switches between the Texas collection and the FUNSD collection. Reviews are saved as
`data/cache/<doc_id>/reviewed.json`, next to the machine readings, which are never modified.

## Run it step by step

The repo ships with the extraction results for every form in `data/cache/`, so steps 1–2 are optional:
you can go straight to step 3.

```bash
# 1. Read forms (writes one JSON record per form to data/cache/<doc_id>/<reader>.json), then locate the
#    values on the scans and rebuild the index. hybrid = vision LLM + pixel checkbox check + local OCR
#    cross-check; already-cached forms are skipped.
python -m form_agent extract data/forms --reader hybrid            # add --force to re-read

# 2. Only needed after changing code: recompute where values are on the scans (local OCR, no API) and
#    rebuild the SQLite index the agent queries from the JSON records (seconds).
python -m form_agent layout --force
python -m form_agent index

# 3. Ask questions.
python -m form_agent ask "What is the patient's member ID?" --doc 0a01f77b      # one form
python -m form_agent summarize 0a18d138                                          # summary of one form
python -m form_agent ask "Which forms have an NPI with the wrong number of digits?" --trace   # all forms
python -m form_agent chat                                                        # conversation with follow-ups

# 4. Evaluate.
python -m form_agent eval tx --reader hybrid      # extraction vs. hand-labeled ground truth
python -m form_agent eval qa                      # the agent on 30 questions with known answers
pytest                                            # 126 unit tests, no API calls
```

Document ids can be shortened to their first 8 characters. `--trace` prints the tool calls the agent made.

**The demo notebook**, [notebooks/demo.ipynb](notebooks/demo.ipynb), runs all of this with the outputs saved:
the extraction and checkbox check on one scan, then the required demos below, then the evaluation tables.

**Your own forms:** upload them in the web UI, or put PDFs/images in a folder and run `extract <folder>`, then
`ask`. Forms of an unknown type go through the universal record; nothing else is needed.

## Demo runs

Real outputs (`gpt-6-luna`), also in the notebook.

### 1 · Question about one form

```text
$ python -m form_agent ask "Is the service provider's NPI valid?" --doc 0ab19462

No. The service provider's NPI is recorded as "7660952," and it has only 7 digits; the form is flagged
because NPIs have 10 digits. [0ab19462]
```

```text
$ python -m form_agent ask "What is the clinical reason for urgency?" --doc 00c33de5

The clinical reason for urgency was left blank. The form is marked "Urgent," and an automatic review flag
notes that no clinical reason was provided. [00c33de5]
```

### 2 · Summary of one form

```text
$ python -m form_agent summarize 0a18d138

**Summary** — This is a Texas standard prior authorization request, submitted by requesting provider Amy
Johnson for patient Donna Lee. It is marked as an urgent initial request for outpatient occupational
therapy, with two services listed; the form is dated April 15, 2022. [0a18d138]

**Key details**
- Patient: Donna Lee; member/Medicaid ID 35946; DOB June 15, 1985; sex marked male.
- Requesting provider: Amy Johnson; NPI 8041191212; contact Richard Gray.
- Services: "Excision of" — code 26116, April 15–18, 2022, ICD-10 S49.149D; "Surgical laparoscopy" —
  code 43287, May 20–22, 2023, ICD-10 S20.452D.  …

**Needs review**
- The form is marked urgent, but no clinical reason for urgency is provided.
- The requesting provider NPI (8041191212) fails the NPI check-digit test.  …
- The second service starts 400 days after the request date; confirm the dates.
- OCR disagrees on the second service description: extracted as "Surgical laparoscopy," while local OCR
  reads "Surgical laparascopy." Confirm the spelling.
```

The last flag is real. The scan says *laparascopy* (a typo on the form), the vision model silently fixed it,
and the OCR cross-check caught it.

### 3 · Answer across many forms

```text
$ python -m form_agent ask "Which patients are getting speech therapy, and who is the oldest patient?" --trace
tools:
  → sql({"query": "SELECT doc_id, patient_name, patient_dob, services_therapy_types FROM tx_prior_auth WHERE services_therapy_types LIKE '%Speech Therapy%' ORDER BY patient_dob;"})
  → sql({"query": "SELECT doc_id, patient_name, patient_dob FROM tx_prior_auth WHERE patient_dob IS NOT NULL ORDER BY patient_dob ASC LIMIT 1;"})
  → get_document({"doc_id": "0ab19462-881c-49bc-802c-17bd7a02c1ca_TX_page_1"})
  … (3 more get_document calls to confirm)

Patients marked for Speech Therapy are Richard Mckee [0ab19462], Danielle Pratt [0a99e209], and James
Williams [0ac25a81].

The oldest patient is Philip Gilbert, born 03/01/1957 (69 years old as of 2026-10-19) [0adc26e6].

[gpt-6-luna · 6 tool calls · 11.0s · $0.0014]
```

```text
$ python -m form_agent ask "What fax number should behavioral health requests be sent to?"

Fax behavioral health requests to 866-617-4967. This is the pre-printed "Fax: BH Requests" number on the
form. [00c33de5]
```

### 4 · A different kind of form

The same code on FUNSD, 50 noisy 1980s–90s business forms with no typed schema:

```text
$ python -m form_agent ask "What brand is this report about and where was it tested?" --doc 82250337 \
    --db data/funsd.db --cache data/cache_funsd

The report is about Kool Waterfall, tested in Divisions 621 and 627 (Wisconsin). [82250337]
```

## Project layout

```text
src/form_agent/
  schema.py          the data contract: universal record + typed view + flags + run info
  readers/           swappable readers: openai (vision LLM), hybrid (LLM + pixels + OCR); register more with @register
  forms/             form-type plugins: tx_prior_auth.py (typed schema, detection, rules)
  checkboxes.py      OpenCV checkbox detection (boxes, and marks on blank lines) and ink measurement
  ocr.py, verify.py  local OCR (RapidOCR) and cross-checking of LLM output
  validate.py        rule-based review flags (generic by value type + per form type)
  store.py           JSON cache (source of truth) → SQLite index (FTS5 search, typed tables, template constants)
  tools.py, agent.py the agent's read-only tools and the LLM tool loop (with conversation history)
  layout.py          where text and checkboxes are on each scan (cached OCR + boxes)
  view.py            a form as reviewable items: value, position, confidence, flags; evidence for answers
  review.py          human corrections / confirmations → a separate "reviewed" record with a log
  pipeline.py        upload → read → verify → validate → locate → index, in one call
  evaluate.py, qa_eval.py   extraction and Q&A scoring
app/streamlit_app.py the web UI
data/forms/          the 11 Texas prior-auth scans
data/ground_truth/   hand-labeled records for them
data/cache/, data/cache_funsd/   extraction results (JSON), so nothing has to be re-read
data/qa/             the Q&A test set
notebooks/demo.ipynb the demo runs
reports/             evaluation reports (JSON)
docs/                design notes and evaluation
tests/               unit tests (pytest)
```

## Design notes

See [docs/design.md](docs/design.md) for the architecture, the decisions behind it, and the creative
extensions. See [docs/evaluation.md](docs/evaluation.md) for how everything was measured, the error
analysis, and known limitations.

**Creative extensions** (beyond the brief):

- **Self-verification.** Pixel-level checkbox reading and a local OCR cross-check correct or flag the vision
  model's mistakes. This is what moves checkboxes from "usually right" to 100% here.
- **Review flags.** NPI check digits, ICD-10/CPT/HCPCS formats, impossible dates, pick-one groups with two
  ticks, urgent requests without a reason, and retroactive services. They surface in every answer and summary.
- **Template awareness.** Pre-printed values that repeat on every form of a type are detected and stored
  once, so they answer questions without polluting cross-form statistics.
- **Measured, not claimed.** Hand-labeled ground truth, a public benchmark (FUNSD), and a Q&A test set, each
  with a one-line command to reproduce.
- **Swappable readers and form plugins.** A new model or a new form type is one registered class.
- **Web UI with evidence and review** (`app/`). Values are outlined on the scan by reading confidence, answers
  highlight their sources, and corrections are saved separately from the machine reading with a log. The agent
  prefers the reviewed values.

## Data and licenses

The Texas forms were provided with the assignment. FUNSD ([Jaume et al., 2019](https://guillaumejaume.github.io/FUNSD/))
is for non-commercial research use. Only our extraction results are in the repo; run
`scripts/download_funsd.sh` to get the scans and labels for re-scoring.
