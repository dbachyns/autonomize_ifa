# Evaluation

Three questions, each with its own measurement:

1. **Does extraction get the values right?** Strict scoring on the Texas forms against hand labels.
2. **Does it work on forms it was not built for?** Lenient scoring on FUNSD.
3. **Does the agent answer correctly?** A question set with known answers.

All runs used `gpt-6-luna`. The reports are in `reports/`.

## 1 · Extraction on the Texas prior-auth forms

```bash
python -m form_agent eval tx --reader hybrid     # or --reader openai
```

**Ground truth.** I hand-labeled all 11 scans (`data/ground_truth/`) against the typed schema: 695 values
per run. That includes 252 fields that are blank on the form, which must stay blank, and the services
table rows.

**Scoring.** Every field of the typed schema is compared after light normalization: whitespace and case
for text, digits for phones and NPIs, ISO format for dates. Table rows are aligned to ground-truth rows
before comparison. A value counts as correct only if it matches exactly; inventing a value for a blank
field counts as an error.

| | gpt-6-luna alone | **hybrid** (+ pixels + OCR) |
|---|---|---|
| all fields | 99.6% (692/695) | **99.9% (694/695)** |
| filled fields | 99.3% (440/443) | **99.8% (442/443)** |
| blank fields left blank | 100% (252/252) | 100% (252/252) |
| checkboxes | 53/55 | **55/55** |
| forms with zero errors | 8/11 | **10/11** |
| errors that carry a review flag | 2/3 | **1/1** |
| form type detected | 11/11 | 11/11 |

**Error analysis.**

- **gpt-6-luna alone** made 3 errors.
  - Twice it read a stray mark in the *Female* box as a tick, so *Sex* came out as two options. The
    validation rule "two ticks in a pick-one group" flags both.
  - Once it "corrected" the typo *Surgical laparascopy* on the scan to *laparoscopy*. Nothing flags that.
- **hybrid** fixes both checkbox errors from pixels: the Female box has 3.7% ink, well under the threshold.
  The remaining *laparoscopy* error now carries an `ocr_disagrees` flag. **Every remaining error is
  flagged for review.**

**Cost and time:** $0.121 for 11 forms (~$0.011 each), with two LLM passes per form. The model
calls take 1–4 minutes per form at `gpt-6-luna`'s current latency; forms run 4 at a time. The local OCR
and pixel checks add about 10–15 seconds per form.

**Caveat:** 11 forms of one layout is a small sample. The FUNSD test below is the check that the approach
generalizes.

## 2 · Generalization to FUNSD

```bash
scripts/download_funsd.sh
python -m form_agent eval funsd --reader openai --cache data/cache_funsd
```

[FUNSD](https://guillaumejaume.github.io/FUNSD/) is 199 real scanned business forms from the 1980s–90s.
They are noisy, varied (fax covers, reports, requisitions, registrations) and partly handwritten. I used the
50-form test split. **Nothing in the system was tuned on these forms**, and no plugin applies: every form
goes through the universal record only.

**Scoring.** FUNSD labels question→answer links. For each labeled pair, I check whether the record contains a
field (or table cell, checkbox or text block) whose value matches the answer (fuzzy similarity ≥ 0.85) with
a compatible label. This is lenient on wording but strict on content.

| | |
|---|---|
| label→value pairs found | **86.5%** (723/836) |
| … and exactly matching | 81.3% (680/836) |
| … counting only values written into blanks (not pre-printed) | 80.5% found |
| cost | $0.136 for 50 forms (~$0.003 each) |

Capturing pre-printed values (tagged `printed`) raised pairs found from 84.8% to 86.5%. FUNSD counts many
printed values as answers, such as a pre-printed company address next to "FROM:".

**Where the misses come from.** I classified the 127 misses from the first run, before printed values
were captured, by hand:

| | misses |
|---|---|
| value not captured at all: mostly pre-printed text FUNSD counts as an answer (addresses, long legal notices, instructions) | 94 |
| one FUNSD answer split across two of our fields (e.g. "Portland # REPS: 6" → "Portland") | 16 |
| misread (faint or handwritten text) | 14 |
| two FUNSD answers merged into one field | 3 |

Capturing printed values recovered part of the first group. Of the 113 misses that remain, only 17 are long
notices or instructions, which the prompt deliberately skips. Most are short values in dense grids, where
FUNSD links a header to a cell differently than we read the table (e.g. `VOLUME` → `77 / 1`). Others are
placeholder answers like `?`, `-` or `N`, which our reader treats as blank.

## 3 · The Q&A agent

```bash
python -m form_agent eval qa
```

**Question set.** `data/qa/tx_questions.json` has 30 questions about the 11 Texas forms. The answers were
checked against the ground truth.

- **13 single-form questions.** Values, checkboxes, a blank field, a field that doesn't exist on the form
  (email), an invalid NPI, and "does anything look wrong?".
- **14 cross-form questions.** Counts, lists, the oldest patient, the most recent form, an average (4.18
  sessions), a tie (physical and speech therapy are both most common), retroactive service dates, a
  pre-printed fax number, and urgent requests missing a reason.
- **3 summaries.** Each must name the patient, providers and codes, and must surface the flagged problems.

**Grading** is deterministic, with no LLM judge. Each question lists requirements that must all appear in the
answer, each with acceptable alternatives (e.g. `03/20/1992` or `March 20, 1992`). Some questions also list
text that must not appear: a wrong form id, *female* when the answer is *male*, or an `@` when there is no
email.

| | passed |
|---|---|
| single-form | 13/13 |
| cross-form | 14/14 |
| summaries | 3/3 |
| **total** | **30/30** |

Cost: $0.023 for all 30 questions, about 6 seconds and 1.8 tool calls per question. Two consecutive full runs
both scored 30/30. One earlier run scored 29/30, but that was a grading bug: "npi" didn't match "NPIs". The
answer itself was correct.

**Caveats.**

- Keyword grading checks that the right facts are present, not that the prose is good. I read the answers
  in `reports/eval_qa_gpt-6-luna.json` by hand as well.
- 30 questions on 11 forms is a smoke test, not a benchmark.
- Answers vary between runs on loosely defined questions. "Which forms are fax cover sheets?" on FUNSD
  returned 10, 11 or 13 forms across runs, depending on whether the model counted borderline titles like
  "TELECOPY INFORMATION". Questions with a crisp definition were stable.

## Known limitations

- **The vision model can silently normalize text** (spelling fixes, date reformatting). The OCR cross-check
  catches cases where OCR reads the text clearly. It can't help where OCR also fails, e.g. on handwriting.
- **Checkbox verification needs printed square boxes.** Circles, underlines or handwritten ✓ next to text
  fall back to the LLM's reading, with no pixel confirmation.
- **The ground truth is mine.** On 11 forms it is easy to check, but it has a single labeler.
- **Latency.** Reading is slow (minutes per form) because of two high-detail vision passes on a
  budget model. Answers are fast because they come from the index.
- **Privacy.** Forms are sent to the OpenAI API. For real PHI this needs a BAA-covered deployment or a local
  reader.
