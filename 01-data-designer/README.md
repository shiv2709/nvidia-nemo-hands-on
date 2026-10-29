# NeMo Data Designer: a beginner's hands-on tutorial

## What is Data Designer?

Fine-tuning a model needs training data, and writing thousands of examples by hand is slow.
**NVIDIA NeMo Data Designer** generates **synthetic training data** from a *recipe*: you declare
the columns your dataset should have, and Data Designer fills in every row, calling an LLM where
a column needs generated text.

A recipe is built from four kinds of columns:

| Column type | Uses an LLM? | Job |
|---|---|---|
| **Sampler** | No | Picks random values (language, topic, audience) to keep rows varied |
| **LLM text** | Yes | Writes content from a prompt that can reference other columns with `{{ column }}` |
| **LLM judge** | Yes | Scores other columns against a written rubric |
| **Expression** | No | Simple Jinja logic, e.g. choosing which answer is "chosen" |

## What I did

I built two small datasets end to end:

1. **An SFT dataset**: greetings in 7 languages, exported in the chat format used for fine-tuning.
2. **A preference (DPO) dataset**: two models answer the same question and an LLM judge picks the better answer.

**Where it ran:** on my MacBook Air (M2). No GPU is needed, because Data Designer sends each
LLM request to **NVIDIA's hosted Nemotron models** (build.nvidia.com) using a free API key.

## Setup

```bash
conda create -n nemo-lab python=3.11 -y
conda activate nemo-lab
pip install data-designer          # I used version 0.9.4
export NVIDIA_API_KEY="nvapi-..."  # free key from build.nvidia.com (never commit it!)
data-designer config list          # shows the available model aliases
```

---

## Part 1: SFT data (`01-sft-greetings/`)

### Step 1: A vague prompt (`greetings.py`)
A recipe with one sampler (`language`) and one LLM column (`greeting`), previewed in memory:
```bash
python greetings.py
```
**Prompt:** "Write a casual and formal greeting in {{ language }}."
**Result:** 10 rows in about 7 seconds, but the model wrote **mini-lessons** (headers,
explanations, translations, placeholders) instead of two greetings.
**Lesson:** when a prompt leaves gaps, the model fills them with its own assistant habits.

### Step 2: A strict prompt and more samplers (`workflow.py`)
Three samplers (7 languages × 2 formality × 6 situations = 84 combinations) and a strict prompt:
"Return only the greeting itself, with no explanation, labels, quotes, or translation."
A second LLM column translates each greeting into English so it can be checked.
```bash
data-designer preview workflow.py -n 3          # quick quality check (n = next, q = quit)
data-designer create workflow.py --num-records 20 --dataset-name greetings_v1
```
**Result:** every greeting came back short. The format problem was fixed.

### Step 3: Inspect and export (`inspect_export.py`)
```bash
python inspect_export.py
```
Checks balance and duplicates, prints every row, and exports `greetings_v1.jsonl` in chat format:
```json
{"messages": [{"role": "user", "content": "Write a formal greeting in Italian for greeting a teacher."},
              {"role": "assistant", "content": "Buongiorno professore."}]}
```
**What the checks found** (in 20 rows): a Hindi row that wasn't a greeting, garbled Kannada,
2 duplicates, contradictory combinations (a *formal* text to a friend → "Ciao!"),
and uneven languages (German appeared only once).
**Lesson:** strict prompts fix the *format*, but only inspection catches the remaining errors.

---

## Part 2: Preference data for DPO (`02-preference-dpo/`)

DPO (Direct Preference Optimization) teaches a model *which* of two answers is better.
It needs rows of `prompt`, `chosen` and `rejected`.

### Step 4: Two models and a judge (`preference.py`)
| Role | Model |
|---|---|
| Writes the question and answer A | Nemotron 3.5 Lightning (30B, ~3B active) |
| Writes answer B and **judges** both | Nemotron 3.5 Super VL (120B, ~12B active) |

The judge follows a rubric: (1) factual accuracy, (2) clarity for the audience,
(3) under 120 words. Expression columns turn its verdict into `chosen` / `rejected`.
```bash
data-designer preview preference.py -n 3
data-designer create preference.py --num-records 30 --dataset-name prefs_v1
```

### Step 5: Export and remove ties (`export_dpo.py`)
```bash
python export_dpo.py
```
Ties give DPO nothing to learn, so they are removed.
**Result:** 15 pairs, Super won 14 and Lightning 1, with no ties → `prefs_v1_dpo.jsonl`.

### Step 6: Check the judge for bias (`analyze_bias.py`)
```bash
python analyze_bias.py
```
| Check | Result |
|---|---|
| Average words: Lightning vs Super | 79 vs 86 (small gap) |
| Longer answer won | 67% (below the 80% warning level) |
| Answers / chosen answers over 120 words | 0 / 0 (the rubric was followed) |
| Super win rate by audience | 100% everywhere except high-school students (75%) |

**Lesson:** the judge followed the written rules, but because Super judged its own answers and
won 14 of 15, **self-preference bias can't be ruled out**. A fairer test would use a different judge model.

---

## What I learned
- Vague prompts give messy data; strict prompts fix the format.
- Always inspect generated data: a model learns every bad row it's trained on.
- Random samplers don't guarantee balance; larger runs (or rebalancing) are needed.
- An LLM judge only enforces the rules written in its rubric, and should be checked for bias.

## Notes
- **Model swap:** the original judge, `nvidia/nemotron-3-super-120b-a12b`, returned HTTP 404 even
  though it was still listed, so `preference.py` defines its own aliases and uses
  `nvidia/nemotron-3.5-super-vl-120b-a12b` instead. Aliases make this a one-line change.
- **Only 15 of 30 preference rows completed**, most likely due to free-tier rate limits.
- Nemotron models were used because their licence allows training other models on their outputs.
