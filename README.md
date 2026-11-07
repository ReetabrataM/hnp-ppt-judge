# hack-n-pitch-judge

An AI-assisted pipeline that reads hackathon pitch decks (PDF) and gives organizers a consistent set of scores and flags for every team. Built for **Hack-n-Pitch**, where 200+ teams are compared against each other regardless of track.

> **The AI never decides who wins.** It only produces metrics, evidence, and red flags. The organizers make the final call.

---

## Overview

Every team submits a PDF deck. The stages below are wired together as a **LangGraph** graph, so the pipeline can branch, retry, and resume after a crash. The pipeline:

1. **Checks** each PDF (size, page count, hidden text, prompt-injection attempts).
2. **Extracts** the text and renders each page as a high-res image.
3. **Reads diagrams** (architecture images, charts, screenshots) with a vision model.
4. **Builds a standard "team profile"** for each deck (problem, solution, tech stack, claimed results, etc.), so a game-dev deck and a healthcare deck can be compared on the same terms.
5. **Scores** every team on the same rubric (novelty, feasibility, technical depth, impact, evidence, etc.). Each deck is scored multiple times and the median is used.
6. **Compares** teams using embeddings and clustering, to judge novelty relative to similar ideas and to catch near-duplicates or plagiarism.
7. **Exports** everything to CSV, with a short reason and slide reference behind every score.

---

## What each file does

| Path | Purpose |
|---|---|
| `main.py` | Entry point: builds the graph and runs it over all decks |
| `config.py` | Paths, model names, rate limits, score weights. Keep this identical for everyone on the final run |
| `pipeline/graph.py` | LangGraph wiring: nodes, conditional routing, retries, checkpointing |
| `pipeline/state.py` | The shared state object passed between nodes (deck path, pages, profile, scores, flags) |
| `pipeline/llm.py` | Thin wrapper over the Gemini and Groq SDKs: rate-limit delays, caching, fallback between providers |
| `pipeline/ingest.py` | File checks: size and page limits, hidden-text flags |
| `pipeline/extract.py` | PyMuPDF: pulls text and renders page images |
| `pipeline/vision.py` | Sends diagram pages to the vision model (Gemini) |
| `pipeline/profile.py` | Builds the standard team profile using an LLM |
| `pipeline/score.py` | Rubric scoring, multiple runs, median, variance |
| `pipeline/compare.py` | Embeddings, clustering, pairwise ranking of the top tier |
| `pipeline/integrity.py` | Plagiarism, unsupported claims, prompt-injection flags |
| `pipeline/export.py` | Writes the result CSVs with pandas |
| `prompts/*.txt` | Prompt templates for profiling, vision, scoring, pairwise comparison |
| `rubric/rubric.yaml` | Score level descriptions and weights |
| `tests/test_sample.py` | Basic tests |
| `data/input_pdfs/` | Drop team PDFs here (gitignored) |
| `data/pages/` | Rendered page images (gitignored) |
| `data/cache/` | Saved API responses so reruns don't waste free-tier quota (gitignored) |
| `data/output/` | Final CSVs (gitignored) |

---

## How the graph works

Each file in `pipeline/` is a node. LangGraph only controls the flow; the actual Gemini and Groq calls are plain SDK calls inside the nodes (via `llm.py`), so rate limits and caching stay under our control.

```
ingest -> extract -> vision -> profile -> score -> compare -> integrity -> export
```

Decision points handled by the graph:

- **Text-less pages:** pages with no extractable text are routed to the vision model.
- **Score variance:** if repeated scoring runs disagree too much, re-run scoring, or flag the deck as "needs human review".
- **Provider fallback:** if Gemini is rate-limited, `llm.py` switches to Groq.
- **Resume after a crash:** progress is checkpointed (SQLite), so a failure midway continues instead of restarting all 200+ decks.

---

## Tech stack

- **Language:** Python 3.10+
- **Pipeline orchestration:** LangGraph (flow only, no LangChain agents)
- **PDF reading and rendering:** PyMuPDF
- **LLM and vision model:** Google Gemini API (free tier)
- **Fast text LLM (extraction and scoring runs):** Groq API (free tier)
- **Embeddings and clustering:** Gemini embeddings + scikit-learn
- **Data and export:** pandas, numpy (CSV output)
- **Config:** PyYAML, python-dotenv
- **Utilities and testing:** tqdm, pytest

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/jadavpur-university-ed-cell/hnp-ppt-judge
cd hnp-ppt-judge

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your own .env file
cp .env.example .env            # Windows: copy .env.example .env
```

Then open `.env` and paste in your own API keys (see below).

---

## Getting your API keys

Both are free and need no credit card. **Each teammate should use their own keys.** Never share keys in chats and never commit `.env`.

### Gemini API key
1. Go to **https://aistudio.google.com/apikey**
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy it into `.env` as `GEMINI_API_KEY=...`

### Groq API key
1. Go to **https://console.groq.com/keys**
2. Sign up or log in.
3. Click **Create API Key** and give it a name.
4. Copy it right away (it's shown only once) into `.env` as `GROQ_API_KEY=...`

Your `.env` should look like this:

```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

> Free-tier limits change often. Check your own limits in Google AI Studio and in Groq Console under Limits.

---

## Usage

1. Put the team PDFs in `data/input_pdfs/`.
2. Run:
   ```bash
   python main.py
   ```
3. Find the results in `data/output/`:
   - `summary.csv`: scores, aggregates, rank, flags, confidence, needs-human-review
   - `details.csv`: a brief (maximum 50-word) deck summary, criterion justifications,
     strengths and weaknesses, and evidence slide numbers

CSVs are saved with `utf-8-sig` encoding so Excel opens special characters correctly.

---

## Team rules

- **Use your own `.env`.** Never commit it.
- **Agree on who processes which decks.** The cache is local to each machine, so don't process the same decks twice.
- **Lock model names and settings in `config.py`** for the final run, so scores are comparable across all teams.
- **Never commit team PDFs.** They contain participant data. `data/` is gitignored.
- **Free-tier APIs may use prompts to improve their models.** Tell participants in the rules that decks will be processed by AI tools.

---

## Safeguards

- Deck content is treated as untrusted data, never as instructions to the model.
- Every score comes with a reason and slide references.
- Low-confidence or high-variance decks are flagged for human review.
- AI-written-text detection is only a soft flag with no penalty, since detectors are unreliable and biased against non-native English writers.
- Calibrate first: run the pipeline on a few old decks and check it agrees with organizer judgment before the real run.
