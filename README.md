# Surge GDP.pdf Benchmark

Standalone runners for reproducing GDP.pdf target generation and rubric judging.

The dataset is loaded from the public Hugging Face dataset
`surgeai/GDP.pdf`. PDF input is sent per provider in whichever way that
provider supports best: OpenAI receives the public Hugging Face
`/resolve/main/...` URL, while Google and Anthropic upload the PDF via their
Files APIs (see [Provider PDF Input](#provider-pdf-input)).

## Setup

Install dependencies:

```bash
uv sync
```

Set the API keys for the providers you plan to run:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...        # or GEMINI_API_KEY, depending on your Gemini SDK setup
export REDUCTO_API_KEY=...       # required for the PDF + Reducto arm
export OPENROUTER_API_KEY=...    # required for judging
```

No private environment manager or external artifact store is required.

## Artifacts

Artifacts are written locally under:

```text
.data/gdp_pdf_benchmark
```

Override the location with:

```bash
export GDP_PDF_BENCHMARK_OUTPUT_DIR=/path/to/artifacts
```

The runner caches sample metadata, source PDFs downloaded by Hugging Face, target
answers, target metadata, Reducto parse artifacts, grades, and `summary.json`
under that artifact directory.

## Benchmark Arms

`plain_pdf` sends the native PDF input first, then `row["prompt"]` verbatim.

`reducto_parse_plus_pdf` first generates a Reducto parse using the local
`REDUCTO_API_KEY`, then sends the native PDF input plus:

```text
Document:
{parsed_doc}

{row["prompt"]}
```

Target models never receive rubrics, gold answers, examples, page-citation
instructions, or a system prompt. The runners do not pass temperature.

## Run Targets

Run native PDF targets:

```bash
uv run gdp-run-pdf \
  --samples 0-9 \
  --models gpt_5_5 opus_4_8 gemini_3_1 \
  --concurrency 4
```

Run native PDF + Reducto parse targets:

```bash
uv run gdp-run-reducto-pdf \
  --samples 0-9 \
  --models gpt_5_5 opus_4_8 gemini_3_1 \
  --parse-concurrency 4 \
  --target-concurrency 4
```

Judge completed targets:

```bash
uv run gdp-judge \
  --samples 0-9 \
  --models gpt_5_5 opus_4_8 gemini_3_1 \
  --arms plain_pdf reducto_parse_plus_pdf \
  --concurrency 8
```

## Judging

Rubric judging follows the dataset evaluation instruction: for each example,
score the response against the rubric columns independently. The current judge
is text-only: it uses OpenRouter `deepseek/deepseek-v4-pro`, receives the
original question, the target response, and all rubric items for the sample in
one LLM call, but does not receive the source PDF or provider PDF-parser output.

OpenRouter judge settings can be overridden with:

```bash
export GDP_PDF_OPENROUTER_JUDGE_MODEL=deepseek/deepseek-v4-pro
export GDP_PDF_OPENROUTER_TIMEOUT_S=900
export GDP_PDF_OPENROUTER_MAX_TOKENS=16384
```

## Provider PDF Input

Each provider receives the PDF in the way that provider handles most reliably:

- **OpenAI** references the PDF by the public Hugging Face `/resolve/main/...`
  URL (`input_file.file_url`).
- **Google** uploads the PDF via the Gemini Files API and references the
  returned file URI.
- **Anthropic** uploads the PDF via the Files API (`document.source.file_id`,
  beta `files-api-2025-04-14`) and streams the response. Uploading avoids the
  URL validator and base64 request-body limits that reject large PDFs, and
  streaming keeps long `effort=max` generations from stalling silently. The
  file is uploaded once per sample and reused across both arms.

## Costs

Costs are estimated locally from provider usage fields and the pricing constants
in `surge_gdp_benchmark.vendor_clients`. Prompt-cache discounts are ignored so
comparisons assume zero cache usage.
