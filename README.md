# review-panel-ai

Missing the feel of conference reviewing? Here's a way to emulate it (built with the help of Codex). 

Run a small panel of OpenRouter models over a manuscript (plain text, LaTeX, markdown, etc.), let them deliberate for a few rounds, then synthesize a concise consensus review focused on correctness, novelty, and possible improvements.

## Setup

```bash
uv sync
cp .env.example .env
```

Then edit `.env`:

```bash
OPENROUTER_API_KEY=your_openrouter_key
```

## Usage

With a YAML config:

```bash
cp panel_config.example.yaml panel_config.yaml
uv run python main.py --config panel_config.yaml
```

You can still override individual config values from the CLI:

```bash
uv run python main.py --config panel_config.yaml --rounds 3
```

Or pass everything directly as flags:

```bash
uv run python main.py \
  --paper path/to/paper.md \
  --models openai/gpt-4o-mini anthropic/claude-3.5-sonnet google/gemini-2.0-flash-001 \
  --rounds 2 \
  --output panel_review.md \
  --transcript panel_transcript.json
```

## Notes

- `--models` accepts any OpenRouter chat model IDs you want on the panel.
- `--chair-model` lets you choose a different model for the final synthesis pass.
- `--max-paper-chars` can cap very large manuscripts if a selected model has a smaller context window.
- The script expects `OPENROUTER_API_KEY` in the environment.
