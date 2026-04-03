from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = """You are an expert academic paper reviewer.
Be rigorous, concrete, and fair, but think like a thoughtful coauthor rather than
an adversarial conference reviewer. What you will read is a work in progress,
not a finalized manuscript, so your job is to identify what already works,
what needs tightening, and what to do next.

Focus on:
- correctness: are the claims, derivations, and conclusions valid?
- what works: which ideas, arguments, or results seem most promising?
- what needs tightening: which claims, definitions, experiments, or exposition need repair?
- what to do next: which follow-up problems, analyses, or experiments would most improve the project?
"""


@dataclass(frozen=True)
class JudgeReview:
    model: str
    round_name: str
    content: str


console = Console()


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file with review settings.",
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_defaults = load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Run a multi-model reviewing panel over a manuscript.",
        parents=[bootstrap_parser],
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=None,
        help="Path to a manuscript file (plain text, Markdown, LaTeX, etc.).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="OpenRouter model ids to use as panel judges, e.g. openai/gpt-4o-mini.",
    )
    parser.add_argument(
        "--chair-model",
        default=None,
        help="Model used to synthesize the final consensus report. Defaults to the first judge.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of deliberation rounds after the initial independent reviews.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for every OpenRouter request.",
    )
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=None,
        help="Optional hard cap for the manuscript text sent to each model. 0 means no cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the final consensus review markdown.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Optional path to save the full panel transcript as JSON.",
    )
    parser.add_argument(
        "--http-referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter rankings/analytics.",
    )
    parser.add_argument(
        "--app-title",
        default=None,
        help="Optional X-Title header for OpenRouter rankings/analytics.",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable live terminal rendering of the panel discussion.",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    apply_builtin_defaults(args)
    validate_args(args, parser)
    return args


def load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a top-level YAML mapping.")

    defaults: dict[str, Any] = {}
    for key, value in raw_config.items():
        arg_name = key.replace("-", "_")
        if arg_name in {"paper", "output", "transcript"} and value is not None:
            defaults[arg_name] = Path(value)
        else:
            defaults[arg_name] = value
    return defaults


def apply_builtin_defaults(args: argparse.Namespace) -> None:
    if args.rounds is None:
        args.rounds = 2
    if args.temperature is None:
        args.temperature = 0.2
    if args.max_paper_chars is None:
        args.max_paper_chars = 0
    if args.output is None:
        args.output = Path("panel_review.md")
    if args.http_referer is None:
        args.http_referer = os.getenv(
            "OPENROUTER_HTTP_REFERER", "https://localhost/review-panel-ai"
        )
    if args.app_title is None:
        args.app_title = os.getenv("OPENROUTER_APP_TITLE", "review-panel-ai")


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    missing: list[str] = []
    if args.paper is None:
        missing.append("--paper or config key `paper`")
    if not args.models:
        missing.append("--models or config key `models`")
    if missing:
        parser.error("Missing required settings: " + ", ".join(missing))


def load_paper(path: Path, max_chars: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {path}")
    paper_text = path.read_text(encoding="utf-8")
    if max_chars > 0 and len(paper_text) > max_chars:
        return (
            paper_text[:max_chars]
            + "\n\n[TRUNCATED: manuscript exceeded --max-paper-chars]\n"
        )
    return paper_text


def call_openrouter(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    temperature: float,
    http_referer: str,
    app_title: str,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer,
        "X-Title": app_title,
    }
    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter request failed for {model}: HTTP {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response shape for {model}:\n"
            f"{json.dumps(data, indent=2)}"
        ) from exc


def build_initial_prompt(paper_text: str, model: str) -> list[dict[str, str]]:
    user_prompt = f"""Please independently review the manuscript below.

Return concise but specific feedback in markdown with exactly these sections:
## What Works
## What Needs Tightening
## What To Do Next
## Correctness Check
## Confidence

In Confidence, provide one sentence saying how certain you are and why.

Manuscript:
```text
{paper_text}
```

You are judge model: {model}
"""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_discussion_prompt(
    paper_text: str,
    model: str,
    own_latest: str,
    panel_state: str,
    round_index: int,
) -> list[dict[str, str]]:
    user_prompt = f"""This is deliberation round {round_index}.

You are judge model {model}. Re-read the manuscript and the panel's current positions,
then update your review if another judge raised a valid point or if you think they are wrong.

Return markdown with exactly these sections:
## Agreements
## Disagreements
## Revised What Works
## Revised What Needs Tightening
## Revised What To Do Next
## Revised Correctness Check

Keep the response concise, but mention specific claims or sections where possible.

Your previous review:
{own_latest}

Current panel state:
{panel_state}

Manuscript:
```text
{paper_text}
```
"""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_consensus_prompt(
    paper_text: str,
    transcript: list[JudgeReview],
) -> list[dict[str, str]]:
    panel_state = format_panel_state(transcript)
    user_prompt = f"""You are the panel chair. Synthesize the panel's discussion into one concise
consensus review document. Prefer concrete shared conclusions, but note unresolved disagreement
briefly if the panel did not truly converge.

Return markdown with exactly these sections:
# Panel Review
## What Works
## What Needs Tightening
## What To Do Next
## Correctness Check
## Panel Agreement

Constraints:
- Be concise and avoid repeating each judge separately.
- Lead with concrete, author-actionable advice.
- Distinguish manuscript fixes from deeper research directions.
- In Correctness Check, distinguish confirmed issues from uncertain concerns.

Panel transcript:
{panel_state}

Manuscript:
```text
{paper_text}
```
"""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_panel_state(reviews: list[JudgeReview]) -> str:
    blocks: list[str] = []
    for review in reviews:
        blocks.append(
            f"### {review.model} | {review.round_name}\n{review.content.strip()}"
        )
    return "\n\n".join(blocks)


def request_many(
    *,
    models: list[str],
    round_name: str,
    message_builder,
    api_key: str,
    temperature: float,
    http_referer: str,
    app_title: str,
    on_review=None,
) -> dict[str, str]:
    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), 8)) as pool:
        future_to_model = {
            pool.submit(
                call_openrouter,
                model=model,
                messages=message_builder(model),
                api_key=api_key,
                temperature=temperature,
                http_referer=http_referer,
                app_title=app_title,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            results[model] = future.result()
            if on_review is not None:
                on_review(
                    JudgeReview(
                        model=model,
                        round_name=round_name,
                        content=results[model],
                    )
                )
            print(f"[done] received response from {model}", file=sys.stderr)
    return results


def run_panel(args: argparse.Namespace) -> tuple[str, list[JudgeReview]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running the panel."
        )

    watch_enabled = not args.no_watch
    paper_text = load_paper(args.paper, args.max_paper_chars)
    chair_model = args.chair_model or args.models[0]
    transcript: list[JudgeReview] = []
    latest_by_model: dict[str, str] = {}

    announce_stage("1/3 collecting independent judge reviews", watch_enabled)
    initial_reviews = request_many(
        models=args.models,
        round_name="initial_review",
        message_builder=lambda model: build_initial_prompt(paper_text, model),
        api_key=api_key,
        temperature=args.temperature,
        http_referer=args.http_referer,
        app_title=args.app_title,
        on_review=render_live_review if watch_enabled else None,
    )
    for model in args.models:
        content = initial_reviews[model]
        latest_by_model[model] = content
        transcript.append(
            JudgeReview(model=model, round_name="initial_review", content=content)
        )

    for round_index in range(1, args.rounds + 1):
        announce_stage(
            f"2/3 deliberation round {round_index}/{args.rounds}",
            watch_enabled,
        )
        snapshot = format_panel_state(transcript)
        round_reviews = request_many(
            models=args.models,
            round_name=f"deliberation_round_{round_index}",
            message_builder=lambda model, round_index=round_index, snapshot=snapshot: (
                build_discussion_prompt(
                    paper_text=paper_text,
                    model=model,
                    own_latest=latest_by_model[model],
                    panel_state=snapshot,
                    round_index=round_index,
                )
            ),
            api_key=api_key,
            temperature=args.temperature,
            http_referer=args.http_referer,
            app_title=args.app_title,
            on_review=render_live_review if watch_enabled else None,
        )
        for model in args.models:
            content = round_reviews[model]
            latest_by_model[model] = content
            transcript.append(
                JudgeReview(
                    model=model,
                    round_name=f"deliberation_round_{round_index}",
                    content=content,
                )
            )

    announce_stage(f"3/3 synthesizing consensus with {chair_model}", watch_enabled)
    final_review = call_openrouter(
        model=chair_model,
        messages=build_consensus_prompt(paper_text, transcript),
        api_key=api_key,
        temperature=args.temperature,
        http_referer=args.http_referer,
        app_title=args.app_title,
    )
    return final_review, transcript


def announce_stage(message: str, watch_enabled: bool) -> None:
    if watch_enabled:
        console.print(Rule(f"[bold cyan]{message}[/bold cyan]"))
    else:
        print(f"[{message}]", file=sys.stderr)


def render_live_review(review: JudgeReview) -> None:
    title = f"{review.model} | {review.round_name}"
    console.print(
        Panel(
            Markdown(review.content),
            title=title,
            border_style="magenta"
            if review.round_name == "initial_review"
            else "green",
            expand=False,
        )
    )


def save_transcript(path: Path, transcript: list[JudgeReview]) -> None:
    serializable = [
        {
            "model": review.model,
            "round": review.round_name,
            "content": review.content,
        }
        for review in transcript
    ]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        final_review, transcript = run_panel(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.output.write_text(final_review.strip() + "\n", encoding="utf-8")
    if args.transcript:
        save_transcript(args.transcript, transcript)

    if not args.no_watch:
        console.print(
            Panel(
                Markdown(final_review),
                title="Consensus Review",
                border_style="cyan",
                expand=False,
            )
        )

    print(f"\nWrote consensus review to {args.output}")
    if args.transcript:
        print(f"Saved full transcript to {args.transcript}")
    if args.no_watch:
        print("\n" + textwrap.dedent(final_review).strip())


if __name__ == "__main__":
    main()
