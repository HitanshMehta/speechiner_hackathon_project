"""The `lfe` command: prepare the data, check and run an extractor, evaluate runs, rebuild the annotations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from live_form_eval.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lfe", description="Evaluate extractors that fill a form from a live "
                                                              "transcript, on SpokenWOZ and AMI.")
    parser.add_argument("--config", help="evaluation config (default: the repo's config.yaml, or $LFE_CONFIG)")
    commands = parser.add_subparsers(dest="command", required=True)

    if _builds_data():  # `prepare` and `annotate`: only in a copy that holds the code that builds the data
        _add_building_commands(commands)

    check = commands.add_parser("check", help="run an extractor on a short made-up call and check the contract")
    check.add_argument("extractor", help="module:Class")

    run = commands.add_parser("run", help="run an extractor on a dataset (official unless a sampling option or "
                                          "--speed is given)")
    run.add_argument("extractor", help="module:Class")
    run.add_argument("--dataset", choices=("spokenwoz", "ami"), required=True)
    run.add_argument("--split", help="spokenwoz: dev or test (default: the config's)")
    run.add_argument("--n", type=int, help="number of conversations to draw (default: the config's)")
    run.add_argument("--seed", type=int, help="seed of the draw (default: the config's)")
    run.add_argument("--ids", nargs="+", help="run exactly these conversations")
    run.add_argument("--speed", type=float, default=1.0, help="replay speed; anything but 1 is unofficial")
    run.add_argument("--concurrency", type=int, default=4, help="conversations replayed at the same time")
    run.add_argument("--name", help="run label (default: the extractor's class name)")

    evaluate = commands.add_parser(
        "evaluate", help="score run folders against the gold and write the report: the runs side by side, and a "
                         "page for every form")
    evaluate.add_argument("runs", nargs="+", type=Path,
                          help="run folders, or results folders of earlier evaluations (their runs are scored "
                               "again). One run per dataset for a full result; the runs of several submissions "
                               "to compare them in one report")
    evaluate.add_argument("--out", type=Path,
                          help="where to write the results (default: results/<name>, or results/comparison when "
                               "the runs have several names)")
    evaluate.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                          help="where the judge of free text runs (AMI): auto = a CUDA GPU if there is one, else "
                               "Apple's GPU, else the CPU. It changes the speed, not the scores")

    ui = commands.add_parser(
        "ui", help="open a page that plays one conversation and shows its form filling: live, with an extractor "
                   "found in this folder (what it calls is really called), or as an earlier run filled it")
    ui.add_argument("--port", type=int, help="the port the page is served on (default: the config's ui.port)")
    ui.add_argument("--no-open", action="store_true", help="don't open the page in a browser")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    command = {"prepare": _prepare, "check": _check, "run": _run, "evaluate": _evaluate, "annotate": _annotate,
               "ui": _ui}
    try:
        command[args.command](args, config)
    except RuntimeError as error:  # something the user can fix (a missing extra, no credentials): say it plainly
        sys.exit(f"lfe {args.command}: {error}")


def _builds_data() -> bool:
    """Does this copy hold the code that builds the data? (A copy that gets the data ready-made doesn't.)"""
    return importlib.util.find_spec("live_form_eval.data.spokenwoz.build") is not None


def _add_building_commands(commands: Any) -> None:
    """`lfe prepare` and `lfe annotate`: building the data, and the LLM annotations it is built from."""
    prepare = commands.add_parser("prepare", help="download the datasets and build conversations, forms and gold")
    prepare.add_argument("--dataset", choices=("spokenwoz", "ami", "all"), default="all")

    annotate = commands.add_parser(
        "annotate", help="organisers: rebuild the LLM annotations (annotations/): the said times of SpokenWOZ, its "
                         "labels checked (--labels), or the answer key of AMI (two Claude readers each: --claude)")
    annotate.add_argument("--dataset", choices=("spokenwoz", "ami"), default="spokenwoz")
    annotate.add_argument("--split", choices=("dev", "test", "all"), default="all", help="spokenwoz only")
    annotate.add_argument("--validate", type=int, metavar="N",
                          help="spokenwoz: only measure the model on N dev values whose spans are hidden")
    annotate.add_argument("--save", action="store_true",
                          help="with --validate: keep the measurement as annotations/spokenwoz/validation-dev.json, "
                               "the shipped model's (`lfe annotate --validate` compares every other model with it)")
    annotate.add_argument("--meetings", nargs="+", metavar="ID",
                          help="ami: only these meetings (what annotations/ami/key/ already holds is always kept)")
    annotate.add_argument("--labels", action="store_true",
                          help="spokenwoz: have two Claude readers check the labels of the official sample's calls "
                               "(config.yaml `labels`; needs --claude) instead of finding said times")
    annotate.add_argument("--ids", nargs="+", metavar="ID",
                          help="spokenwoz --labels: these calls of the official split too (what "
                               "annotations/spokenwoz/ already holds is never asked again)")
    annotate.add_argument("--endpoint", help="OpenAI-compatible server, e.g. http://localhost:8000/v1 "
                                             "(default: run the config's open model here with MLX)")
    annotate.add_argument("--claude", action="store_true",
                          help="use Claude through the Anthropic API instead (the config's claude_model)")
    annotate.add_argument("--model", help="model name on the server, or a Claude model id with --claude")
    annotate.add_argument("--api-key-env", help="environment variable that holds the server's API key")
    annotate.add_argument("--parallel", type=int, default=8, help="requests sent at the same time")
    annotate.add_argument("--extra-body", type=json.loads, default={},
                          help="JSON merged into every request, e.g. "
                               "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")


def _ui(args: argparse.Namespace, config) -> None:
    from live_form_eval.ui.server import serve

    serve(config, port=args.port, open_browser=not args.no_open,
          config_path=Path(args.config).resolve() if args.config else None)


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value:,}")


def _needs_building_sections(config, command: str) -> None:
    if not config.can_build_data():
        sys.exit(f"lfe {command}: this config.yaml has no sections that build the data (said_time, atoms, quotes, "
                 "key, labels); use the organisers' config.yaml")


def _prepare(args: argparse.Namespace, config) -> None:
    _needs_building_sections(config, "prepare")
    if args.dataset in ("spokenwoz", "all"):
        from live_form_eval.data.spokenwoz import build as spokenwoz

        for split in spokenwoz.SPLITS:
            _print_counts(f"spokenwoz/{split}", spokenwoz.prepare(config, split))
            if not spokenwoz.annotation_path(split).exists():
                print(f"  note: {spokenwoz.annotation_path(split).name} is missing, so values without a span or a "
                      "rule are said at the end of the caller's turn (see `lfe annotate`)")
    if args.dataset in ("ami", "all"):
        from live_form_eval.data.ami import build as ami

        _print_counts("ami", ami.prepare(config))
    print(f"written to {config.paths.processed}")


def _check(args: argparse.Namespace, config) -> None:
    from live_form_eval.harness.runner import load_extractor
    from live_form_eval.interface.contract import check_extractor

    report = check_extractor(load_extractor(args.extractor), config)
    for line in report.lines():
        print(line)
    sys.exit(0 if report.ok else 1)


def _run(args: argparse.Namespace, config) -> None:
    from live_form_eval.harness.runner import run

    run_dir = run(args.extractor, args.dataset, config=config, split=args.split, n=args.n, seed=args.seed,
                  ids=args.ids, speed=args.speed, concurrency=args.concurrency, name=args.name)
    print(f"run folder: {run_dir}\nnext: lfe evaluate {run_dir}")


def _evaluate(args: argparse.Namespace, config) -> None:
    from live_form_eval.evaluation.evaluate import evaluate

    result = evaluate(_run_folders(args.runs, config), config=config, out=args.out,
                      device=None if args.device == "auto" else args.device)
    print(result.summary_text())
    print(f"results: {result.out}")
    print(f"open {result.out / 'report.html'}: the runs side by side, and one page per form (the selector at the top)")


def _run_folders(paths: list[Path], config) -> list[Path]:
    """The run folders behind what was named: a run folder itself, or a results folder (the runs it was made
    from, looked for in the config's runs folder). This is how reports are combined: their runs are scored
    together."""
    import pandas as pd

    runs: list[Path] = []
    for path in paths:
        if (path / "run.json").exists():
            runs.append(path)
        elif (path / "summary.csv").exists():
            for row in pd.read_csv(path / "summary.csv").to_dict("records"):
                places = [config.paths.runs / str(row["run"]), Path(str(row.get("run_dir") or ""))]
                found = next((place for place in places if (place / "run.json").exists()), None)
                if found is None:
                    sys.exit(f"lfe evaluate: {path} was made from the run {row['run']}, which is not in "
                             f"{config.paths.runs}")
                runs.append(found)
        else:
            sys.exit(f"lfe evaluate: {path} is neither a run folder (with run.json) nor a results folder (with "
                     "summary.csv)")
    return list({run.resolve(): run for run in runs}.values())  # each run once, in the order given


def _backend(args: argparse.Namespace, config):
    """The model that annotates: Claude, an OpenAI-compatible server, or the config's open model with MLX."""
    from live_form_eval.llm import ClaudeBackend, EndpointBackend, MlxBackend

    if args.claude and args.endpoint:
        sys.exit("choose one: --claude or --endpoint")
    if args.claude:
        return ClaudeBackend(args.model or config.said_time.claude_model, config.said_time.claude_effort,
                             config.said_time.claude_max_tokens, parallel=args.parallel)
    if args.endpoint:
        if not args.model:
            sys.exit("--endpoint needs --model: the model's name on that server")
        return EndpointBackend(args.endpoint, args.model, parallel=args.parallel, extra_body=args.extra_body,
                               api_key=os.environ.get(args.api_key_env) if args.api_key_env else None)
    return MlxBackend(config.said_time.matcher_model, config.said_time.matcher_batch)


def _annotate(args: argparse.Namespace, config) -> None:
    _needs_building_sections(config, "annotate")
    if args.dataset == "ami":
        from live_form_eval.data.ami import build as ami
        from live_form_eval.evaluation.matching.judge.nli import load_judge
        from live_form_eval.llm import ClaudeBackend

        if not args.claude:
            raise SystemExit("the answer key of AMI is made by the two readers of config.yaml (`key.readers`), "
                             "which are Claude models: add --claude (and set ANTHROPIC_API_KEY)")
        counts = ami.annotate_key(
            config, lambda reader: ClaudeBackend(reader.model, reader.effort, config.key.max_tokens,
                                                 parallel=args.parallel), load_judge(config.judge), args.meetings)
        _print_counts("ami: the answer key (annotations/ami/key/)", counts)
        print("next: lfe prepare --dataset ami   (rebuilds the forms and the gold from the key)")
        return
    if args.labels:
        from live_form_eval.data.spokenwoz import build as spokenwoz
        from live_form_eval.llm import ClaudeBackend

        if not args.claude:
            raise SystemExit("SpokenWOZ's labels are checked by the two readers of config.yaml (`labels.readers`), "
                             "which are Claude models: add --claude (and set ANTHROPIC_API_KEY)")
        counts = spokenwoz.annotate_labels(
            config, lambda reader: ClaudeBackend(reader.model, reader.effort, config.labels.max_tokens,
                                                 parallel=args.parallel), args.ids)
        _print_counts("spokenwoz: the labels checked (annotations/spokenwoz/labels-*.json)", counts)
        print("next: lfe prepare --dataset spokenwoz   (rebuilds the gold with the corrected labels)")
        return
    backend = _backend(args, config)

    def report_usage() -> None:
        usage = getattr(backend, "usage", None)
        if usage:
            print(f"  {usage['requests']:,} requests: {usage['input_tokens']:,} input tokens, "
                  f"{usage['output_tokens']:,} output tokens, {usage['cache_write_tokens']:,} written to the cache "
                  f"and {usage['cache_read_tokens']:,} read from it; declined {usage['refusals']}, cut off "
                  f"{usage['cut_off']}, failed {usage['failed']}")

    def progress(done: int, total: int) -> None:
        step = max(1, total // 20)
        if done == total or done // step != (done - 1) // step:
            print(f"  {done}/{total}", flush=True)

    from live_form_eval.data.spokenwoz import build as spokenwoz

    if args.validate:
        report = spokenwoz.validate_matcher(config, backend, args.validate, progress)
        print(f"\n{report['model']}: {report['cases']} dev values with their span hidden, "
              f"{report['seconds_per_value']:.2f} s per value, no words found for {report['no_words_found']}")
        print(f"{'':28}{'within 0.3 s':>14}{'within 1 s':>12}{'median':>9}{'p90':>8}")
        for label, key in (("end of the caller's turn", "turn_end"), ("this model", "matcher")):
            row = report[key]
            print(f"{label:28}{row['within_0.3s']:>14.1%}{row['within_1s']:>12.1%}{row['median_s']:>8.2f}s"
                  f"{row['p90_s']:>7.2f}s")
        shipped = spokenwoz.read_validation()
        if shipped:  # the same first values, as answered by the model behind the shipped annotations
            rows = shipped["rows"][: report["cases"]]
            within = [sum(abs(row["error_s"]) <= limit for row in rows) / len(rows) for limit in (0.3, 1.0)]
            print(f"the shipped annotations' model ({shipped['model']}) on the first {len(rows)} of these values: "
                  f"{within[0]:.1%} within 0.3 s, {within[1]:.1%} within 1 s")
        if args.save:
            effort = config.said_time.claude_effort if args.claude else None
            print(f"kept as {spokenwoz.write_validation(report, effort)}")
        report_usage()
        return
    for split in spokenwoz.SPLITS if args.split == "all" else (args.split,):
        print(f"spokenwoz/{split}")
        meta = spokenwoz.annotate_split(config, split, backend, progress)
        print(f"  {meta['values_asked']} values asked, words found for {meta['words_found']} "
              f"→ {spokenwoz.annotation_path(split)}")
    report_usage()
    print("next: lfe prepare --dataset spokenwoz   (rebuilds the gold with these said times)")


if __name__ == "__main__":
    main()
