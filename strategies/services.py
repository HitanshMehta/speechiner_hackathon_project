"""Recognize explicit requests for a service directly from live speech."""

from __future__ import annotations

import re

from live_form_eval import FieldType, Form, Partial

SERVICE_FIELD = "services"

_HANDOFF = re.compile(
    r"(?:^|(?<=[.!?;]))\s*(?:how (?:can|may) i help(?: you)?|what can i do for you)(?: today)?[.!?]\s+"
)
_REQUEST = re.compile(
    r"^(?:(?:hello|hi|yes|yeah|okay|ok|oh|well|please|actually|um|uh)[, ]+)*"
    r"(?:(?:i(?: am|'m)|we(?: are|'re)) (?:looking for|trying to (?:book|find|get|reserve))"
    r"|(?:i|we) (?:need|want|would like)(?: to (?:book|find|get|reserve))?"
    r"|(?:i|we)'d like(?: to (?:book|find|get|reserve))?)\s+"
)
_DETERMINER = re.compile(r"^(?:a|an|the)\s+")
_LINK = re.compile(r"^\s*,?\s*and\s+")
_QUALIFIER = re.compile(
    r"^(?:to|from|in|at|for|on|with|without|near|around|by|under|between|leaving|arriving|please|instead|that)\b"
)
_RETRACTION = re.compile(
    r"(?:^|[,;]|\bbut\b)\s*(?:(?:actually|wait|sorry)[, ]+)*"
    r"(?:no(?:$|[, ]+(?:i|we|wait|sorry|actually|thanks)\b)|never mind\b|forget (?:it|that)\b|"
    r"(?:i|we) (?:don't|do not|no longer) (?:need|want|require)\b)"
)
_CLAUSE = re.compile(r"([^.!?;]+)([.!?;]|$)")
# A caller asking to book a table is asking for a restaurant even without the word.
_TABLE = re.compile(r"\b(?:book|reserve|get|need|want|like) (?:a|us a|me a) table\b(?! (?:lamp|cloth))")
_NOUNS = {"hotel": re.compile(r"^rooms?\b"), "train": re.compile(r"^tickets?\b")}


def _clause_services(
    clause: str,
    options: dict[str, str],
    option_pattern: re.Pattern[str],
    *,
    complete: bool,
) -> set[str]:
    request = _REQUEST.match(clause)
    if request is None:
        return set()
    tail = clause[request.end() :]
    services = set()
    while True:
        target = _DETERMINER.sub("", tail, count=1)
        option = option_pattern.match(target)
        if option is None:
            return set()
        word = option.group()
        services.add(options[word])
        tail = target[option.end() :].lstrip()
        if noun := _NOUNS.get(word):
            tail = noun.sub("", tail, count=1).lstrip()
        link = _LINK.match(tail)
        if link is None:
            break
        tail = tail[link.end() :]
    # A partial ending in "a train" might yet say "station". A destination or
    # completed clause confirms the sense. Unknown noun compounds go to the model.
    if not tail:
        return services if complete else set()
    return services if _QUALIFIER.match(tail.strip(", ")) else set()


def _piece_services(
    text: str,
    options: dict[str, str],
    option_pattern: re.Pattern[str],
    *,
    finished: bool,
    first_only: bool,
) -> set[str]:
    services: set[str] = set()
    for index, match in enumerate(_CLAUSE.finditer(text)):
        clause, separator = match.groups()
        clause = clause.strip()
        if _RETRACTION.search(clause):
            services.clear()
        elif not first_only or index == 0:
            services.update(_clause_services(clause, options, option_pattern, complete=finished or bool(separator)))
            if "restaurant" in options and _TABLE.search(clause) and (finished or separator):
                services.add(options["restaurant"])
    return services


def requested_services(form: Form, message: Partial, *, caller: str | None = None) -> set[str]:
    """Return positive service options supported by this message.

    The harness assigns speaker letters in order of first speech. A partial
    carries only its dominant speaker, so without a known caller we inspect
    only a request immediately after a conventional agent handoff.
    """
    spec = next((field for field in form.fields if field.field_id == SERVICE_FIELD), None)
    if spec is None or spec.type != FieldType.multiselect or not spec.options:
        return set()
    options = {option.casefold(): option for option in spec.options}
    literals = sorted(options, key=len, reverse=True)
    option_pattern = re.compile(r"(?:" + "|".join(re.escape(option) for option in literals) + r")(?=$|[\s,])")
    services: set[str] = set()
    finished = message.kind == "turn"
    for speaker, piece in message.pieces():
        text = " ".join(piece.casefold().replace("’", "'").split())
        if speaker == "PENDING":
            continue
        handoffs = list(_HANDOFF.finditer(text))
        if handoffs:
            services.update(
                _piece_services(
                    text[handoffs[-1].end() :],
                    options,
                    option_pattern,
                    finished=finished,
                    first_only=True,
                )
            )
        elif (finished and caller is None) or (caller is not None and speaker == caller):
            services.update(_piece_services(text, options, option_pattern, finished=finished, first_only=False))
    return services
