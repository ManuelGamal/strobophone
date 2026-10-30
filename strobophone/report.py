"""Lab report written by Muse Spark, with every number traced to a tool output.

Spark never types a measured value: it cites facts as [[fact_id]] placeholders,
which deterministic code fills in from the measurement. A checker then flags any
digit left in the prose that no fact accounts for, and the report is sent back
for revision (or blocked) until it is clean.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
# Text that contains digits but is not a measured number.
NOT_NUMBERS = [
    PLACEHOLDER.pattern,
    r"m/s(?:\^2|²)",
    r"s(?:\^2|²)",
    r"²",
    r"SAM\s?3\.1",
    r"Muse Spark(?:\s?1\.3)?",
    r"A4",
]
# Arabic-Indic and Eastern Arabic digits (and the Arabic decimal separator) read as
# ASCII, so a number typed in Arabic script is checked like any other.
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
LANGUAGES = {"en": "English", "ar": "Modern Standard Arabic"}
G_REFERENCE = 9.79  # local g in Cairo (latitude 30 N), for comparison only


@dataclass(frozen=True)
class Fact:
    id: str
    value: float | str
    shown: str  # how the report prints it
    source: str  # the tool that produced it
    label: str


def render(text: str, facts: dict[str, Fact]) -> str:
    """Replace [[id]] placeholders with the facts' printed values."""

    def sub(m: re.Match) -> str:
        if m[1] not in facts:
            raise KeyError(f"unknown fact [[{m[1]}]]")
        return facts[m[1]].shown

    return PLACEHOLDER.sub(sub, text)


# Word-valued facts as they should read inside a report in each language.
TRANSLATIONS = {
    "ar": {
        "progressive": "تقدمي (غير متشابك)",
        "tff": "تشابك بالحقل العلوي أولاً",
        "bff": "تشابك بالحقل السفلي أولاً",
        "inside": "داخل",
        "outside": "خارج",
        "consistent": "متسقة",
        "the arcs disagree beyond their errors, so the interval was widened": "الأقواس غير متسقة بما يتجاوز أخطاءها، لذا وُسّعت الفترة",
    }
}


def localize(facts: dict[str, Fact], language: str) -> dict[str, Fact]:
    """Facts whose value is a word, shown in the report's language; numbers untouched."""
    words = TRANSLATIONS.get(language, {})
    return {
        k: dataclasses.replace(f, shown=words[f.shown]) if isinstance(f.value, str) and f.shown in words else f
        for k, f in facts.items()
    }


def segments(text: str, facts: dict[str, Fact]) -> list[dict]:
    """The rendered report as plain-text runs and fact runs, so a page can mark
    every number with the tool it came from."""
    out, pos = [], 0
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            out.append({"text": text[pos : m.start()]})
        f = facts[m[1]]
        out.append({"text": f.shown, "fact": f.id, "source": f.source, "label": f.label})
        pos = m.end()
    if pos < len(text):
        out.append({"text": text[pos:]})
    return out


def _traced(token: str, facts: dict[str, Fact]) -> bool:
    x = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10**-decimals + 1e-9
    for f in facts.values():
        if isinstance(f.value, (int, float)) and abs(float(f.value) - x) <= tol:
            return True
        if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d])", f.shown):
            return True
    return False


def untraced_numbers(text: str, facts: dict[str, Fact]) -> list[str]:
    """Digits in `text` (outside placeholders, units, model names) that no fact explains."""
    stripped = text.translate(DIGITS)
    for pattern in NOT_NUMBERS:
        stripped = re.sub(pattern, " ", stripped)
    return [
        tok for tok in NUMBER.findall(stripped) if not _traced(tok.lstrip("+"), facts)
    ]


def facts_for(m, scale) -> dict[str, Fact]:
    """Facts from a strobophone.measure.VideoMeasurement, each tagged with its source."""
    r = m.result
    lo, hi = r.g_ci_with_scale
    dev = 100 * (r.fit.g - G_REFERENCE) / G_REFERENCE
    items = [
        (
            "g",
            r.fit.g,
            f"{r.fit.g:.2f} m/s²",
            "fit_free_fall",
            "measured gravitational acceleration",
        ),
        (
            "g_ci_low",
            lo,
            f"{lo:.2f} m/s²",
            "fit_free_fall + scale",
            "95% CI lower bound (fit and scale)",
        ),
        (
            "g_ci_high",
            hi,
            f"{hi:.2f} m/s²",
            "fit_free_fall + scale",
            "95% CI upper bound (fit and scale)",
        ),
        (
            "g_half_width",
            (hi - lo) / 2,
            f"{(hi - lo) / 2:.2f} m/s²",
            "fit_free_fall + scale",
            "95% CI half-width",
        ),
        (
            "g_reference",
            G_REFERENCE,
            f"{G_REFERENCE:.2f} m/s²",
            "constant",
            "reference g in Cairo",
        ),
        (
            "deviation_pct",
            dev,
            f"{dev:+.1f}%",
            "report.facts_for",
            "measured g minus reference, percent",
        ),
        (
            "n_points",
            r.fit.n_points,
            str(r.fit.n_points),
            "fit_free_fall",
            "tracked points used in the fit",
        ),
        (
            "residual_px",
            r.fit.residual_rms_px,
            f"{r.fit.residual_rms_px:.2f} px",
            "fit_free_fall",
            "rms fit residual",
        ),
        ("fps", m.fps, f"{m.fps:g} fps", "prepare", "frame rate of the analysed clip"),
        (
            "field_order",
            m.prep.parity or "progressive",
            m.prep.parity or "progressive",
            "prepare",
            "interlacing field order",
        ),
        (
            "zoom",
            m.prep.zoom,
            f"{m.prep.zoom:g}×",
            "prepare",
            "magnification applied before tracking",
        ),
        (
            "concept",
            m.concept,
            f"“{m.concept}”",
            "track_with_fallbacks",
            "SAM 3.1 concept phrase that found the object",
        ),
        (
            "px_per_m",
            scale.px_per_m,
            f"{scale.px_per_m:.1f} px/m",
            "calibration",
            "image scale",
        ),
        (
            "scale_unc_pct",
            100 * scale.rel_uncertainty,
            f"{100 * scale.rel_uncertainty:.1f}%",
            "calibration",
            "scale uncertainty",
        ),
        (
            "fall_time",
            (r.segments[0][-1] - r.segments[0][0]) / m.fps,
            f"{(r.segments[0][-1] - r.segments[0][0]) / m.fps:.3f} s",
            "fit_free_fall",
            "duration of the fitted fall",
        ),
    ]
    inside = "inside" if lo <= G_REFERENCE <= hi else "outside"
    items.append(("ci_level", 95, "95%", "convention", "confidence level of every interval"))
    items.append(("ref_vs_ci", inside, inside, "report.facts_for", "whether the reference g lies inside or outside the 95% interval"))
    consistency = "the arcs disagree beyond their errors, so the interval was widened" if r.warnings else "consistent"
    items.append(("arc_consistency", consistency, consistency, "measure_free_fall", "whether the per-arc g values agree"))
    items.append(("n_arcs", len(r.segments), str(len(r.segments)), "measure_free_fall", "ballistic arcs fitted jointly (drop and rebounds)"))
    if m.restitution is not None:
        e, (elo, ehi) = m.restitution.e, m.restitution.ci
        items += [
            ("e", e, f"{e:.3f}", "measure_restitution", "coefficient of restitution at the first bounce"),
            ("e_ci_low", elo, f"{elo:.3f}", "measure_restitution", "95% CI lower bound of e"),
            ("e_ci_high", ehi, f"{ehi:.3f}", "measure_restitution", "95% CI upper bound of e"),
        ]
    return {i: Fact(i, v, s, src, lab) for i, v, s, src, lab in items}


SYSTEM = """You write the results section of a physics lab report for a free-fall experiment.
Rules:
- Never type a digit. Cite every number as a placeholder [[fact_id]] from the fact list; it is replaced by the measured value.
- Use only the facts given. Do not add textbook values, other experiments or numbers of your own.
- Say plainly whether the reference value lies inside the 95% interval, and name the largest source of uncertainty you can identify from the facts.
- Plain, precise {language}; 120-200 words; no headings, no lists. Keep every placeholder exactly as written (do not translate the ids inside [[ ]])."""

SCHEMA = {
    "type": "object",
    "properties": {"report": {"type": "string"}},
    "required": ["report"],
    "additionalProperties": False,
}


def _fact_table(facts: dict[str, Fact], with_values: bool) -> str:
    rows = []
    for f in facts.values():
        val = f" = {f.shown}" if with_values else ""
        rows.append(f"[[{f.id}]]{val}: {f.label} (from {f.source})")
    return "\n".join(rows)


def _ask(client, model: str, system: str, user: str) -> str:
    r = client.responses.create(
        model=model,
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "report",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(r.output_text)["report"]


@dataclass(frozen=True)
class Report:
    template: str  # Spark's text with placeholders
    text: str  # rendered
    untraced: list[str]  # numbers the checker could not trace (empty when accepted)
    revisions: int
    accepted: bool


def write_report(
    facts: dict[str, Fact],
    client,
    model: str = "muse-spark-1.3",
    max_revisions: int = 2,
    language: str = "en",
) -> Report:
    """Spark drafts with placeholders; the checker sends it back until no untraced
    number or unknown placeholder remains, or blocks it after `max_revisions`."""
    user = "Facts (cite as placeholders):\n" + _fact_table(facts, with_values=True)
    system = SYSTEM.format(language=LANGUAGES[language])
    template = _ask(client, model, system, user)
    for revision in range(max_revisions + 1):
        problems = []
        unknown = [p for p in PLACEHOLDER.findall(template) if p not in facts]
        if unknown:
            problems.append(
                "unknown placeholders: " + ", ".join(f"[[{p}]]" for p in unknown)
            )
        bad = untraced_numbers(template, facts)
        if bad:
            problems.append("numbers typed directly: " + ", ".join(bad))
        if not problems:
            return Report(template, render(template, facts), [], revision, True)
        if revision == max_revisions:
            break
        template = _ask(
            client,
            model,
            system,
            user
            + "\n\nYour previous draft broke the rules ("
            + "; ".join(problems)
            + "). Rewrite it.\n\nPrevious draft:\n"
            + template,
        )
    return Report(
        template, "", bad + [f"[[{p}]]" for p in unknown], max_revisions, False
    )


FREE_SYSTEM = """You write the results section of a physics lab report for a free-fall experiment from the measured values given.
Plain, precise English; 120-200 words; no headings, no lists. Say whether the reference value lies inside the 95% interval."""


def write_free_report(
    facts: dict[str, Fact], client, model: str = "muse-spark-1.3"
) -> str:
    """Baseline: Spark writes numbers itself from the same facts (no placeholders, no checker)."""
    return _ask(
        client,
        model,
        FREE_SYSTEM,
        "Measured values:\n"
        + _fact_table(facts, with_values=True).replace("[[", "").replace("]]", ""),
    )
