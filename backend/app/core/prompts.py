"""All prompt templates in one place."""

from collections.abc import Sequence
from typing import Protocol

# Qwen3-Embedding is trained to receive queries with a task instruction; documents
# are embedded as-is. Omitting the instruction typically costs a few points of recall.
QUERY_INSTRUCTION = "Given a biomedical question, retrieve relevant PubMed abstracts"


def embedding_query(question: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"


# --- answer generation ------------------------------------------------------------


class Source(Protocol):
    pmid: int
    text: str


RAG_SYSTEM = """\
You are a biomedical research assistant. Answer the user's question using ONLY the \
numbered PubMed sources provided.

Rules:
- Cite the sources that support each claim inline as [1], [2], etc. Cite only sources \
that actually support the claim.
- Do not use knowledge that is not in the sources. If the sources do not contain the \
answer, say so clearly, and say what they do cover if it is related.
- If sources disagree, say so and cite both sides.
- Be concise and precise: lead with the direct answer, then the key supporting details.
- This is a research and study tool, not medical advice. Do not give personal medical \
recommendations."""

NO_RAG_SYSTEM = """\
You are a biomedical research assistant. Answer the user's question from your own \
knowledge. Be concise and precise: lead with the direct answer, then the key supporting \
details. If you are not sure, say so. This is a research and study tool, not medical \
advice."""

EVAL_LENGTH_HINT = "Answer in at most 5 sentences."


def format_sources(sources: Sequence[Source]) -> str:
    return "\n\n".join(f"[{i}] (PMID {s.pmid})\n{s.text}" for i, s in enumerate(sources, start=1))


def rag_user_message(question: str, sources: Sequence[Source], hint: str = "") -> str:
    parts = [f"Sources:\n\n{format_sources(sources)}", f"Question: {question}"]
    if hint:
        parts.append(hint)
    return "\n\n".join(parts)


def rag_messages(question: str, sources: Sequence[Source], hint: str = "") -> list[dict]:
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": rag_user_message(question, sources, hint)},
    ]


def no_rag_messages(question: str, hint: str = "") -> list[dict]:
    content = f"Question: {question}" + (f"\n\n{hint}" if hint else "")
    return [
        {"role": "system", "content": NO_RAG_SYSTEM},
        {"role": "user", "content": content},
    ]


# --- LLM-as-judge ----------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a strict expert grader of answers to biomedical questions. You compare a \
candidate answer with a reference answer written by biomedical experts. Judge meaning, \
not wording: a shorter answer that states the key facts correctly can deserve a 5. \
Respond only with the requested JSON."""

CORRECTNESS_RUBRIC = """\
correctness (1-5): does the candidate answer agree with the reference answer?
  5 = states the key facts of the reference correctly, nothing contradicts it
  4 = mostly correct; minor omissions or imprecision
  3 = partially correct; some key facts missing or vague
  2 = mostly incorrect or misses the main point
  1 = wrong, contradicts the reference, or gives no answer"""

GROUNDEDNESS_RUBRIC = """\
groundedness (1-5): are the candidate answer's claims supported by the numbered sources?
  5 = every claim is supported by the sources it cites
  4 = nearly all claims supported; a minor detail is not
  3 = some claims are not supported by the sources
  2 = many claims are not supported by the sources
  1 = the answer mostly ignores or contradicts the sources
  If the answer says the sources do not contain the answer and that is true, \
groundedness is 5 (correctness can still be low)."""


def judge_rag_messages(
    question: str, reference: str, answer: str, sources: Sequence[Source]
) -> list[dict]:
    user = f"""\
Question: {question}

Reference answer (experts):
{reference}

Sources given to the candidate:

{format_sources(sources)}

Candidate answer:
{answer}

Score the candidate answer.

{CORRECTNESS_RUBRIC}

{GROUNDEDNESS_RUBRIC}

Return JSON with "reason" (one or two sentences), "correctness" and "groundedness"."""
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_no_rag_messages(question: str, reference: str, answer: str) -> list[dict]:
    user = f"""\
Question: {question}

Reference answer (experts):
{reference}

Candidate answer:
{answer}

Score the candidate answer.

{CORRECTNESS_RUBRIC}

Return JSON with "reason" (one or two sentences) and "correctness"."""
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


# --- topic labels ----------------------------------------------------------------

TOPIC_SYSTEM = """\
You name clusters of PubMed abstracts for a topic map of biomedical literature. \
Respond only with the requested JSON."""


def topic_label_messages(
    excerpts: Sequence[str], size: int, used_labels: Sequence[str] = ()
) -> list[dict]:
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(excerpts, start=1))
    avoid = (
        "\n\nLabels already used for other clusters (choose a different, more specific "
        "one if this cluster overlaps them): " + "; ".join(used_labels)
        if used_labels
        else ""
    )
    user = f"""\
These are the {len(excerpts)} most typical abstracts from a cluster of {size} abstracts:

{numbered}

Name the shared topic of this cluster.
- "label": 2 to 5 words, specific and in title case (e.g. "Breast Cancer Genetics", \
"Malaria Drug Resistance"). No generic words like "Studies" or "Research".
- "description": one sentence on what the abstracts in this cluster are about.{avoid}

Return JSON with "label" and "description"."""
    return [{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": user}]
