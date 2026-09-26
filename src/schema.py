"""Step 2 - Frozen extraction schema (the research instrument).

Defines the entity types, relation types, and annotation rules that the whole
study measures reliability for. Frozen once finalized; consumed by Step 3
(extraction prompt + OpenAI JSON schema) and Step 4 (human verification rubric).
"""


from dataclasses import dataclass


ENTITY_TYPES = ["Paper", "Method", "Task", "Dataset", "Metric"]


@dataclass(frozen=True)
class Relation:
    name: str
    head: tuple
    tail: tuple
    definition: str
    example: str

RELATIONS = [
    Relation(
        name="PROPOSES",
        head=("Paper",),
        tail=("Method",),
        definition="The paper introduces a NEW method, model, framework, or "
                   "benchmark as its own contribution.",
        example='"we propose RING, a novel attack" -> (Paper, PROPOSES, RING)',
    ),
    Relation(
        name="USES_METHOD",
        head=("Paper",),
        tail=("Method",),
        definition="The paper APPLIES an existing method or technique that it "
                   "did not introduce.",
        example='"we apply a maximum-entropy closure" '
                '-> (Paper, USES_METHOD, maximum-entropy closure)',
    ),
    Relation(
        name="ADDRESSES_TASK",
        head=("Paper",),
        tail=("Task",),
        definition="The paper targets or aims to solve this task or problem.",
        example='"output vector editing for memorization mitigation" '
                '-> (Paper, ADDRESSES_TASK, memorization mitigation)',
    ),
    Relation(
        name="EVALUATED_ON",
        head=("Method",),
        tail=("Dataset",),
        definition="The method is tested or evaluated on this NAMED dataset or "
                   "benchmark.",
        example='"evaluated across 20 tasks from MetaWorld" '
                '-> (ARB4WM, EVALUATED_ON, MetaWorld)',
    ),
    Relation(
        name="REPORTS_METRIC",
        head=("Paper",),
        tail=("Metric",),
        definition="The paper reports results measured by this evaluation metric.",
        example='"an attack success rate of 90.3%" '
                '-> (Paper, REPORTS_METRIC, attack success rate)',
    ),
    Relation(
        name="IMPROVES_OVER",
        head=("Method",),
        tail=("Method",),
        definition="The proposed method is claimed to OUTPERFORM this method or "
                   "baseline. A 'comparable / on par' result does NOT count.",
        example='"a 2.7x gap over zero ablation" '
                '-> (output vector editing, IMPROVES_OVER, zero ablation)',
    ),
]

ANNOTATION_RULES = [
    "Extract only NAMED/specific entities; skip vague mentions like 'four datasets'.",
    "IMPROVES_OVER requires an explicit 'better than' claim; 'comparable/on par' does NOT count.",
    "Named pretrained models (e.g. Llama2, TRELLIS) are Method entities.",
    "A proposed benchmark/framework/schema is recorded as a Method (PROPOSES).",
    "New artifact -> PROPOSES; an existing technique applied -> USES_METHOD.",
]

RELATION_NAMES = [r.name for r in RELATIONS]