from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from new_scanner.persona import PersonaCloneService


BENCHMARK_CASES: list[dict[str, str]] = [
    {
        "name": "likely_border_tariffs",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "China has taken advantage of the United States for years, but those days are over. Our tariffs are bringing jobs, factories, and tremendous strength back to America. We are finally putting American workers first.",
    },
    {
        "name": "likely_fake_news_media",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "The Fake News Media never wants to talk about how strong our economy is. Record jobs, great growth, and America is respected again. They hate reporting success, but the people see it.",
    },
    {
        "name": "likely_border_crime",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "We need a strong border and strong law enforcement. Without safety and security, you do not have a country. The American people want common sense, not open borders and chaos.",
    },
    {
        "name": "likely_election_fraud",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "Millions of Americans still know the election was handled very badly and very unfairly. We need transparency, voter security, and a system people can trust. Without that, democracy suffers.",
    },
    {
        "name": "likely_christmas_message",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "Merry Christmas to everyone, including the people who have worked so hard to make our country stronger, safer, and more prosperous. We love our military, we love our police, and we love America.",
    },
    {
        "name": "likely_democrats_radical",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "The Radical Left wants weakness, high crime, and total dependence on government. We want strength, success, and freedom. That is the choice facing our country.",
    },
    {
        "name": "likely_stock_market_jobs",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "The stock market is doing great, jobs are coming back, and people are feeling optimistic again. We built the strongest economy in history once, and we can do it again.",
    },
    {
        "name": "likely_energy_independence",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "American energy means American strength. We should drill, build, and dominate, not beg foreign countries for the resources we already have right under our feet.",
    },
    {
        "name": "likely_nato_fairness",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "For too long, other countries took advantage of us on trade and defense. We made them pay their fair share, and suddenly the world started respecting the United States again.",
    },
    {
        "name": "likely_bless_america",
        "category": "trump_like",
        "expected_label": "Likely",
        "text": "Our country is on the verge of something really special. Strong borders, strong families, strong growth, and pride in our flag. God Bless America!",
    },
    {
        "name": "formal_voting_rights",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Today’s ruling weakens a central safeguard of representative democracy by allowing legislative maps to be drawn in ways that systematically dilute minority participation under the cover of partisan neutrality.",
    },
    {
        "name": "formal_judicial_language",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "The Court’s reasoning reflects an overly formal reading of institutional power and fails to account for the lived democratic consequences of administrative disenfranchisement.",
    },
    {
        "name": "formal_constitutional_rights",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Constitutional governance requires a principled balance between executive authority and minority protections, especially where public legitimacy depends on broad participation and procedural fairness.",
    },
    {
        "name": "formal_public_policy",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "A sustainable public policy response must address structural inequalities through transparent regulation, community consultation, and evidence based institutional reform.",
    },
    {
        "name": "formal_democratic_norms",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Democratic norms are eroded when partisan actors instrumentalize administrative discretion to create asymmetrical barriers to participation among historically marginalized communities.",
    },
    {
        "name": "formal_civic_discourse",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Healthy civic discourse depends on pluralism, procedural restraint, and a shared commitment to institutions that mediate conflict without amplifying exclusionary narratives.",
    },
    {
        "name": "formal_foreign_policy",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "A coherent foreign policy should privilege multilateral stability, calibrated deterrence, and humanitarian proportionality over reactive rhetorical escalation.",
    },
    {
        "name": "formal_legislative_analysis",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "The legislation introduces an asymmetrical compliance burden that will likely intensify regional disparities while weakening administrative accountability mechanisms.",
    },
    {
        "name": "formal_ethics_statement",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Public officials have an ethical obligation to avoid conflicts of interest and to uphold transparency standards that preserve confidence in democratic institutions.",
    },
    {
        "name": "formal_minority_protections",
        "category": "formal_political",
        "expected_label": "Unlikely",
        "text": "Any legitimate reform agenda must ensure that minority protections are not subordinated to short term partisan advantage or selectively enforced administrative priorities.",
    },
    {
        "name": "offdomain_tabletop_rpg",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "I keep running tabletop RPG campaigns and every session turns chaotic because one player tries to min max everything and another blames the dice instead of his own decisions.",
    },
    {
        "name": "offdomain_anime_review",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "The latest anime season had fantastic pacing, better animation than last year, and a surprisingly emotional finale that completely redefined the main character’s arc.",
    },
    {
        "name": "offdomain_python_debugging",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "I spent the whole evening debugging a Python deployment because one dependency broke the environment and the server kept failing after every restart.",
    },
    {
        "name": "offdomain_university_deadline",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "My professor moved the assignment deadline again and now the whole semester schedule is a mess. I still have lab work, a group presentation, and three readings left.",
    },
    {
        "name": "offdomain_gaming_stream",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "The streamer finally beat the boss after six hours, but the chat was more entertaining than the gameplay because everyone kept spamming bad strategy suggestions.",
    },
    {
        "name": "offdomain_recipe_post",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "I tried a new pasta recipe tonight with roasted tomatoes, garlic, and basil, and it actually tasted better than most of the expensive restaurant versions.",
    },
    {
        "name": "offdomain_roommate_complaint",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "My roommate keeps leaving dishes in the sink for days and somehow still acts surprised when the kitchen starts smelling terrible by the end of the week.",
    },
    {
        "name": "offdomain_cosplay_event",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "The cosplay event was packed this year and the craftsmanship on some of the armor builds was unbelievable. You could tell people had spent months on them.",
    },
    {
        "name": "offdomain_board_game_night",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "Board game night was a disaster because the rules explanation took forever and two people checked out before we even finished the first round.",
    },
    {
        "name": "offdomain_fitness_tracker",
        "category": "off_domain",
        "expected_label": "Unlikely",
        "text": "My fitness tracker says I slept well, but I still feel exhausted and my whole workout was terrible. Apparently numbers and reality are not always the same thing.",
    },
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_root = root / "artifacts" / "persona_benchmark"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = artifacts_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    persona = PersonaCloneService(
        subject="Donald Trump",
        corpus_dir=root / "persona_data" / "donald_trump" / "twitter_archive",
        artifacts_dir=root / "artifacts" / "persona" / "donald_trump",
        config_path=root / "config.yaml",
        top_k=5,
    )

    results: list[dict[str, Any]] = []
    for case in BENCHMARK_CASES:
        try:
            verdict = persona.assess(case["text"])
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "expected_label": case["expected_label"],
                    "predicted_label": verdict.label,
                    "correct": verdict.label == case["expected_label"],
                    "text": case["text"],
                    "explanation": verdict.explanation,
                    "sources": verdict.sources,
                    "evidence_count": len(verdict.evidence),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "expected_label": case["expected_label"],
                    "predicted_label": "Error",
                    "correct": False,
                    "text": case["text"],
                    "explanation": str(exc),
                    "sources": [],
                    "evidence_count": 0,
                }
            )

    summary = build_summary(results, run_id=run_id, root=root)

    (output_dir / "benchmark_cases.json").write_text(
        json.dumps(BENCHMARK_CASES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_summary(results: list[dict[str, Any]], *, run_id: str, root: Path) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row["correct"])
    predicted_counter = Counter(row["predicted_label"] for row in results)
    expected_counter = Counter(row["expected_label"] for row in results)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)

    category_summary: dict[str, Any] = {}
    for category, rows in by_category.items():
        category_total = len(rows)
        category_correct = sum(1 for row in rows if row["correct"])
        category_summary[category] = {
            "count": category_total,
            "correct": category_correct,
            "accuracy": round(category_correct / category_total, 4) if category_total else 0.0,
            "predicted_labels": dict(Counter(row["predicted_label"] for row in rows)),
        }

    confusion: dict[str, dict[str, int]] = {}
    for expected in sorted(expected_counter):
        rows = [row for row in results if row["expected_label"] == expected]
        confusion[expected] = dict(Counter(row["predicted_label"] for row in rows))

    errors = [
        {
            "name": row["name"],
            "category": row["category"],
            "expected_label": row["expected_label"],
            "predicted_label": row["predicted_label"],
            "text": row["text"],
        }
        for row in results
        if not row["correct"]
    ]

    return {
        "project_root": str(root),
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "subject": "Donald Trump",
        "benchmark_size": total,
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "correct_predictions": correct,
        "expected_labels": dict(expected_counter),
        "predicted_labels": dict(predicted_counter),
        "category_summary": category_summary,
        "confusion_by_expected_label": confusion,
        "error_count": len(errors),
        "sample_errors": errors[:10],
    }


if __name__ == "__main__":
    main()
