from src.services.retrieval import Reranker, build_retrieval_plan


def test_retrieval_plan_decomposes_experiment_question():
    plan = build_retrieval_plan("实验部分用了什么方法，有什么效果，和哪些方法进行对比了？")

    assert "experiment" in plan.section_types
    assert len(plan.subqueries) > 1
    assert any("baseline" in subquery for subquery in plan.subqueries)


def test_reranker_boosts_matching_section_type():
    hits = [
        {
            "chunk_id": "method-1",
            "chunk_text": "The controller uses a policy network.",
            "section_title": "Method",
            "section_type": "method",
            "score": 4.0,
        },
        {
            "chunk_id": "experiment-1",
            "chunk_text": "Experiments compare against baseline controllers and report success rate.",
            "section_title": "Experiments",
            "section_type": "experiment",
            "score": 1.0,
        },
    ]

    ranked = Reranker().rerank(
        query="实验部分和哪些方法进行对比",
        hits=hits,
        top_k=1,
        section_types=["experiment"],
    )

    assert ranked[0]["chunk_id"] == "experiment-1"
