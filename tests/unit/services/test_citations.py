from src.services.citations import compact_repeated_single_paper_citations


def test_compact_repeated_single_paper_citations_for_numbered_list():
    answer = (
        "1. 背景与问题：传统方法依赖准确模型 [arXiv:2603.16279v1]。现代四旋翼机动性强 "
        "[arXiv:2603.16279v1]。\n"
        "2. 方法：双方用 PPO 协同进化 [arXiv:2603.16279v1]。\n"
        "3. 结果：捕获率更高 [arXiv:2603.16279v1]。"
    )

    compacted = compact_repeated_single_paper_citations(answer)

    assert compacted.startswith("以下内容基于 [arXiv:2603.16279v1]。")
    assert compacted.count("[arXiv:2603.16279v1]") == 1
    assert "1. 背景与问题：传统方法依赖准确模型。现代四旋翼机动性强。" in compacted
    assert "2. 方法：双方用 PPO 协同进化。" in compacted


def test_compact_repeated_single_paper_citations_keeps_heading_citation():
    answer = (
        "《Drone Interception》[arXiv:2603.16279v1]\n\n"
        "1. 背景与问题：传统方法依赖准确模型 [arXiv:2603.16279v1]。\n"
        "2. 方法：双方用 PPO 协同进化 [arXiv:2603.16279v1]。"
    )

    compacted = compact_repeated_single_paper_citations(answer)

    assert compacted.startswith("《Drone Interception》[arXiv:2603.16279v1]")
    assert compacted.count("[arXiv:2603.16279v1]") == 1
    assert "以下内容基于" not in compacted


def test_compact_repeated_single_paper_citations_preserves_multi_paper_answer():
    answer = (
        "论文 A 强调拦截控制 [arXiv:2603.16279v1]。\n"
        "论文 B 强调视觉导航 [arXiv:2604.09544v1]。\n"
        "二者侧重点不同 [arXiv:2603.16279v1]。"
    )

    assert compact_repeated_single_paper_citations(answer) == answer
