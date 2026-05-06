from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_prompt_builder():
    prompts_path = Path(__file__).resolve().parents[3] / "src" / "services" / "ollama" / "prompts.py"
    spec = spec_from_file_location("ollama_prompts_under_test", prompts_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGPromptBuilder


def test_rag_prompt_uses_readable_citation_guidance():
    RAGPromptBuilder = _load_prompt_builder()
    prompt = RAGPromptBuilder().create_rag_prompt(
        query="详细讲解这篇论文",
        chunks=[
            {
                "arxiv_id": "2603.16279v1",
                "chunk_text": "Agile interception is formulated as a competitive MARL task.",
            }
        ],
    )

    assert "[1. arXiv:2603.16279v1]" in prompt
    assert "not after every sentence" in prompt
    assert "avoid repeating the same citation after every point" in prompt
