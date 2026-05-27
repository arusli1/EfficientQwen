"""Chat-template gating tests.

Render every tracked `chat_template.jinja` across the three `enable_thinking`
states ({undefined, true, false}) and assert it renders without error and does
not silently inject an unrequested system directive. Catches Jinja gating bugs
that would otherwise only surface on a cloud cold-start.

Run via `make test` or `pytest tests/test_chat_template.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _render(template_path: Path, enable_thinking_state: str) -> str:
    """Render a chat template with `enable_thinking` in one of three states:
    'true', 'false', or 'undefined'. Returns the rendered string."""
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(
        trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=False,
    )
    tmpl = env.from_string(template_path.read_text())
    ctx = {
        "messages": [{"role": "user", "content": "test"}],
        "add_generation_prompt": True,
    }
    if enable_thinking_state == "true":
        ctx["enable_thinking"] = True
    elif enable_thinking_state == "false":
        ctx["enable_thinking"] = False
    # "undefined" -> leave it out of the context entirely
    return tmpl.render(**ctx)


_CYANKIWI_TEMPLATE = REPO_ROOT / "weights" / "cyankiwi" / "chat_template.jinja"


@pytest.mark.skipif(not _CYANKIWI_TEMPLATE.exists(),
                    reason="weights/cyankiwi/chat_template.jinja not present on this host")
@pytest.mark.parametrize("state", ["true", "false", "undefined"])
def test_cyankiwi_template_renders_cleanly(state):
    """The reference template must render under every enable_thinking state
    without injecting a reasoning-budget directive of its own."""
    rendered = _render(_CYANKIWI_TEMPLATE, state)
    assert "Think concisely" not in rendered
    assert "Avoid re-deriving" not in rendered


def _iter_variant_templates() -> list[Path]:
    """All chat_template.jinja files under experiments/ (excluding symlinks)."""
    exp = REPO_ROOT / "experiments"
    if not exp.exists():
        return []
    return [p for p in exp.rglob("chat_template.jinja")
            if p.is_file() and not p.is_symlink()]


@pytest.mark.parametrize("template_path", _iter_variant_templates(),
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_variant_templates_render_without_error(template_path):
    """Every variant template must render for all three enable_thinking states
    without raising."""
    for state in ("true", "false", "undefined"):
        try:
            _render(template_path, state)
        except Exception as e:
            pytest.fail(f"{template_path.name} failed to render with "
                        f"enable_thinking={state}: {e}")
