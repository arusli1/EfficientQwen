"""Tests for scripts/verify_checkpoint.py — fast checkpoint sanity validation."""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location("verify",
                                                  "scripts/verify_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]):
    """Write a minimal safetensors file. tensors: name -> (dtype, shape, raw_bytes)."""
    header: dict = {}
    cursor = 0
    parts = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [cursor, cursor + len(raw)]}
        parts.append(raw)
        cursor += len(raw)
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in parts:
            f.write(chunk)


def _write_minimal_checkpoint(
    root: Path, *,
    has_mtp: bool = True,
    has_template: bool = True,
    arch: str = "Qwen3_5ForConditionalGeneration",
    total_extra_bytes: int = 0,
):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "architectures": [arch],
        "model_type": "qwen3_5",
    }))
    if has_template:
        (root / "chat_template.jinja").write_text("{{ message }}")
    tensors = {
        "model.language_model.layers.0.weight": ("F32", [1], b"\x00" * 4),
    }
    if has_mtp:
        tensors["mtp.fc.weight"] = ("F32", [1], b"\x01" * 4)
    if total_extra_bytes > 0:
        tensors["filler"] = ("F32", [total_extra_bytes // 4],
                             b"\x00" * total_extra_bytes)
    _write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors},
    }))


def _run(monkeypatch, *args):
    verify = _load()
    monkeypatch.setattr("sys.argv", ["verify_checkpoint.py", *args])
    return verify.main()


def test_clean_checkpoint_passes(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path)
    code = _run(monkeypatch, str(tmp_path))
    assert code == 0
    out = capsys.readouterr().out
    assert "architecture" in out
    assert "MTP head: present" in out


def test_missing_config_fails(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path)
    (tmp_path / "config.json").unlink()
    code = _run(monkeypatch, str(tmp_path))
    assert code == 1
    assert "config.json missing" in capsys.readouterr().out


def test_wrong_architecture_fails(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path, arch="GPT2LMHeadModel")
    code = _run(monkeypatch, str(tmp_path))
    assert code == 1
    assert "architecture" in capsys.readouterr().out


def test_missing_mtp_warns_not_fails(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path, has_mtp=False)
    code = _run(monkeypatch, str(tmp_path))
    assert code == 2  # warning only
    assert "MTP head" in capsys.readouterr().out


def test_require_mtp_flag_promotes_to_error(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path, has_mtp=False)
    code = _run(monkeypatch, str(tmp_path), "--require-mtp")
    assert code == 1
    assert "Phase 3 unavailable" in capsys.readouterr().out


def test_missing_chat_template_warns(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path, has_template=False)
    # no tokenizer_config.json either
    code = _run(monkeypatch, str(tmp_path))
    assert code == 2
    assert "chat_template" in capsys.readouterr().out


def test_template_in_tokenizer_config_satisfies(tmp_path, monkeypatch, capsys):
    _write_minimal_checkpoint(tmp_path, has_template=False)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ message }}"})
    )
    code = _run(monkeypatch, str(tmp_path))
    assert code == 0
    assert "chat template: present" in capsys.readouterr().out


def test_oversize_checkpoint_fails(tmp_path, monkeypatch, capsys):
    verify = _load()
    # Lower the limit so we don't have to write 9 GB to test the boundary.
    monkeypatch.setattr(verify, "BASE_MODEL_BYTES_LIMIT", 10_000)
    _write_minimal_checkpoint(tmp_path, total_extra_bytes=200_000)  # 200 KB > 10 KB
    monkeypatch.setattr("sys.argv", ["verify_checkpoint.py", str(tmp_path)])
    code = verify.main()
    assert code == 1
    assert "exceeds on-disk budget" in capsys.readouterr().out


def test_real_cyankiwi_checkpoint_if_present(monkeypatch):
    """Smoke test against the actually-downloaded cyankiwi if it exists."""
    p = Path("weights/cyankiwi")
    if not p.is_dir():
        pytest.skip("weights/cyankiwi not downloaded")
    code = _run(monkeypatch, str(p))
    # Should pass cleanly — cyankiwi has architecture, MTP head, template
    assert code == 0
