"""Реєстр моделей: звідки качати, sha256, розмір, ліцензія.

Кожен файл закріплено за конкретним комітом джерела, тож вміст не зміниться без
зміни цього файлу. sha256 великих файлів — це LFS-ідентифікатори Hugging Face,
решту пораховано при закріпленні.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from voice_normal import paths
from voice_normal.paths import long_path

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int
    # None — потрібен на будь-якому пристрої; інакше лише для "cpu" або "cuda".
    device: str | None = None


@dataclass(frozen=True)
class Model:
    id: str
    purpose: str
    license: str
    source: str
    files: tuple[ModelFile, ...]

    def files_for(self, device: str) -> list[ModelFile]:
        return [f for f in self.files if f.device in (None, device)]


def _hf(repo: str, revision: str, name: str, sha256: str, size: int, device: str | None = None) -> ModelFile:
    return ModelFile(name, f"https://huggingface.co/{repo}/resolve/{revision}/{name}", sha256, size, device)


_SIDON = ("sarulab-speech/sidon-v0.1", "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52")
_W2V_BERT = ("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b")
_WHISPER = ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120")
_ECAPA = ("Wespeaker/wespeaker-voxceleb-ecapa-tdnn512", "89426e111abc9ffcc40279baf808e087693c6e9b")
_DNS_CHALLENGE = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"

REGISTRY: tuple[Model, ...] = (
    Model(
        id="sidon",
        purpose="speech restoration (w2v-BERT 2.0 feature cleanser + vocoder, 16 kHz -> 48 kHz)",
        license="MIT",
        source="https://huggingface.co/sarulab-speech/sidon-v0.1",
        files=(
            _hf(*_SIDON, "feature_extractor_cpu.pt", "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019", 795498929, "cpu"),
            _hf(*_SIDON, "decoder_cpu.pt", "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4", 209824877, "cpu"),
            _hf(*_SIDON, "feature_extractor_cuda.pt", "c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce", 795490336, "cuda"),
            _hf(*_SIDON, "decoder_cuda.pt", "a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc", 209825094, "cuda"),
        ),
    ),
    Model(
        id="w2v-bert-2.0",
        purpose="feature extractor config for Sidon input",
        license="MIT",
        source="https://huggingface.co/facebook/w2v-bert-2.0",
        files=(
            _hf(*_W2V_BERT, "preprocessor_config.json", "8e6281aad64f97e40534135a59dcc5d33571efae376f2a25adf5551951897ab4", 275),
        ),
    ),
    Model(
        id="dnsmos",
        purpose="speech quality estimate (DNSMOS P.835: SIG/BAK/OVRL)",
        license="CC BY 4.0 (weights), MIT (code) — Microsoft DNS Challenge",
        source="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS",
        files=(
            ModelFile(
                "sig_bak_ovr.onnx",
                f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{_DNS_CHALLENGE}/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
                "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
                1157965,
            ),
        ),
    ),
    Model(
        id="whisper-small",
        purpose="speech recognition for the word check (multilingual)",
        license="MIT",
        source="https://huggingface.co/Systran/faster-whisper-small",
        files=(
            _hf(*_WHISPER, "config.json", "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828", 2370),
            _hf(*_WHISPER, "model.bin", "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671", 483546902),
            _hf(*_WHISPER, "tokenizer.json", "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab", 2203239),
            _hf(*_WHISPER, "vocabulary.txt", "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913", 459861),
        ),
    ),
    Model(
        id="ecapa",
        purpose="speaker embedding for the voice check (ECAPA-TDNN 512)",
        license="Apache-2.0",
        source="https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512",
        files=(
            _hf(*_ECAPA, "voxceleb_ECAPA512.onnx", "f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736", 24861931),
        ),
    ),
)


def get_model(model_id: str) -> Model:
    return next(m for m in REGISTRY if m.id == model_id)


def file_path(model: Model, file: ModelFile, root: Path | None = None) -> Path:
    return (root or paths.models_dir()) / model.id / file.name


def required_files(device: str) -> list[tuple[Model, ModelFile]]:
    return [(m, f) for m in REGISTRY for f in m.files_for(device)]


def total_size(device: str) -> int:
    return sum(f.size for _, f in required_files(device))


def manifest_path(root: Path | None = None) -> Path:
    return (root or paths.models_dir()) / MANIFEST_NAME


def benchmark_path(root: Path | None = None) -> Path:
    return (root or paths.models_dir()) / "benchmark.json"


def read_manifest(root: Path | None = None) -> dict | None:
    path = manifest_path(root)
    try:
        with open(long_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if data.get("version") == MANIFEST_VERSION else None


def write_manifest(device: str, root: Path | None = None) -> None:
    files = {f"{m.id}/{f.name}": {"sha256": f.sha256, "size": f.size} for m, f in required_files(device)}
    data = {"version": MANIFEST_VERSION, "device": device, "files": files}
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(long_path(tmp), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(long_path(tmp), long_path(path))


def missing_files(device: str, root: Path | None = None) -> list[str]:
    """Швидка перевірка за наявністю й розміром; повну sha256 робить install --verify."""
    missing = []
    for model, file in required_files(device):
        path = file_path(model, file, root)
        try:
            if os.stat(long_path(path)).st_size != file.size:
                missing.append(f"{model.id}/{file.name}")
        except OSError:
            missing.append(f"{model.id}/{file.name}")
    return missing
