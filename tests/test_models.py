import re

import pytest

from voice_normal import doctor, models

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_every_file_is_pinned_and_hashed():
    for model in models.REGISTRY:
        assert model.license and model.source.startswith("https://")
        for file in model.files:
            assert HEX64.match(file.sha256), file.name
            assert file.size > 0
            assert file.url.startswith("https://")
            # Ніяких "/main/": вміст має бути закріплений за комітом.
            assert "/main/" not in file.url and "/master/" not in file.url, file.url


def test_device_specific_files():
    cpu = {f.name for _, f in models.required_files("cpu")}
    cuda = {f.name for _, f in models.required_files("cuda")}
    assert "feature_extractor_cpu.pt" in cpu and "feature_extractor_cuda.pt" not in cpu
    assert "decoder_cuda.pt" in cuda and "decoder_cpu.pt" not in cuda
    assert "model.bin" in cpu and "model.bin" in cuda


def test_cpu_download_is_about_one_and_a_half_gb():
    assert 1.4e9 < models.total_size("cpu") < 1.7e9


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_NORMAL_HOME", str(tmp_path / "Дані"))
    # Справжні файли важать 1.5 ГБ; для перевірки логіки вистачить крихітного реєстру
    # з тією самою структурою (спільні файли + файли під конкретний пристрій).
    fake = tuple(
        models.Model(
            id=m.id,
            purpose=m.purpose,
            license=m.license,
            source=m.source,
            files=tuple(models.ModelFile(f.name, f.url, f.sha256, 16 + i, f.device) for i, f in enumerate(m.files)),
        )
        for m in models.REGISTRY
    )
    monkeypatch.setattr(models, "REGISTRY", fake)
    return tmp_path / "Дані"


def _fake_install(device):
    for model, file in models.required_files(device):
        path = models.file_path(model, file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.truncate(file.size)
    models.write_manifest(device)


def test_doctor_sees_complete_install(home):
    _fake_install("cpu")
    assert models.missing_files("cpu") == []
    assert doctor.check_models("cpu").status == doctor.OK


def test_doctor_sees_truncated_file(home):
    _fake_install("cpu")
    model = models.get_model("whisper-small")
    with open(models.file_path(model, model.files[1]), "r+b") as fh:
        fh.truncate(10)
    check = doctor.check_models("cpu")
    assert check.status == doctor.PENDING
    assert check.data["missing"] == ["whisper-small/model.bin"]


def test_doctor_flags_device_mismatch(home):
    _fake_install("cpu")
    assert doctor.check_models("cuda").status == doctor.WARN


def test_no_manifest_means_not_installed(home):
    assert doctor.check_models("cpu").status == doctor.PENDING
