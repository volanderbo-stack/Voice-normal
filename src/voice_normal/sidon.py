"""Обгортка над TorchScript-моделями Sidon: 16 кГц на вході, 48 кГц на виході.

Тут лише один прохід моделі по фрагменту. Нарізку довгих записів, склейку й
вирівнювання робить конвеєр обробки (етап 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_normal import models, paths

INPUT_SR = 16_000
OUTPUT_SR = 48_000
# Як в офіційному демо: 10 мс контексту з кожного боку для препроцесора.
_PAD = 160


class Sidon:
    def __init__(self, device: str = "cpu", root: Path | None = None) -> None:
        import torch
        from transformers import SeamlessM4TFeatureExtractor

        self.torch = torch
        self.device = device
        sidon = models.get_model("sidon")
        files = {f.name: models.file_path(sidon, f, root) for f in sidon.files_for(device)}
        self.feature_extractor = torch.jit.load(
            paths.long_path(files[f"feature_extractor_{device}.pt"]), map_location=device
        ).eval()
        self.decoder = torch.jit.load(paths.long_path(files[f"decoder_{device}.pt"]), map_location=device).eval()
        w2v = models.get_model("w2v-bert-2.0")
        self.preprocessor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(models.file_path(w2v, w2v.files[0], root).parent)
        )

    def restore(self, wav_16k: np.ndarray) -> np.ndarray:
        """Один фрагмент моно float32 16 кГц -> відновлений моно float32 48 кГц."""
        torch = self.torch
        padded = np.pad(wav_16k.astype(np.float32), (_PAD, _PAD))
        inputs = self.preprocessor(padded, sampling_rate=INPUT_SR, return_tensors="pt")
        with torch.inference_mode():
            features = self.feature_extractor(inputs["input_features"].to(self.device))["last_hidden_state"]
            restored = self.decoder(features.transpose(1, 2)).reshape(-1)
        return restored.float().cpu().numpy()
