import hashlib
import http.server
import os
import threading

import pytest

from voice_normal import download

PAYLOAD = os.urandom(3 * download.CHUNK + 12345)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    supports_range = True
    requests: list[str | None] = []
    # Скільки байтів віддати, перш ніж "обірвати" з'єднання (None — без обриву).
    cut_after: int | None = None

    def do_GET(self):
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        start = 0
        if rng and self.supports_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).cut_after is not None:
            limit, type(self).cut_after = type(self).cut_after, None
            self.wfile.write(body[:limit])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.requests = []
    _Handler.supports_range = True
    _Handler.cut_after = None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/model.bin"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "Моделі з пробілами" / "model.bin"


def test_fresh_download(server, dest):
    download.download(server, dest, DIGEST, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert not download.part_path(dest).exists()


def test_resume_requests_only_the_rest(server, dest):
    dest.parent.mkdir(parents=True)
    half = len(PAYLOAD) // 2
    download.part_path(dest).write_bytes(PAYLOAD[:half])
    download.download(server, dest, DIGEST, len(PAYLOAD))
    assert _Handler.requests == [f"bytes={half}-"]
    assert dest.read_bytes() == PAYLOAD


def test_server_without_range_restarts_cleanly(server, dest):
    _Handler.supports_range = False
    dest.parent.mkdir(parents=True)
    download.part_path(dest).write_bytes(PAYLOAD[:1000])
    download.download(server, dest, DIGEST, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_dropped_connection_is_resumed(server, dest, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    _Handler.cut_after = download.CHUNK + 7
    download.download(server, dest, DIGEST, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert _Handler.requests[0] is None and _Handler.requests[-1].startswith("bytes=")


def test_checksum_mismatch_removes_partial(server, dest):
    with pytest.raises(download.ChecksumError):
        download.download(server, dest, "0" * 64, len(PAYLOAD))
    assert not dest.exists()
    assert not download.part_path(dest).exists()


def test_oversized_partial_is_discarded(server, dest):
    dest.parent.mkdir(parents=True)
    download.part_path(dest).write_bytes(PAYLOAD + b"junk")
    download.download(server, dest, DIGEST, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_unreachable_host_raises_download_error(dest, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    with pytest.raises(download.DownloadError):
        download.download("http://127.0.0.1:9/nothing", dest, DIGEST, len(PAYLOAD), retries=1, timeout=2)
