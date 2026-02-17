#!/usr/bin/env python3
"""
Local Edge-TTS HTTP server for Prompt Studio cards.

Install:
  pip install edge-tts flask flask-cors

Run:
  python3 tts_edge_server.py

API:
  POST /api/tts
  body: { "text": "...", "voice": "zh-CN-XiaoxiaoNeural", "rate": "+0%", "pitch": "+0Hz" }
  resp: audio/mpeg bytes
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import edge_tts
from flask import Flask, jsonify, request, Response
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"
DEFAULT_PITCH = "+0Hz"
FALLBACK_VOICES = [
    {"ShortName": "zh-CN-XiaoxiaoNeural", "Locale": "zh-CN", "Gender": "Female"},
    {"ShortName": "zh-CN-YunxiNeural", "Locale": "zh-CN", "Gender": "Male"},
    {"ShortName": "zh-CN-XiaoyiNeural", "Locale": "zh-CN", "Gender": "Female"},
    {"ShortName": "zh-CN-YunjianNeural", "Locale": "zh-CN", "Gender": "Male"},
]


def run_async(coro):
    # Flask/desktop runtime sometimes has a running event loop; isolate in thread.
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def build_boundary_option(word_boundary: bool, sentence_boundary: bool) -> str | None:
    opts: list[str] = []
    if sentence_boundary:
        opts.append("SentenceBoundary")
    if word_boundary:
        opts.append("WordBoundary")
    if not opts:
        return None
    return ",".join(opts)


async def synthesize_to_mp3_bytes(
    text: str,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    word_boundary: bool,
    sentence_boundary: bool,
    proxy: str | None,
    connect_timeout: float | None,
    receive_timeout: float | None,
) -> bytes:
    kwargs: dict[str, Any] = {
        "text": text,
        "voice": voice,
        "rate": rate,
        "volume": volume,
        "pitch": pitch,
    }
    boundary = build_boundary_option(word_boundary, sentence_boundary)
    if boundary:
        kwargs["boundary"] = boundary
    if proxy:
        kwargs["proxy"] = proxy
    if connect_timeout is not None:
        kwargs["connect_timeout"] = float(connect_timeout)
    if receive_timeout is not None:
        kwargs["receive_timeout"] = float(receive_timeout)

    communicate = edge_tts.Communicate(**kwargs)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data", b"")
            if isinstance(data, (bytes, bytearray)) and data:
                chunks.append(bytes(data))
    return b"".join(chunks)


@app.post("/api/tts")
def tts_api() -> Response:
    data: dict[str, Any] = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    voice = str(data.get("voice", DEFAULT_VOICE)).strip() or DEFAULT_VOICE
    rate = str(data.get("rate", DEFAULT_RATE)).strip() or DEFAULT_RATE
    volume = str(data.get("volume", DEFAULT_VOLUME)).strip() or DEFAULT_VOLUME
    pitch = str(data.get("pitch", DEFAULT_PITCH)).strip() or DEFAULT_PITCH
    word_boundary = bool(data.get("word_boundary", False))
    sentence_boundary = bool(data.get("sentence_boundary", False))
    proxy = data.get("proxy")
    proxy = str(proxy).strip() if isinstance(proxy, str) and proxy.strip() else None
    connect_timeout = data.get("connect_timeout", None)
    receive_timeout = data.get("receive_timeout", None)

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        audio = run_async(
            synthesize_to_mp3_bytes(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                word_boundary=word_boundary,
                sentence_boundary=sentence_boundary,
                proxy=proxy,
                connect_timeout=connect_timeout,
                receive_timeout=receive_timeout,
            )
        )
        if not audio:
            return jsonify({"error": "empty audio"}), 500
        return Response(audio, mimetype="audio/mpeg")
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500


@app.get("/api/voices")
def voices_api() -> Response:
    try:
        if hasattr(edge_tts, "list_voices"):
            voices = run_async(edge_tts.list_voices())
        else:
            manager = run_async(edge_tts.VoicesManager.create())
            voices = getattr(manager, "voices", []) or []
        return jsonify({"ok": True, "fallback": False, "voices": voices})
    except Exception as exc:  # pragma: no cover
        # Keep UI usable even if remote voice list fetch fails.
        return jsonify({"ok": False, "fallback": True, "error": str(exc), "voices": FALLBACK_VOICES}), 200


@app.get("/api/health")
def health() -> Response:
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
