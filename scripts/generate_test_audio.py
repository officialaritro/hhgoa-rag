"""Synthesizes a small, fixed set of representative spoken questions into
16kHz PCM audio, once, for scripts/benchmark_latency.py to reuse -- avoids
needing real human recordings for a repeatable latency benchmark (plan
Task 9). This does NOT test retrieval quality or query diversity -- that is
scripts/evaluate_retrieval.py's job, using real corpus text queries directly.
A latency benchmark only needs enough real STT/generation round trips to get
a stable percentile, not semantic variety.
"""

import os
import shutil
from pathlib import Path

DEFAULT_OUTPUT_DIR = "data/benchmark_audio"

# Pre-generated 16kHz mono PCM clips committed to the repo. Preferred over
# synthesizing, because the ElevenLabs API key is scoped to Speech-to-Text
# only -- TTS returns 401 "missing the permission text_to_speech", and
# widening the key just to build benchmark fixtures is the wrong trade.
# Regenerate on macOS with:
#   say -o q.aiff "<question>"
#   afconvert -f WAVE -d LEI16@16000 -c 1 q.aiff q.wav   # then strip 44-byte header
FIXTURE_DIR = "assets/benchmark_audio"

TEST_QUESTIONS = [
    "What was the immediate impact of the Manhattan Project?",
    "How does photosynthesis work?",
    "What is the capital of France?",
    "Why do leaves change color in autumn?",
    "What causes earthquakes?",
]


def synthesize_test_audio(output_dir: str = DEFAULT_OUTPUT_DIR) -> list[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_paths = [
        Path(output_dir) / f"question_{i}.pcm" for i in range(len(TEST_QUESTIONS))
    ]
    # Committed fixtures first; TTS only for anything they don't cover.
    for i, path in enumerate(audio_paths):
        fixture = Path(FIXTURE_DIR) / f"question_{i}.pcm"
        if not path.exists() and fixture.exists():
            shutil.copyfile(fixture, path)

    missing = [
        (path, q) for path, q in zip(audio_paths, TEST_QUESTIONS) if not path.exists()
    ]

    if missing:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        voice_id = client.voices.get_all().voices[0].voice_id
        for audio_path, question in missing:
            audio_chunks = client.text_to_speech.convert(
                voice_id=voice_id, text=question, output_format="pcm_16000"
            )
            with open(audio_path, "wb") as f:
                f.writelines(audio_chunks)

    return [str(path) for path in audio_paths]


if __name__ == "__main__":
    written_paths = synthesize_test_audio()
    print(f"synthesized {len(written_paths)} test audio clips in {DEFAULT_OUTPUT_DIR}")
