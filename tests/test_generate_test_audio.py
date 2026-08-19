from unittest.mock import MagicMock, patch

from scripts.generate_test_audio import TEST_QUESTIONS, synthesize_test_audio


@patch("elevenlabs.client.ElevenLabs")
def test_synthesize_test_audio_writes_one_file_per_question(
    mock_client_cls, tmp_path, monkeypatch
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    # Bypass the committed-fixture copy path so the TTS mock is actually used.
    monkeypatch.setattr(
        "scripts.generate_test_audio.FIXTURE_DIR", str(tmp_path / "no-such-dir")
    )
    mock_client = mock_client_cls.return_value
    mock_client.voices.get_all.return_value.voices = [MagicMock(voice_id="v1")]
    mock_client.text_to_speech.convert.side_effect = lambda **kwargs: iter(
        [b"chunk1", b"chunk2"]
    )

    paths = synthesize_test_audio(output_dir=str(tmp_path))

    assert len(paths) == len(TEST_QUESTIONS)
    for path in paths:
        assert (tmp_path / path.split("/")[-1]).read_bytes() == b"chunk1chunk2"


@patch("elevenlabs.client.ElevenLabs")
def test_synthesize_test_audio_skips_already_synthesized_files(
    mock_client_cls, tmp_path, monkeypatch
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    (tmp_path / "question_0.pcm").write_bytes(b"already-there")
    mock_client = mock_client_cls.return_value
    mock_client.voices.get_all.return_value.voices = [MagicMock(voice_id="v1")]
    mock_client.text_to_speech.convert.return_value = iter([b"new"])

    synthesize_test_audio(output_dir=str(tmp_path))

    assert (tmp_path / "question_0.pcm").read_bytes() == b"already-there"
