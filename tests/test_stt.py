from unittest.mock import patch

from app.stt import transcribe


@patch("app.stt._stream_transcript")
def test_transcribe_returns_ok_with_assembled_text(mock_stream):
    mock_stream.return_value = "what is the manhattan project"

    result = transcribe(audio_chunks=[b"chunk1", b"chunk2"])

    assert result.ok is True
    assert result.value.transcript == "what is the manhattan project"


@patch("app.stt._stream_transcript")
def test_transcribe_retries_once_then_succeeds(mock_stream):
    mock_stream.side_effect = [ConnectionError("dropped"), "recovered transcript"]

    result = transcribe(audio_chunks=[b"chunk1"])

    assert mock_stream.call_count == 2
    assert result.ok is True
    assert result.value.transcript == "recovered transcript"


@patch("app.stt._stream_transcript")
def test_transcribe_returns_error_result_on_persistent_failure(mock_stream):
    mock_stream.side_effect = ConnectionError("dropped")

    result = transcribe(audio_chunks=[b"chunk1"])

    assert mock_stream.call_count == 2
    assert result.ok is False
    assert result.value is None
    assert "dropped" in result.error


@patch("app.stt._stream_transcript")
def test_transcribe_handles_empty_transcript_without_error(mock_stream):
    mock_stream.return_value = ""

    result = transcribe(audio_chunks=[b"silence"])

    assert result.ok is True
    assert result.value.transcript == ""
