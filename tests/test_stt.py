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


# The payload shape below is captured verbatim from a live scribe_v2_realtime
# session. The previous tests mocked _stream_transcript wholesale, so the
# parsing inside it was never exercised -- and it read the wrong key, which
# made every real voice query return an empty transcript.
def test_extract_transcript_reads_the_text_key_from_a_real_payload():
    from app.stt import extract_transcript

    payload = {
        "message_type": "committed_transcript",
        "text": "What was the immediate impact of the Manhattan Project?",
    }
    assert (
        extract_transcript(payload)
        == "What was the immediate impact of the Manhattan Project?"
    )


def test_extract_transcript_is_empty_for_silence_or_missing_payload():
    from app.stt import extract_transcript

    assert (
        extract_transcript({"message_type": "committed_transcript", "text": ""}) == ""
    )
    assert extract_transcript({"message_type": "committed_transcript"}) == ""
    assert extract_transcript(None) == ""


def test_extract_transcript_ignores_the_wrong_key():
    """Guards the exact regression: 'transcript' is not the key the API uses."""
    from app.stt import extract_transcript

    assert extract_transcript({"transcript": "should not be read"}) == ""
