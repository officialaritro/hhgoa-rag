from app.harness import run_stage


def test_run_stage_returns_ok_value_on_first_success():
    result = run_stage(lambda: "value")

    assert result.ok is True
    assert result.value == "value"
    assert result.error is None


def test_run_stage_retries_once_then_succeeds():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ConnectionError("transient failure")
        return "recovered"

    result = run_stage(flaky, retries=1)

    assert calls["count"] == 2
    assert result.ok is True
    assert result.value == "recovered"


def test_run_stage_returns_error_result_without_raising_after_retries_exhausted():
    calls = {"count": 0}

    def always_fails():
        calls["count"] += 1
        raise ConnectionError("persistent failure")

    result = run_stage(always_fails, retries=1)

    assert calls["count"] == 2
    assert result.ok is False
    assert result.value is None
    assert "persistent failure" in result.error
