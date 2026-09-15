"""The user-facing throughput readout reports DECODE tokens/sec, not output / whole-call wall clock.

``agent/turn_usage.py`` keeps three sample-aligned per-call lanes (``_api_latency_history``,
``_api_output_history``, ``_api_ttfb_history``); ``throughput_rate`` divides output tokens by
DECODE seconds (latency - ttfb), so provider queue + prefill never sit in the denominator and a
lane that truly decodes at 215-275 t/s stops displaying ~100 t/s. Both consumers (the CLI status
bar snapshot and the TUI ``_get_usage`` readout) read that one helper, and the lanes are written
only by ``record_response_usage`` — these tests pin the contract on the helper and on the real
recording path that feeds it.

Contract under test:

- A call with a recorded TTFT reports ``output_tokens / (latency - ttfb)``, strictly above the
  whole-call rate for the same samples.
- Samples without a usable TTFT (non-streaming, a stream that died before its first chunk, a
  stamp at/after the call end) fall back to the whole-call latency, so a mixed window is never
  overstated.
- An empty window — or one with no positive decode time — reports None, and the consumers then
  omit the readout instead of fabricating a number.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_usage import throughput_rate


# ── Helpers ─────────────────────────────────────────────────────────────


def _usage(out_tokens: int, prompt_tokens: int = 1_000):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=out_tokens,
        total_tokens=prompt_tokens + out_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )


def _agent(tmp_path, monkeypatch):
    """A real AIAgent (so the lanes come from agent/agent_init.py, not a stub)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    return AIAgent(
        api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
        api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t",
        platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )


def _record(agent, out_tokens, *, latency, start_time=None, first_chunk=None):
    """One completed API call through the production recorder — the only writer of the lanes.

    ``first_chunk`` is what the loop stamps per attempt (``_last_api_first_chunk_at``), and it is
    reset between calls exactly here, mirroring agent/turn_api_request.py.
    """
    from agent import turn_usage

    agent._last_api_first_chunk_at = first_chunk
    turn_usage.record_response_usage(
        agent, SimpleNamespace(usage=_usage(out_tokens), id="r", model=agent.model),
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=latency, compression_attempts=0, max_compression_attempts=3,
        api_start_time=start_time,
    )


def _lanes(agent):
    return (list(agent._api_latency_history), list(agent._api_output_history), list(agent._api_ttfb_history))


# ─ The rate itself ──────────────────────────────────────────────────────


def test_decode_rate_excludes_ttft_and_beats_the_whole_call_rate(tmp_path, monkeypatch):
    """output / decode seconds, per attempt's own TTFT — strictly above output / wall clock."""
    agent = _agent(tmp_path, monkeypatch)
    try:
        # 200 tokens after a 9s wait inside a 10s call -> 200 t/s decode, 20 t/s whole-call
        _record(agent, 200, latency=10.0, start_time=1_000.0, first_chunk=1_009.0)
        # 300 tokens after a 4s wait inside a 5s call -> 300 t/s decode, 60 t/s whole-call
        _record(agent, 300, latency=5.0, start_time=2_000.0, first_chunk=2_004.0)
        lats, outs, ttfbs = _lanes(agent)
    finally:
        agent.close()

    # The recorder captured each attempt's own ttfb, sample-aligned with the other two lanes.
    assert ttfbs == [9.0, 4.0]
    assert len(lats) == len(outs) == len(ttfbs) == 2

    rate = throughput_rate(lats, outs, ttfbs)
    assert rate == pytest.approx(500 / 2.0)
    # the lie being fixed: the whole-call denominator folds TTFT in and understates the lane
    assert rate > sum(outs) / sum(lats)
    # ...and it is what the user-facing label renders (same "{:.0f} t/s" format)
    assert f"{rate:.0f} t/s" == "250 t/s"


def test_samples_without_a_usable_ttfb_keep_the_whole_call_latency(tmp_path, monkeypatch):
    """Unknown/absurd TTFT samples get no decode credit, so a mixed window is never overstated."""
    agent = _agent(tmp_path, monkeypatch)
    try:
        _record(agent, 200, latency=10.0, start_time=1_000.0, first_chunk=1_009.0)
        _record(agent, 100, latency=2.0, start_time=2_000.0)                        # non-streamed
        _record(agent, 100, latency=4.0, start_time=3_000.0, first_chunk=3_003.0)   # streamed
        _record(agent, 100, latency=2.0, start_time=4_000.0, first_chunk=4_000.0)   # stamp at call end
        lats, outs, ttfbs = _lanes(agent)
    finally:
        agent.close()

    # No leak from the previous attempt: the unstamped call records None, not 9.0.
    assert ttfbs == [9.0, None, 3.0, None]
    # decode time = (10-9) + 2 + (4-3) + 2 for the samples above; only those get decode credit
    assert throughput_rate(lats, outs, ttfbs) == pytest.approx(500 / 6.0)

    # A window with no ttfb data at all reports exactly the pre-change whole-call rate.
    whole_call = sum(outs) / sum(lats)
    assert throughput_rate(lats, outs) == pytest.approx(whole_call)
    assert throughput_rate(lats, outs, [None] * len(outs)) == pytest.approx(whole_call)
    # ...and the decode rate is never lower than the whole-call rate it replaced.
    assert throughput_rate(lats, outs, ttfbs) > whole_call


def test_empty_or_zero_decode_window_reports_none():
    """No samples, or no positive decode time, means no number (callers omit the readout)."""
    assert throughput_rate([], []) is None
    assert throughput_rate([2.0], []) is None          # unaligned window
    assert throughput_rate([], [100], [1.0]) is None
    assert throughput_rate([0.0], [100], [None]) is None
    assert throughput_rate([-0.8], [100], [None]) is None          # absurd provider timing
    assert throughput_rate([10.0], [100], [10.0]) == pytest.approx(10.0)   # ttfb not < latency
    assert throughput_rate([10.0], [100], [9.0]) == pytest.approx(100.0)


# ── The real recording path (streamed turn) ──────────────────────────────


def _stream_chunk(content=None, finish_reason=None, model=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def test_streamed_turn_feeds_the_decode_lanes(tmp_path, monkeypatch):
    """A real streamed turn writes a positive TTFT into the aligned lane, and the label the
    status bar would print is the decode rate for that turn."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _make_tool_defs

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    try:
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.stream_delta_callback = lambda _text: None  # forces the streaming path
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([
            _stream_chunk(content="Hello"),
            _stream_chunk(content=" world", finish_reason="stop", model="test-model", usage=_usage(200)),
        ])
        with (
            patch("run_agent.AIAgent._create_request_openai_client", return_value=mock_client),
            patch("run_agent.AIAgent._close_request_openai_client"),
            patch("hermes_cli.lifecycle.has_hook", return_value=False),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hi")

        assert result["final_response"] == "Hello world"
        lats, outs, ttfbs = _lanes(agent)

        # one sample per API call, all three lanes aligned
        assert len(lats) == len(outs) == len(ttfbs) == 1
        assert outs == [200]
        assert isinstance(ttfbs[0], float) and 0 < ttfbs[0] < lats[0]
        # the turn's decode rate, which is what the status bar shows — never below whole-call
        assert throughput_rate(lats, outs, ttfbs) > throughput_rate(lats, outs, [None])
    finally:
        agent.close()