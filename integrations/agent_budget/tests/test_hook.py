# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import Any

import pytest
from agent_budget import Budget
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_integrations.hooks.agent_budget import CostBudgetHook

# Every fake LLM call reports 1000 prompt and 100 completion tokens: $0.002 at $1/M input and $10/M output.
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
CALL_COST = 0.002


@component
class ToolCallingGenerator:
    """Always asks for the `lookup` tool, so the Agent keeps going until a hook or `max_agent_steps` stops it."""

    def __init__(self) -> None:
        self.calls = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        reply = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="lookup", arguments={}, id=f"call_{self.calls}")],
            meta={"usage": dict(USAGE), "finish_reason": "tool_calls"},
        )
        return {"replies": [reply]}


def _agent(hook: CostBudgetHook) -> tuple[Agent, ToolCallingGenerator]:
    generator = ToolCallingGenerator()
    lookup = Tool(
        name="lookup",
        description="Look something up.",
        parameters={"type": "object", "properties": {}},
        function=lambda: "found it",
    )
    agent = Agent(chat_generator=generator, tools=[lookup], max_agent_steps=10, hooks={"before_llm": [hook]})
    return agent, generator


def _hook(max_cost_usd: float, **kwargs: Any) -> CostBudgetHook:
    return CostBudgetHook(
        max_cost_usd=max_cost_usd,
        input_cost_per_million=1.0,
        output_cost_per_million=10.0,
        max_output_tokens=100,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cost_usd": 0},
        {"max_cost_usd": -1.0},
        {"max_cost_usd": 1.0, "input_cost_per_million": -1.0},
        {"max_cost_usd": 1.0, "output_cost_per_million": -1.0},
        {"max_cost_usd": 1.0, "max_output_tokens": 0},
    ],
)
def test_init_rejects_invalid_values(kwargs):
    params = {"input_cost_per_million": 1.0, "output_cost_per_million": 1.0, **kwargs}
    with pytest.raises(ValueError):
        CostBudgetHook(**params)


def test_to_dict_from_dict_round_trip():
    hook = CostBudgetHook(
        max_cost_usd=2.5,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        max_output_tokens=8192,
        add_final_message=True,
    )
    data = hook.to_dict()
    assert data == {
        "type": "haystack_integrations.hooks.agent_budget.hook.CostBudgetHook",
        "init_parameters": {
            "max_cost_usd": 2.5,
            "input_cost_per_million": 3.0,
            "output_cost_per_million": 15.0,
            "max_output_tokens": 8192,
            "add_final_message": True,
        },
    }
    restored = CostBudgetHook.from_dict(data)
    assert restored.to_dict() == data


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 1000, "completion_tokens": 100}, CALL_COST),
        ({"input_tokens": 1000, "output_tokens": 100}, CALL_COST),
        # Anthropic reports cache reads and writes on top of `input_tokens`.
        (
            {
                "input_tokens": 500,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 200,
                "output_tokens": 100,
            },
            CALL_COST,
        ),
        ({}, 0.0),
    ],
)
def test_cost_of(usage, expected):
    assert _hook(1.0).cost_of(usage) == pytest.approx(expected)


def test_stops_before_the_call_that_would_exceed_the_budget():
    agent, generator = _agent(_hook(0.007))

    result = agent.run(messages=[ChatMessage.from_user("Find it.")])

    # Three calls cost $0.006; a fourth would need about $0.0021 more (its estimated prompt plus 100 output tokens),
    # which does not fit in $0.007, so the run ends before it is sent.
    assert result["exit_reason"] == "cost_budget_exceeded"
    assert generator.calls == 3
    assert agent_cost(result) == pytest.approx(0.006)
    assert agent_cost(result) <= 0.007


def test_adds_final_message():
    agent, _ = _agent(_hook(0.003, add_final_message=True))

    result = agent.run(messages=[ChatMessage.from_user("Find it.")])

    assert result["exit_reason"] == "cost_budget_exceeded"
    assert result["last_message"].text == "The Agent stopped because the cost budget was exceeded."


def test_runs_share_a_budget_passed_in_hook_context():
    shared = Budget(0.011)
    agent, generator = _agent(_hook(0.007))

    first = agent.run(messages=[ChatMessage.from_user("Find it.")], hook_context={"cost_budget": shared})
    gc.collect()  # the run's slice is closed when its State is released
    assert generator.calls == 3
    assert shared.spent_usd == pytest.approx(0.006)
    assert shared.reserved_usd == 0

    # The second run gets only the $0.005 the shared budget has left, so it makes two calls instead of three.
    second = agent.run(messages=[ChatMessage.from_user("Find it.")], hook_context={"cost_budget": shared})
    gc.collect()
    assert generator.calls == 5
    assert first["exit_reason"] == second["exit_reason"] == "cost_budget_exceeded"
    assert shared.spent_usd == pytest.approx(0.010)
    assert shared.spent_usd <= shared.limit_usd


def test_exhausted_shared_budget_stops_before_the_first_call():
    shared = Budget(0.001)
    shared.charge_force(0.001)
    agent, generator = _agent(_hook(0.007))

    result = agent.run(messages=[ChatMessage.from_user("Find it.")], hook_context={"cost_budget": shared})

    assert result["exit_reason"] == "cost_budget_exceeded"
    assert generator.calls == 0


def agent_cost(result: dict[str, Any]) -> float:
    return _hook(1.0).cost_of(result["token_usage"])
