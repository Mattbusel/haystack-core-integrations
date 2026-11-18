# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from agent_budget import Budget, BudgetError, Reservation, Scope

logger = logging.getLogger(__name__)

#: Key in the Agent's `hook_context` under which a shared `agent_budget.Budget` can be passed.
COST_BUDGET_CONTEXT_KEY = "cost_budget"

_EXIT_REASON = "cost_budget_exceeded"
_FINAL_MESSAGE_TEXT = "The Agent stopped because the cost budget was exceeded."

_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens")
# Anthropic reports cache reads and writes on top of `input_tokens`; OpenAI folds cached tokens into `prompt_tokens`.
_ADDITIVE_CACHE_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")
# Conservative characters-per-token ratio used to estimate the prompt of the next call.
_CHARS_PER_TOKEN = 3


def _first_numeric(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def _message_chars(message: ChatMessage) -> int:
    chars = sum(len(text) for text in message.texts)
    chars += sum(len(str(result.result)) for result in message.tool_call_results)
    chars += sum(len(tool_call.tool_name) + len(json.dumps(tool_call.arguments)) for tool_call in message.tool_calls)
    return chars


@dataclass
class _Run:
    budget: Budget
    booked_usd: float = 0.0
    hold: Reservation | None = None


class CostBudgetHook:
    """
    Stop an Agent run before an LLM call would take its cost past a USD budget.

    The hook runs at the `before_llm` hook point. It books the cost of the calls made so far, priced from the
    `token_usage` the Agent records, and then reserves the worst case of the next call: the estimated prompt plus
    `max_output_tokens` of output. When that reservation does not fit in the budget, the run ends before the call
    with the exit reason `"cost_budget_exceeded"`.

    The accounting is done by [llm-agent-budget](https://pypi.org/project/llm-agent-budget/): a hard cap with
    reserve/settle bookkeeping in exact nano-dollar units. To share one cap across concurrent runs, for example one
    budget per user or per request in a server, pass an `agent_budget.Budget` as `hook_context["cost_budget"]`.
    Each run then reserves a slice of up to `max_cost_usd` from it before its first LLM call, books its spend there as
    it happens, and gives back what it did not use, so runs started in parallel can't be admitted for more than the
    shared budget has left.

    Only calls made by the Agent's chat generator are counted, as for `token_usage`. Cached prompt tokens are priced
    at the input price, which overstates the cost of providers that discount cache reads.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack_integrations.hooks.agent_budget import CostBudgetHook

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5-mini", generation_kwargs={"max_completion_tokens": 4096}),
        tools=[web_search],
        hooks={
            "before_llm": [
                CostBudgetHook(
                    max_cost_usd=0.50,
                    input_cost_per_million=0.25,
                    output_cost_per_million=2.00,
                    max_output_tokens=4096,
                )
            ]
        },
    )

    result = agent.run(messages=[...])
    ```
    """

    allowed_hook_points = ("before_llm",)

    def __init__(
        self,
        *,
        max_cost_usd: float,
        input_cost_per_million: float,
        output_cost_per_million: float,
        max_output_tokens: int = 4096,
        add_final_message: bool = False,
    ) -> None:
        """
        Create a cost budget hook.

        :param max_cost_usd: The most one Agent run may spend on LLM calls, in USD.
        :param input_cost_per_million: Price of the model's input (prompt) tokens, in USD per million tokens.
        :param output_cost_per_million: Price of the model's output (completion) tokens, in USD per million tokens.
        :param max_output_tokens: Output tokens to reserve for each call. Set it to the output limit configured on the
            chat generator, so the reservation covers the longest possible reply.
        :param add_final_message: Whether to append an assistant message explaining why the Agent stopped.
        :raises ValueError: If `max_cost_usd` is not positive, a price is negative, or `max_output_tokens` is less
            than 1.
        """
        if not max_cost_usd > 0:
            msg = f"`max_cost_usd` must be a positive amount, got {max_cost_usd}."
            raise ValueError(msg)
        if input_cost_per_million < 0 or output_cost_per_million < 0:
            msg = "Token prices can't be negative."
            raise ValueError(msg)
        if max_output_tokens < 1:
            msg = f"`max_output_tokens` must be a positive number of tokens, got {max_output_tokens}."
            raise ValueError(msg)
        self.max_cost_usd = max_cost_usd
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.max_output_tokens = max_output_tokens
        self.add_final_message = add_final_message
        self._runs: dict[int, _Run] = {}
        self._lock = threading.Lock()

    def cost_of(self, usage: dict[str, Any]) -> float:
        """
        Price a token usage payload, such as the Agent's `token_usage` or a reply's `meta["usage"]`.

        :param usage: Token usage with OpenAI-style (`prompt_tokens`, `completion_tokens`) or Anthropic-style
            (`input_tokens`, `output_tokens`) keys.
        :returns: The cost in USD.
        """
        input_tokens = _first_numeric(usage, _INPUT_TOKEN_KEYS)
        input_tokens += sum(_first_numeric(usage, (key,)) for key in _ADDITIVE_CACHE_KEYS)
        output_tokens = _first_numeric(usage, _OUTPUT_TOKEN_KEYS)
        return (input_tokens * self.input_cost_per_million + output_tokens * self.output_cost_per_million) / 1e6

    def run(self, state: State) -> None:
        """
        Book the cost of the run so far and stop the Agent if the next LLM call might not fit in the budget.

        :param state: Agent state containing the cumulative token usage and the conversation.
        """
        run = self._run_for(state)
        if run is None:
            self._stop(state, "the shared cost budget has nothing left for this run")
            return

        spent_usd = self.cost_of(state.data.get("token_usage") or {})
        cost_usd = max(0.0, spent_usd - run.booked_usd)
        if run.hold is not None:
            hold, run.hold = run.hold, None
            hold.settle(cost_usd)
        elif cost_usd > 0:
            run.budget.charge_force(cost_usd)
        run.booked_usd = max(spent_usd, run.booked_usd)

        estimate_usd = (
            self._estimate_input_tokens(state) * self.input_cost_per_million
            + self.max_output_tokens * self.output_cost_per_million
        ) / 1e6
        try:
            run.hold = run.budget.reserve(estimate_usd)
        except BudgetError as error:
            self._stop(state, str(error))

    def _run_for(self, state: State) -> _Run | None:
        key = id(state)
        with self._lock:
            run = self._runs.get(key)
        if run is not None:
            return run

        shared = (state.data.get("hook_context") or {}).get(COST_BUDGET_CONTEXT_KEY)
        budget: Budget
        if isinstance(shared, Budget):
            slice_usd = min(self.max_cost_usd, shared.remaining_usd)
            if slice_usd <= 0:
                return None
            try:
                budget = shared.child("agent-run", usd=slice_usd)
            except BudgetError:
                return None
        else:
            budget = Budget(self.max_cost_usd, name="agent-run")

        run = _Run(budget=budget)
        with self._lock:
            self._runs[key] = run
        weakref.finalize(state, self._forget, key, run)
        return run

    def _forget(self, key: int, run: _Run) -> None:
        with self._lock:
            if self._runs.get(key) is run:
                del self._runs[key]
        # The Agent's State was released: the last call's hold goes back and, for a run drawn from a shared budget,
        # the unused part of its slice is refunded.
        if run.hold is not None:
            run.hold.release()
            run.hold = None
        if isinstance(run.budget, Scope):
            run.budget.close()

    def _estimate_input_tokens(self, state: State) -> int:
        messages: list[ChatMessage] = state.data.get("messages") or []
        context_tokens = state.data.get("context_tokens")
        if isinstance(context_tokens, int) and context_tokens > 0:
            # `context_tokens` is the prompt plus the reply of the last call; add what was appended after that reply.
            last_reply = max((i for i, m in enumerate(messages) if m.is_from("assistant")), default=-1)
            new_chars = sum(_message_chars(m) for m in messages[last_reply + 1 :])
            return context_tokens + math.ceil(new_chars / _CHARS_PER_TOKEN)
        chars = sum(_message_chars(m) for m in messages)
        chars += sum(
            len(json.dumps(tool.tool_spec)) for tool in state.data.get("tools") or [] if hasattr(tool, "tool_spec")
        )
        return math.ceil(chars / _CHARS_PER_TOKEN)

    def _stop(self, state: State, reason: str) -> None:
        logger.warning(
            "Agent reached its cost budget of ${max_cost_usd}: {reason}; requesting a stop.",
            max_cost_usd=self.max_cost_usd,
            reason=reason,
        )
        state.set("stop_run", _EXIT_REASON)
        if self.add_final_message:
            state.set("messages", [ChatMessage.from_assistant(_FINAL_MESSAGE_TEXT)])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(
            self,
            max_cost_usd=self.max_cost_usd,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
            max_output_tokens=self.max_output_tokens,
            add_final_message=self.add_final_message,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostBudgetHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data=data)
