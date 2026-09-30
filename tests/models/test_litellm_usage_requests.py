import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing


async def _get_response(monkeypatch, *, response: ModelResponse):
    async def fake_acompletion(model, messages=None, **kwargs):
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return await LitellmModel(model="test-model").get_response(
        system_instructions=None,
        input=[],
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
    )


def _response_with_null_counts() -> ModelResponse:
    response = ModelResponse(
        choices=[Choices(index=0, message=Message(role="assistant", content="ok"))]
    )
    # Provider SDK parsing can construct typed usage without validating null counts,
    # so a present usage object can still carry None where an int is declared.
    response.usage = Usage.model_construct(  # type: ignore[attr-defined]
        prompt_tokens=None,
        completion_tokens=5,
        total_tokens=None,
    )
    return response


def _response_without_usage() -> ModelResponse:
    response = ModelResponse(
        choices=[Choices(index=0, message=Message(role="assistant", content="ok"))]
    )
    # LiteLLM providers that report nothing leave usage unset or None.
    response.usage = None  # type: ignore[attr-defined]
    return response


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_request_is_counted_when_litellm_reports_no_usage(monkeypatch) -> None:
    """The call happened, so it counts, even though no token counts came back."""
    resp = await _get_response(monkeypatch, response=_response_without_usage())

    assert resp.usage.requests == 1
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0
    assert resp.usage.total_tokens == 0


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_null_token_counts_are_normalized(monkeypatch) -> None:
    """A present usage object may still carry null counts; they read as zero, not a crash."""
    resp = await _get_response(monkeypatch, response=_response_with_null_counts())

    assert resp.usage.requests == 1
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 0
