"""Agent-loop tests with a scripted fake LLM (no network)."""

import asyncio
import json

from health_agent.agent import MAX_STEPS, HealthAgent, Session
from health_agent.llm import LLMError, LLMResult, ToolCall
from health_agent.model_tool import predict_diabetes_risk
from health_agent.rules import RuleBasedClient


class ScriptedLLM:
    """Returns queued extraction dicts for json_mode calls and queued LLMResults otherwise."""

    name = "scripted"

    def __init__(self, extractions, replies):
        self.extractions = list(extractions)
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        if json_mode:
            e = self.extractions.pop(0)
            return LLMResult(content=e if isinstance(e, str) else json.dumps(e))
        return self.replies.pop(0) if self.replies else LLMResult(content="ok")


def call(name, **args):
    return LLMResult(content=None, tool_calls=[ToolCall(id="c1", name=name, arguments=json.dumps(args))])


FULL = {"age": 50, "glucose": 148, "bmi": 33.6, "insulin": 100, "intent": "health_info"}


def run(agent, session, msg):
    return asyncio.run(agent.handle(session, msg))


def test_tool_call_produces_structured_result():
    llm = ScriptedLLM([FULL], [call("predict_diabetes_risk", age=50, glucose=148, bmi=33.6, insulin=100), LLMResult(content="High risk estimated. See a doctor.")])
    reply, metrics = run(HealthAgent(llm), Session(), "I'm 50, sugar 148, bmi 33.6, insulin 100")
    assert reply.status == "result"
    expected = predict_diabetes_risk(age=50, glucose=148, bmi=33.6, insulin=100)
    assert reply.risk_class == expected["risk_class"] and reply.probability == expected["probability"]
    assert reply.model_used == "mlp"
    assert reply.tool_calls[0].ok
    assert reply.disclaimer.startswith("Educational demo")
    assert metrics.llm_calls == 3


def test_invented_value_is_rejected():
    # User gave no insulin; the LLM invents insulin=80 -> guard rejects the call.
    ext = {"age": 50, "glucose": 148, "bmi": 33.6, "intent": "health_info"}
    llm = ScriptedLLM([ext], [call("predict_diabetes_risk", age=50, glucose=148, bmi=33.6, insulin=80), LLMResult(content="Do you know your insulin level?")])
    reply, _ = run(HealthAgent(llm), Session(), "50, sugar 148, bmi 33.6")
    assert reply.status != "result" and reply.probability is None
    assert not reply.tool_calls[0].ok and "not stated" in reply.tool_calls[0].error


def test_known_optional_values_are_filled_in():
    # User gave blood pressure; the LLM left it out of the call -> code adds it.
    ext = {"age": 50, "glucose": 148, "bmi": 33.6, "blood_pressure": 85, "intent": "health_info"}
    llm = ScriptedLLM([ext], [call("predict_diabetes_risk", age=50, glucose=148, bmi=33.6), LLMResult(content="Done.")])
    reply, _ = run(HealthAgent(llm), Session(), "msg")
    assert reply.status == "result"
    assert reply.inputs_used.blood_pressure == 85


def test_missing_required_field_asks_follow_up():
    llm = ScriptedLLM([{"age": 45, "glucose": 160, "intent": "health_info"}], [LLMResult(content="What is your BMI?")])
    reply, _ = run(HealthAgent(llm), Session(), "45, sugar 160")
    assert reply.status == "need_info" and reply.missing_fields == ["bmi"]


def test_form_values_bypass_extraction_errors():
    llm = ScriptedLLM([{"intent": "health_info"}], [call("predict_diabetes_risk", age=40, glucose=120, bmi=27), LLMResult(content="ok")])
    reply, _ = run_values(HealthAgent(llm), Session(), "Age 40, glucose 120, BMI 27", {"age": 40, "glucose": 120, "bmi": 27})
    assert reply.status == "result" and reply.inputs_used.age == 40


def run_values(agent, session, msg, values):
    return asyncio.run(agent.handle(session, msg, values=values))


def test_out_of_range_not_merged_and_flagged():
    llm = ScriptedLLM([{"age": 45, "bmi": 200, "intent": "health_info"}], [LLMResult(content="BMI 200 looks wrong, please re-check.")])
    session = Session()
    reply, _ = run(HealthAgent(llm), session, "45, bmi 200")
    assert reply.status == "invalid_input" and reply.invalid_fields == ["bmi"]
    assert session.inputs.bmi is None and session.inputs.age == 45


def test_step_cap_stops_infinite_tool_loop():
    bad = call("predict_diabetes_risk", age=1, glucose=50, bmi=11, insulin=1)  # never matches user values
    llm = ScriptedLLM([FULL], [bad] * 10)
    reply, metrics = run(HealthAgent(llm), Session(), "msg")
    assert metrics.llm_calls == 1 + MAX_STEPS
    assert reply.status != "result" and reply.explanation


def test_unsupported_numeric_claim_is_blocked():
    llm = ScriptedLLM([FULL], [LLMResult(content="Your risk is 78%."), LLMResult(content="Your risk is 78%.")])
    reply, metrics = run(HealthAgent(llm), Session(), "msg")
    assert "78%" not in reply.explanation
    assert metrics.blocked_unsupported_results >= 1


def test_invalid_json_extraction_retried_once():
    llm = ScriptedLLM(["not json at all", FULL], [LLMResult(content="Need nothing else.")])
    session = Session()
    _, metrics = run(HealthAgent(llm), session, "msg")
    assert metrics.extraction_retries == 1
    assert session.inputs.age == 50


def test_off_topic_gets_no_tools_and_refused():
    seen = {}

    class Spy(ScriptedLLM):
        async def chat(self, messages, tools=None, json_mode=False):
            if not json_mode:
                seen["tools"] = tools
            return await super().chat(messages, tools, json_mode)

    llm = Spy([{"intent": "off_topic"}], [LLMResult(content="Sorry, I can only help with diabetes screening.")])
    reply, _ = run(HealthAgent(llm), Session(), "tell me a joke")
    assert reply.status == "refused" and seen["tools"] is None


def test_llm_outage_returns_error_reply():
    class Down:
        name = "down"

        async def chat(self, *a, **k):
            raise LLMError("timeout")

    reply, _ = run(HealthAgent(Down()), Session(), "hi")
    assert reply.status == "error"


def test_what_if_memory_with_rule_baseline():
    agent, s = HealthAgent(RuleBasedClient()), Session()
    r1, _ = run(agent, s, "I'm 45, sugar 160, BMI 29, insulin 120")
    assert r1.status == "result"
    r2, _ = run(agent, s, "what if my BMI was 25?")
    assert r2.status == "result" and r2.what_if
    assert r2.inputs_used.bmi == 25 and r2.inputs_used.glucose == 160 and r2.inputs_used.age == 45
    assert s.inputs.bmi == 29  # hypothetical change is not saved
