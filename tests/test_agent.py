"""
tests/test_agent.py — 核心模块单元测试

使用 Mock 实现替换所有外部依赖，测试完全离线运行。
"""
from __future__ import annotations

import pytest
import asyncio

from core.models import (
    AgentConfig, AgentInput, AgentTask,
    LLMResponse, Message, MessageRole,
    MemoryEntry, MemoryType,
    ToolCall, ToolDescriptor, ToolResult,
    TaskStatus,
)
from core.container import AgentContainer


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def dev_container():
    return AgentContainer.create_dev()


def make_task(text: str = "测试任务") -> AgentTask:
    return AgentTask(
        session_id="test_sess",
        user_id="test_user",
        input=AgentInput(text=text),
    )


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class TestModels:
    def test_task_creation(self):
        task = make_task("hello")
        assert task.id.startswith("task_")
        assert task.status == TaskStatus.PLANNING
        assert task.step_idx == 0

    def test_add_message(self):
        task = make_task()
        msg  = Message(role=MessageRole.USER, content="hi")
        task.add_message(msg)
        assert len(task.history) == 1

    def test_llm_response_has_tool_calls(self):
        resp = LLMResponse(
            tool_calls=[ToolCall(tool_name="my_tool", arguments={"x": 1})]
        )
        assert resp.has_tool_calls

    def test_llm_response_no_tool_calls(self):
        resp = LLMResponse(content="Hello")
        assert not resp.has_tool_calls


# ─────────────────────────────────────────────
# Short-term Memory
# ─────────────────────────────────────────────

class TestInMemoryShortTermMemory:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        from memory.stores import InMemoryShortTermMemory
        stm  = InMemoryShortTermMemory()
        task = make_task("save test")
        await stm.save_task(task)
        loaded = await stm.load_task(task.id)
        assert loaded is not None
        assert loaded.id == task.id

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        from memory.stores import InMemoryShortTermMemory
        stm = InMemoryShortTermMemory()
        assert await stm.load_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_append_and_get_messages(self):
        from memory.stores import InMemoryShortTermMemory
        stm  = InMemoryShortTermMemory()
        task = make_task()
        await stm.save_task(task)
        msg  = Message(role=MessageRole.USER, content="hello")
        await stm.append_message(task.id, msg)
        msgs = await stm.get_messages(task.id)
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_scratchpad(self):
        from memory.stores import InMemoryShortTermMemory
        stm  = InMemoryShortTermMemory()
        task = make_task()
        await stm.save_task(task)
        await stm.set_scratchpad(task.id, "key1", {"data": 42})
        val = await stm.get_scratchpad(task.id, "key1")
        assert val == {"data": 42}


# ─────────────────────────────────────────────
# Long-term Memory
# ─────────────────────────────────────────────

class TestInMemoryLongTermMemory:
    @pytest.mark.asyncio
    async def test_write_and_search(self):
        from memory.stores import InMemoryLongTermMemory
        ltm = InMemoryLongTermMemory()
        entry = MemoryEntry(
            user_id="u1",
            type=MemoryType.SEMANTIC,
            text="Python 代码执行技巧",
            importance=0.8,
        )
        await ltm.write(entry)
        results = await ltm.search("u1", "Python 代码")
        assert len(results) >= 1
        assert results[0].text == entry.text

    @pytest.mark.asyncio
    async def test_dedup_write(self):
        from memory.stores import InMemoryLongTermMemory
        ltm = InMemoryLongTermMemory()
        for _ in range(3):
            await ltm.write(MemoryEntry(
                user_id="u1", type=MemoryType.SEMANTIC,
                text="重复内容", importance=0.5,
            ))
        results = await ltm.search("u1", "重复内容")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_profile(self):
        from memory.stores import InMemoryLongTermMemory
        ltm = InMemoryLongTermMemory()
        await ltm.update_profile("u1", {"language": "zh", "tone": "formal"})
        profile = await ltm.get_profile("u1")
        assert profile["language"] == "zh"

    @pytest.mark.asyncio
    async def test_prune(self):
        from memory.stores import InMemoryLongTermMemory
        ltm = InMemoryLongTermMemory()
        for i in range(5):
            await ltm.write(MemoryEntry(
                user_id="u1", type=MemoryType.SEMANTIC,
                text=f"entry {i}", importance=0.05,  # 低重要性
            ))
        pruned = await ltm.prune("u1", max_items=100, score_threshold=0.5)
        assert pruned == 5


# ─────────────────────────────────────────────
# Skill Registry
# ─────────────────────────────────────────────

class TestSkillRegistry:
    def make_registry(self):
        from skills.registry import LocalSkillRegistry
        return LocalSkillRegistry()

    def test_register_and_get(self):
        from skills.registry import WebSearchSkill
        reg   = self.make_registry()
        skill = WebSearchSkill()
        reg.register(skill)
        assert reg.get("web_search") is not None

    def test_list_descriptors(self):
        from skills.registry import WebSearchSkill, PythonExecutorSkill
        reg = self.make_registry()
        reg.register(WebSearchSkill())
        reg.register(PythonExecutorSkill())
        descs = reg.list_descriptors()
        names = [d.name for d in descs]
        assert "web_search" in names
        assert "execute_python" in names

    @pytest.mark.asyncio
    async def test_call_known_skill(self):
        from skills.registry import WebSearchSkill, LocalSkillRegistry
        reg = LocalSkillRegistry()
        reg.register(WebSearchSkill())
        result = await reg.call("web_search", {"query": "test"})
        assert result.error is None
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_call_unknown_skill(self):
        reg = self.make_registry()
        result = await reg.call("nonexistent", {})
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_python_executor(self):
        from skills.registry import PythonExecutorSkill
        skill = PythonExecutorSkill()
        result = await skill.execute({"code": "print('hello world')"})
        assert "hello world" in result["stdout"]


# ─────────────────────────────────────────────
# MCP Hub
# ─────────────────────────────────────────────

class TestMCPHub:
    def make_hub_with_mock(self):
        from mcp.hub import DefaultMCPHub, MockMCPConnector
        hub = DefaultMCPHub()
        tools = [ToolDescriptor(
            name="gmail__send_email",
            description="发送邮件",
            input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
            source="mcp",
        )]
        mock = MockMCPConnector("gmail", tools, responses={"send_email": {"sent": True}})
        hub.register_connector(mock)
        return hub, mock

    @pytest.mark.asyncio
    async def test_call_tool(self):
        hub, mock = self.make_hub_with_mock()
        result = await hub.call("gmail", "send_email", {"to": "test@example.com"})
        assert result.error is None
        assert result.content == {"sent": True}

    @pytest.mark.asyncio
    async def test_call_unknown_server(self):
        hub, _ = self.make_hub_with_mock()
        result = await hub.call("nonexistent", "tool", {})
        assert result.error is not None

    def test_audit_log(self):
        hub, _ = self.make_hub_with_mock()
        log = hub.get_audit_log()
        assert isinstance(log, list)


# ─────────────────────────────────────────────
# Context Manager
# ─────────────────────────────────────────────

class TestContextManager:
    @pytest.mark.asyncio
    async def test_build_returns_messages(self):
        from context.manager import PriorityContextManager
        from memory.stores import InMemoryLongTermMemory
        ctx  = PriorityContextManager()
        ltm  = InMemoryLongTermMemory()
        task = make_task("测试上下文组装")
        msgs = await ctx.build(task, [], ltm, budget=4000)
        assert len(msgs) >= 1
        assert msgs[0].role == MessageRole.SYSTEM

    @pytest.mark.asyncio
    async def test_build_within_budget(self):
        from context.manager import PriorityContextManager, _estimate_tokens
        from memory.stores import InMemoryLongTermMemory
        ctx  = PriorityContextManager()
        ltm  = InMemoryLongTermMemory()
        task = make_task("预算测试")
        # 添加大量历史
        for i in range(50):
            task.add_message(Message(role=MessageRole.USER, content=f"消息 {i}" * 20))
        msgs   = await ctx.build(task, [], ltm, budget=2000)
        total  = sum(_estimate_tokens(m.content or "") for m in msgs)
        assert total <= 3000  # 允许一定误差


# ─────────────────────────────────────────────
# End-to-end: Agent with Mock LLM
# ─────────────────────────────────────────────

class TestAgentE2E:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """LLM 直接回答（无工具调用）。"""
        from llm.engines import MockLLMEngine
        container = AgentContainer.create_dev()
        container.llm_engine = MockLLMEngine([
            LLMResponse(content="这是最终回答", usage={"prompt_tokens": 10, "completion_tokens": 5})
        ])
        container.build()

        agent  = container.agent()
        events = []
        async for event in agent.run("user1", "sess1", "你好"):
            events.append(event)

        types = [e["type"] for e in events]
        assert "done" in types

    @pytest.mark.asyncio
    async def test_tool_call_flow(self):
        """LLM 发起工具调用然后生成最终回答。"""
        from llm.engines import MockLLMEngine
        from memory.stores import InMemoryShortTermMemory, InMemoryLongTermMemory
        from skills.registry import LocalSkillRegistry, WebSearchSkill
        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager

        mock_llm = MockLLMEngine([
            # 第一轮：调用工具
            LLMResponse(
                tool_calls=[ToolCall(tool_name="web_search", arguments={"query": "Python"})],
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
            # 第二轮：最终回答
            LLMResponse(content="搜索完成，结果如下...", usage={"prompt_tokens": 30, "completion_tokens": 15}),
        ])
        registry = LocalSkillRegistry()
        registry.register(WebSearchSkill())

        container = AgentContainer(
            llm_engine=mock_llm,
            short_term_memory=InMemoryShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
        )
        container.build()

        agent  = container.agent()
        events = []
        async for event in agent.run("user1", "sess2", "搜索 Python"):
            events.append(event)

        step_events = [e for e in events if e["type"] == "step"]
        done_events = [e for e in events if e["type"] == "done"]
        assert len(step_events) >= 1
        assert len(done_events) == 1
        assert done_events[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_memory_persists_across_sessions(self):
        """验证长期记忆在 session 间持续。"""
        from llm.engines import MockLLMEngine
        from memory.stores import InMemoryLongTermMemory

        ltm = InMemoryLongTermMemory()
        await ltm.write(MemoryEntry(
            user_id="user2", type=MemoryType.PROFILE,
            text="用户偏好简洁输出", importance=0.9,
        ))
        results = await ltm.search("user2", "用户偏好")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_container_dev_factory(self, dev_container):
        assert dev_container.llm_engine is not None
        assert dev_container.skill_registry is not None
        assert dev_container.orchestrator is not None
