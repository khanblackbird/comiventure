"""Tests for the Emitter base class — event emission, caching, propagation."""

import pytest
from backend.models.emitter import Emitter


class ChildEmitter(Emitter):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def _own_context(self):
        return {"name": self.name}


class TestEmission:
    def test_emit_calls_listener(self):
        emitter = Emitter()
        received = []
        emitter.on("test", lambda data: received.append(data))
        emitter.emit("test", "hello")
        assert received == ["hello"]

    def test_emit_multiple_listeners(self):
        emitter = Emitter()
        received = []
        emitter.on("test", lambda data: received.append("a"))
        emitter.on("test", lambda data: received.append("b"))
        emitter.emit("test", None)
        assert received == ["a", "b"]

    def test_off_removes_listener(self):
        emitter = Emitter()
        received = []
        callback = lambda data: received.append(data)
        emitter.on("test", callback)
        emitter.off("test", callback)
        emitter.emit("test", "hello")
        assert received == []

    def test_emit_up_propagates_to_parent(self):
        parent = Emitter()
        child = Emitter()
        child.set_parent(parent)

        received = []
        parent.on("update", lambda data: received.append("parent"))
        child.on("update", lambda data: received.append("child"))
        child.emit_up("update", None)

        assert "child" in received
        assert "parent" in received

    def test_emit_up_chain_three_levels(self):
        grandparent = Emitter()
        parent = Emitter()
        child = Emitter()
        parent.set_parent(grandparent)
        child.set_parent(parent)

        received = []
        grandparent.on("ping", lambda d: received.append("gp"))
        parent.on("ping", lambda d: received.append("p"))
        child.on("ping", lambda d: received.append("c"))
        child.emit_up("ping", None)

        assert received == ["c", "p", "gp"]

    def test_no_emission_to_unrelated_event(self):
        emitter = Emitter()
        received = []
        emitter.on("other", lambda d: received.append(d))
        emitter.emit("test", "hello")
        assert received == []


class TestContextCaching:
    def test_context_from_parent(self):
        parent = ChildEmitter("parent")
        child = ChildEmitter("child")
        child.set_parent(parent)

        context = child.get_context()
        # Child's context overwrites parent's "name" key
        assert context["name"] == "child"

    def test_context_cached_on_second_call(self):
        emitter = ChildEmitter("test")
        context_first = emitter.get_context()
        context_second = emitter.get_context()
        assert context_first is context_second

    def test_cache_invalidated_on_emit_up(self):
        parent = ChildEmitter("parent")
        child = ChildEmitter("child")
        child.set_parent(parent)

        context_before = child.get_context()
        child.emit_up("update", None)
        context_after = child.get_context()
        assert context_before is not context_after

    def test_cache_invalidated_on_set_parent(self):
        child = ChildEmitter("child")
        context_before = child.get_context()

        parent = ChildEmitter("parent")
        child.set_parent(parent)
        context_after = child.get_context()
        assert context_before is not context_after
