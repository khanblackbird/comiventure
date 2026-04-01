from __future__ import annotations

from typing import Any, Callable


class Emitter:
    """Base class for all domain objects. Provides event emission
    so objects communicate through signals, not direct calls.

    Children emit upward to parents. Parents inherit context downward.
    This separation allows any node to be driven by AI or manual input
    without changing the wiring.

    Performance:
    - Dirty-flag caching: context is only recomputed when stale.
      A change marks ancestors dirty (O(1)), context is rebuilt
      only when read (lazy).
    - emit_up invalidates caches as it propagates, so downstream
      reads always get fresh data without redundant recomputation.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}
        self._parent: Emitter | None = None
        self._context_cache: dict | None = None
        self._context_dirty: bool = True

    def on(self, event: str, callback: Callable) -> None:
        """Subscribe to an event on this object."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unsubscribe from an event."""
        if event in self._listeners:
            self._listeners[event] = [
                listener for listener in self._listeners[event]
                if listener is not callback
            ]

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all listeners on this object."""
        for callback in self._listeners.get(event, []):
            callback(data)

    def emit_up(self, event: str, data: Any = None) -> None:
        """Emit an event upward through the parent chain.
        Invalidates context caches as it goes — O(depth) but only
        touches the dirty flag, no recomputation.
        """
        self._invalidate_context()
        self.emit(event, data)
        if self._parent is not None:
            self._parent.emit_up(event, data)

    def _invalidate_context(self) -> None:
        """Mark this node's context cache as stale. O(1)."""
        self._context_dirty = True
        self._context_cache = None

    def set_parent(self, parent: Emitter) -> None:
        """Set the parent in the hierarchy. Context inherits downward from parent."""
        self._parent = parent
        self._invalidate_context()

    def require_parent(self, expected_type: type | None = None) -> Emitter:
        """Assert this object has a parent. Raises if orphaned."""
        if self._parent is None:
            raise RuntimeError(
                f"{type(self).__name__} is orphaned — must be added to a parent via "
                f"the parent's add/create method, not constructed directly"
            )
        if expected_type and not isinstance(self._parent, expected_type):
            raise RuntimeError(
                f"{type(self).__name__} expected parent of type {expected_type.__name__}, "
                f"got {type(self._parent).__name__}"
            )
        return self._parent

    @property
    def is_orphan(self) -> bool:
        return self._parent is None

    def get_context(self) -> dict:
        """Collect inherited context by walking up the parent chain.
        Cached — only recomputes when dirty. Each subclass contributes
        its own context. Children see everything above them.
        """
        if not self._context_dirty and self._context_cache is not None:
            return self._context_cache

        parent_context = self._parent.get_context() if self._parent else {}
        parent_context.update(self._own_context())
        self._context_cache = parent_context
        self._context_dirty = False
        return parent_context

    def _own_context(self) -> dict:
        """Override in subclasses to contribute context to children."""
        return {}
