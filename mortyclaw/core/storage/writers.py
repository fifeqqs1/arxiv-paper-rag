from __future__ import annotations

import atexit
import queue
import threading
from typing import Any


class AsyncConversationWriter:
    def __init__(self, repository):
        self.repository = repository
        self.queue: queue.Queue[dict | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def append_messages(
        self,
        *,
        thread_id: str,
        turn_id: str,
        messages: list[Any],
        node_name: str = "",
        route: str = "",
    ) -> None:
        if self._closed or not messages:
            return
        self.queue.put({
            "kind": "append_messages",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "messages": list(messages),
            "node_name": node_name,
            "route": route,
        })

    def record_summary(
        self,
        *,
        thread_id: str,
        summary: str,
        summary_type: str = "context_compression",
        messages: list[Any] | None = None,
        metadata: dict | None = None,
    ) -> None:
        if self._closed:
            return
        self.queue.put({
            "kind": "record_summary",
            "thread_id": thread_id,
            "summary": summary,
            "summary_type": summary_type,
            "messages": list(messages or []),
            "metadata": metadata or {},
        })

    def flush(self) -> None:
        self.queue.join()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.queue.put(None)
        self.queue.join()

    def _write_loop(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if item.get("kind") == "append_messages":
                    self.repository.append_messages(
                        thread_id=item["thread_id"],
                        turn_id=item["turn_id"],
                        messages=item["messages"],
                        node_name=item.get("node_name", ""),
                        route=item.get("route", ""),
                    )
                elif item.get("kind") == "record_summary":
                    self.repository.record_conversation_summary(
                        thread_id=item["thread_id"],
                        summary=item["summary"],
                        summary_type=item.get("summary_type", "context_compression"),
                        messages=item.get("messages") or [],
                        metadata=item.get("metadata") or {},
                    )
            except Exception as exc:
                print(f"[ConversationStore Error] 异步写入失败: {exc}")
            finally:
                self.queue.task_done()
