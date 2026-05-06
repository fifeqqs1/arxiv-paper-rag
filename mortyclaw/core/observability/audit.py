import os
import json
import threading
import queue
import atexit
from datetime import datetime, timezone

from ..config import LOGS_DIR


def sanitize_thread_id(thread_id: str) -> str:
    return "".join(c for c in (thread_id or "") if c.isalnum() or c in "-_") or "default"


def build_log_file_path(thread_id: str, log_dir: str = LOGS_DIR) -> str:
    return os.path.join(log_dir, f"{sanitize_thread_id(thread_id)}.jsonl")

# 内存队列 + 守护线程
class JSONLEventLogger:
    # 单例模式
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = LOGS_DIR):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_logger(log_dir)
            return cls._instance
        
    def _init_logger(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 无界内存队列，用于缓冲日志事件
        self.log_queue = queue.Queue()

        self.worker_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.worker_thread.start()

        # 确保程序被关闭时，队列里的剩下日志能写完
        atexit.register(self.shutdown)

    # 后台线程的死循环：一直盯着队列，有日志就写，没日志就阻塞休眠
    def _write_loop(self):
        while True:
            log_item = self.log_queue.get()

            if log_item is None:
                self.log_queue.task_done()
                break

            try:
                thread_id = log_item.get("thread_id", "system")
                file_path = build_log_file_path(thread_id, self.log_dir)

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Logger Error] 异步写日志失败: {e}")
            finally:
                self.log_queue.task_done()

    # 前台调用的埋点方法
    def log_event(self, thread_id: str, event: str, **kwargs):
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log_item = {
            "ts": now_utc,
            "thread_id": thread_id,
            "event": event,
            **kwargs
        }

        self.log_queue.put(log_item)

    def shutdown(self):
        self.log_queue.put(None)
        self.log_queue.join()

audit_logger = JSONLEventLogger()
