from __future__ import annotations

from datetime import datetime


def ensure_task_store_bootstrapped_impl(
    *,
    get_task_repository_fn,
    file_path: str,
) -> None:
    get_task_repository_fn().bootstrap_legacy_tasks(
        file_path=file_path,
        default_thread_id="local_geek_master",
    )


def schedule_task_impl(
    *,
    target_time: str,
    description: str,
    repeat: str | None,
    repeat_count: int | None,
    ensure_task_store_bootstrapped_fn,
    get_active_session_thread_id_fn,
    ensure_session_record_fn,
    get_task_repository_fn,
    tasks_file: str,
) -> str:
    try:
        datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "设定失败：时间格式错误，必须严格遵循 'YYYY-MM-DD HH:MM:SS' 格式。"

    repeat_mode = (repeat or "").strip().lower() or None
    if repeat_mode not in {None, "hourly", "daily", "weekly"}:
        return "设定失败：repeat 只能是 hourly、daily、weekly 或留空。"

    try:
        ensure_task_store_bootstrapped_fn()
        thread_id = get_active_session_thread_id_fn()
        ensure_session_record_fn(thread_id)
        task_repository = get_task_repository_fn()
        task_repository.create_task(
            target_time=target_time,
            description=description,
            repeat=repeat_mode,
            repeat_count=repeat_count,
            thread_id=thread_id,
        )
        task_repository.sync_legacy_file(file_path=tasks_file)
    except Exception as exc:
        return f"设定失败：写入任务队列异常 {str(exc)}"

    msg = f" 任务已成功加入队列。首发时间：{target_time} | 任务：{description}"
    if repeat_mode:
        msg += f" | 循环模式：{repeat_mode} (共 {repeat_count if repeat_count else '无限'} 次)"
    return msg


def list_scheduled_tasks_impl(
    *,
    ensure_task_store_bootstrapped_fn,
    get_active_session_thread_id_fn,
    get_task_repository_fn,
) -> str:
    try:
        ensure_task_store_bootstrapped_fn()
        thread_id = get_active_session_thread_id_fn()
        tasks = get_task_repository_fn().list_tasks(
            thread_id=thread_id,
            statuses=("scheduled",),
            limit=500,
        )
        if not tasks:
            return "当前没有任何定时任务。"

        res = " 当前待执行任务列表：\n"
        for task in tasks:
            res += f"- [ID: {task['task_id']}] 时间: {task['target_time']} | 任务: {task['description']}\n"
        return res
    except Exception as exc:
        return f"查询失败：{str(exc)}"


def delete_scheduled_task_impl(
    *,
    task_id: str,
    ensure_task_store_bootstrapped_fn,
    get_active_session_thread_id_fn,
    get_task_repository_fn,
    tasks_file: str,
) -> str:
    try:
        ensure_task_store_bootstrapped_fn()
        thread_id = get_active_session_thread_id_fn()
        task_repository = get_task_repository_fn()
        deleted = task_repository.cancel_task(task_id, thread_id=thread_id)
        if deleted is None:
            return f"删除失败：未找到 ID 为 {task_id} 的任务。"
        task_repository.sync_legacy_file(file_path=tasks_file)
        return f" 任务 [ID: {task_id}] 已成功取消。"
    except Exception as exc:
        return f"操作异常：{str(exc)}"


def modify_scheduled_task_impl(
    *,
    task_id: str,
    new_time: str | None,
    new_description: str | None,
    ensure_task_store_bootstrapped_fn,
    get_active_session_thread_id_fn,
    get_task_repository_fn,
    tasks_file: str,
) -> str:
    try:
        ensure_task_store_bootstrapped_fn()
        thread_id = get_active_session_thread_id_fn()
        if new_time:
            datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")

        task_repository = get_task_repository_fn()
        updated = task_repository.update_task(
            task_id,
            thread_id=thread_id,
            new_time=new_time,
            new_description=new_description,
        )
        if updated is None:
            return f"修改失败：未找到 ID 为 {task_id} 的任务。"
        task_repository.sync_legacy_file(file_path=tasks_file)
        return f" 任务 [ID: {task_id}] 已成功更新。"
    except ValueError:
        return "修改失败：时间格式错误。"
    except Exception as exc:
        return f"操作异常：{str(exc)}"
