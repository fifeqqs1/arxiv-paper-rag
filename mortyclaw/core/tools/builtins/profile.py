from __future__ import annotations

import os


def save_user_profile_impl(
    new_content: str,
    *,
    get_memory_store_fn,
    build_memory_record_fn,
    memory_dir: str,
    profile_path: str,
    default_long_term_scope: str,
    user_profile_memory_id: str,
    user_profile_memory_type: str,
) -> str:
    get_memory_store_fn().upsert_memory(build_memory_record_fn(
        memory_id=user_profile_memory_id,
        layer="long_term",
        scope=default_long_term_scope,
        type=user_profile_memory_type,
        content=new_content,
        source_kind="manual_tool",
        source_ref="save_user_profile",
        confidence=1.0,
    ))

    os.makedirs(memory_dir, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as handle:
        handle.write(new_content)

    return "记忆档案已成功覆写更新。新的人设画像已生效。"
