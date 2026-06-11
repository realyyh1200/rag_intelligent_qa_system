"""
数据迁移脚本 - 将旧的记忆格式迁移到新的 JSONL 架构
旧格式: data/memories/user_{id}/mem_*.json + index.json
新格式: data/memories/user_{id}/sessions.json + sess_*.jsonl
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging
import sys
import importlib.util

# 直接加载 session_storage 模块，避免依赖 services/__init__.py
spec = importlib.util.spec_from_file_location(
    "session_storage",
    Path(__file__).parent / "services" / "session_storage.py"
)
session_storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_storage_module)
SessionStorage = session_storage_module.SessionStorage

logger = logging.getLogger(__name__)


class MemoryMigrator:
    """记忆数据迁移器"""

    def __init__(self, old_data_dir: str = "data/memories"):
        self.old_data_dir = Path(old_data_dir)
        self.new_data_dir = Path(old_data_dir)  # 同一目录
        self.session_storage = SessionStorage(old_data_dir)

    def _load_old_index(self, user_id: int) -> Dict[str, Any]:
        """加载旧索引或扫描记忆文件"""
        index_path = self.old_data_dir / f"user_{user_id}" / "index.json"

        # 如果索引文件存在，直接加载
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ 加载旧索引失败: {e}")

        # 否则扫描记忆文件
        user_dir = self.old_data_dir / f"user_{user_id}"
        if not user_dir.exists():
            return {"memories": {}}

        memories = {}
        for mem_file in user_dir.glob("mem_*.json"):
            try:
                with open(mem_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    memory_id = memory.get('id')
                    if memory_id:
                        memories[memory_id] = {
                            "memory_type": memory.get("memory_type", "general"),
                            "importance": memory.get("importance", 1),
                            "conversation_id": memory.get("conversation_id"),
                            "created_at": memory.get("created_at", ""),
                            "keywords": memory.get("keywords", []),
                            "content_hash": memory.get("content_hash", "")
                        }
            except Exception as e:
                logger.error(f"❌ 读取记忆文件失败 {mem_file}: {e}")

        return {"memories": memories}

    def _load_old_memory(self, user_id: int, memory_id: str) -> Dict[str, Any]:
        """加载旧记忆文件"""
        memory_path = self.old_data_dir / f"user_{user_id}" / f"{memory_id}.json"
        if not memory_path.exists():
            return None

        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载旧记忆失败 {memory_id}: {e}")
            return None

    def _backup_old_data(self, user_id: int) -> Path:
        """备份旧数据"""
        user_dir = self.old_data_dir / f"user_{user_id}"
        if not user_dir.exists():
            return None

        backup_dir = self.old_data_dir / f"user_{user_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(user_dir, backup_dir)
        logger.info(f"💾 旧数据已备份到: {backup_dir}")
        return backup_dir

    def _group_memories_by_conversation(self, memories: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """按 conversation_id 分组记忆"""
        grouped = {}
        for memory in memories:
            conv_id = memory.get("conversation_id")
            if conv_id is not None:
                if conv_id not in grouped:
                    grouped[conv_id] = []
                grouped[conv_id].append(memory)
        return grouped

    def migrate_user(self, user_id: int, dry_run: bool = False) -> Dict[str, Any]:
        """迁移单个用户的数据"""
        logger.info(f"🚀 开始迁移用户 {user_id} 的数据...")

        # 加载旧索引
        old_index = self._load_old_index(user_id)
        old_memories_info = old_index.get("memories", {})

        if not old_memories_info:
            logger.info(f"⚠️ 用户 {user_id} 没有旧数据，跳过迁移")
            return {"status": "skipped", "reason": "no_old_data"}

        # 备份旧数据
        if not dry_run:
            backup_dir = self._backup_old_data(user_id)

        # 加载所有旧记忆
        all_memories = []
        for memory_id, mem_info in old_memories_info.items():
            memory = self._load_old_memory(user_id, memory_id)
            if memory:
                all_memories.append(memory)

        logger.info(f"📦 加载了 {len(all_memories)} 条旧记忆")

        # 按 conversation_id 分组
        grouped = self._group_memories_by_conversation(all_memories)
        logger.info(f"📊 记忆分布在 {len(grouped)} 个对话中")

        # 迁移统计
        stats = {
            "total_memories": len(all_memories),
            "total_conversations": len(grouped),
            "migrated_sessions": 0,
            "migrated_memories": 0,
            "failed": 0
        }

        # 为每个对话创建会话并迁移记忆
        for conv_id, memories in grouped.items():
            try:
                # 创建新会话
                session = self.session_storage.create_session(
                    user_id=user_id,
                    conversation_id=conv_id,
                    model="migrated",
                    tags=["migrated"]
                )
                session_id = session["id"]

                # 迁移记忆
                for memory in memories:
                    if not dry_run:
                        self.session_storage.append_memory(
                            user_id=user_id,
                            session_id=session_id,
                            content=memory.get("content", ""),
                            memory_type=memory.get("memory_type", "general"),
                            importance=memory.get("importance", 1)
                        )
                    stats["migrated_memories"] += 1

                stats["migrated_sessions"] += 1
                logger.info(f"✅ 对话 {conv_id} 已迁移到会话 {session_id} ({len(memories)} 条记忆)")

            except Exception as e:
                logger.error(f"❌ 迁移对话 {conv_id} 失败: {e}")
                stats["failed"] += 1

        # 迁移没有 conversation_id 的记忆
        orphan_memories = [m for m in all_memories if m.get("conversation_id") is None]
        if orphan_memories:
            try:
                session = self.session_storage.create_session(
                    user_id=user_id,
                    conversation_id=None,
                    model="migrated",
                    tags=["migrated", "orphan"]
                )
                session_id = session["id"]

                for memory in orphan_memories:
                    if not dry_run:
                        self.session_storage.append_memory(
                            user_id=user_id,
                            session_id=session_id,
                            content=memory.get("content", ""),
                            memory_type=memory.get("memory_type", "general"),
                            importance=memory.get("importance", 1)
                        )
                    stats["migrated_memories"] += 1

                stats["migrated_sessions"] += 1
                logger.info(f"✅ {len(orphan_memories)} 条孤立记忆已迁移到会话 {session_id}")

            except Exception as e:
                logger.error(f"❌ 迁移孤立记忆失败: {e}")
                stats["failed"] += 1

        # 删除旧数据（如果不是 dry_run）
        if not dry_run:
            user_dir = self.old_data_dir / f"user_{user_id}"
            # 删除旧的 .json 文件和 index.json
            for file in user_dir.glob("mem_*.json"):
                file.unlink()
            index_file = user_dir / "index.json"
            if index_file.exists():
                index_file.unlink()
            logger.info(f"🗑️ 旧数据文件已删除")

        logger.info(f"✅ 用户 {user_id} 迁移完成: {stats}")
        return {"status": "success", "stats": stats}

    def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """迁移所有用户的数据"""
        logger.info(f"🚀 开始迁移所有用户数据... (dry_run={dry_run})")

        if not self.old_data_dir.exists():
            logger.error(f"❌ 数据目录不存在: {self.old_data_dir}")
            return {"status": "error", "reason": "data_dir_not_found"}

        # 找到所有用户目录
        user_dirs = [d for d in self.old_data_dir.iterdir() if d.is_dir() and d.name.startswith("user_")]
        logger.info(f"📂 找到 {len(user_dirs)} 个用户目录")

        results = {}
        for user_dir in user_dirs:
            try:
                user_id = int(user_dir.name.replace("user_", ""))
                results[user_id] = self.migrate_user(user_id, dry_run)
            except Exception as e:
                logger.error(f"❌ 迁移用户 {user_dir.name} 失败: {e}")
                results[user_dir.name] = {"status": "error", "reason": str(e)}

        # 汇总统计
        total_memories = sum(r.get("stats", {}).get("migrated_memories", 0) for r in results.values())
        total_sessions = sum(r.get("stats", {}).get("migrated_sessions", 0) for r in results.values())

        summary = {
            "status": "completed",
            "total_users": len(user_dirs),
            "total_memories": total_memories,
            "total_sessions": total_sessions,
            "dry_run": dry_run,
            "results": results
        }

        logger.info(f"✅ 迁移完成: {summary}")
        return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    migrator = MemoryMigrator()

    # 检查命令行参数
    dry_run = "--dry-run" in sys.argv
    user_id = None

    for arg in sys.argv:
        if arg.startswith("--user="):
            user_id = int(arg.split("=")[1])

    if user_id:
        # 迁移单个用户
        result = migrator.migrate_user(user_id, dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 迁移所有用户
        result = migrator.migrate_all(dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))