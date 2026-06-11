"""
会话存储模块 - 采用 OpenClaw 架构
Index(JSON) + Log(JSONL) 双层存储
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class MessageLog:
    """JSONL 日志操作类 - 追加写入，流式读取"""

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def append(self, entry: Dict[str, Any]) -> None:
        """追加一条消息到日志"""
        entry.setdefault("timestamp", datetime.now().isoformat())
        line = json.dumps(entry, ensure_ascii=False)
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def read_all(self) -> List[Dict[str, Any]]:
        """读取所有消息"""
        messages = []
        if not self.file_path.exists():
            return messages

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
        except Exception as e:
            logger.error(f"❌ 读取日志失败 {self.file_path}: {e}")

        return messages

    def read_last(self, n: int = 10) -> List[Dict[str, Any]]:
        """读取最后 n 条消息（从文件末尾倒读）"""
        messages = []
        if not self.file_path.exists():
            return messages

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines[-n:]):
                    line = line.strip()
                    if line:
                        messages.insert(0, json.loads(line))
        except Exception as e:
            logger.error(f"❌ 读取日志失败 {self.file_path}: {e}")

        return messages

    def read_by_type(self, message_type: str) -> List[Dict[str, Any]]:
        """按类型读取消息"""
        messages = []
        if not self.file_path.exists():
            return messages

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if entry.get("type") == message_type:
                            messages.append(entry)
        except Exception as e:
            logger.error(f"❌ 读取日志失败 {self.file_path}: {e}")

        return messages

    def count(self) -> int:
        """统计消息总数"""
        if not self.file_path.exists():
            return 0

        count = 0
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.error(f"❌ 统计日志失败 {self.file_path}: {e}")

        return count

    def delete(self) -> bool:
        """删除日志文件"""
        if self.file_path.exists():
            self.file_path.unlink()
            logger.info(f"🗑️ 日志文件已删除: {self.file_path}")
            return True
        return False


class SessionIndex:
    """会话索引管理类"""

    def __init__(self, index_path: Path):
        self.index_path = index_path

    def _load(self) -> Dict[str, Any]:
        """加载索引"""
        if not self.index_path.exists():
            return {
                "user_id": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "sessions": [],
                "by_conversation": {},
                "by_tag": {}
            }

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载索引失败 {self.index_path}: {e}")
            return {
                "user_id": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "sessions": [],
                "by_conversation": {},
                "by_tag": {}
            }

    def _save(self, index: Dict[str, Any]) -> None:
        """保存索引"""
        index["updated_at"] = datetime.now().isoformat()
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def add_session(self, session: Dict[str, Any]) -> None:
        """添加会话到索引"""
        index = self._load()
        session_id = session["id"]

        # 检查是否已存在
        existing = next((s for s in index["sessions"] if s["id"] == session_id), None)
        if existing:
            # 更新现有会话
            existing.update(session)
        else:
            # 添加新会话
            index["sessions"].append(session)

        # 更新 by_conversation 索引
        if "conversation_id" in session:
            index["by_conversation"][str(session["conversation_id"])] = session_id

        # 更新 by_tag 索引
        for tag in session.get("tags", []):
            if tag not in index["by_tag"]:
                index["by_tag"][tag] = []
            if session_id not in index["by_tag"][tag]:
                index["by_tag"][tag].append(session_id)

        self._save(index)
        logger.info(f"✅ 会话已添加到索引: {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        index = self._load()
        return next((s for s in index["sessions"] if s["id"] == session_id), None)

    def get_by_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """通过 conversation_id 获取会话"""
        index = self._load()
        session_id = index["by_conversation"].get(str(conversation_id))
        if session_id:
            return self.get_session(session_id)
        return None

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有会话"""
        index = self._load()
        return index["sessions"][-limit:]

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        index = self._load()

        # 从 sessions 列表移除
        session = next((s for s in index["sessions"] if s["id"] == session_id), None)
        if not session:
            return False

        index["sessions"].remove(session)

        # 从 by_conversation 移除
        if "conversation_id" in session:
            conv_id = str(session["conversation_id"])
            if index["by_conversation"].get(conv_id) == session_id:
                del index["by_conversation"][conv_id]

        # 从 by_tag 移除
        for tag in session.get("tags", []):
            if tag in index["by_tag"] and session_id in index["by_tag"][tag]:
                index["by_tag"][tag].remove(session_id)

        self._save(index)
        logger.info(f"🗑️ 会话已从索引删除: {session_id}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        index = self._load()
        total_messages = sum(s.get("message_count", 0) for s in index["sessions"])
        total_memories = sum(s.get("memory_count", 0) for s in index["sessions"])

        return {
            "total_sessions": len(index["sessions"]),
            "total_messages": total_messages,
            "total_memories": total_memories,
            "tags": list(index["by_tag"].keys())
        }


class SessionStorage:
    """会话存储主类 - 整合索引和日志"""

    def __init__(self, data_dir: str = "data/memories"):
        self.data_dir = Path(data_dir)

    def _get_user_dir(self, user_id: int) -> Path:
        """获取用户目录"""
        return self.data_dir / f"user_{user_id}"

    def _get_index_path(self, user_id: int) -> Path:
        """获取索引文件路径"""
        return self._get_user_dir(user_id) / "sessions.json"

    def _get_session_file(self, user_id: int, session_id: str) -> Path:
        """获取会话日志文件路径"""
        return self._get_user_dir(user_id) / f"{session_id}.jsonl"

    def _ensure_user_dir(self, user_id: int) -> None:
        """确保用户目录存在"""
        self._get_user_dir(user_id).mkdir(parents=True, exist_ok=True)

    def create_session(self,
                      user_id: int,
                      conversation_id: Optional[int] = None,
                      model: str = "default",
                      tags: List[str] = None) -> Dict[str, Any]:
        """创建新会话"""
        self._ensure_user_dir(user_id)

        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{conversation_id or 'new'}"

        # 创建会话信息
        session = {
            "id": session_id,
            "file": f"{session_id}.jsonl",
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
            "memory_count": 0,
            "model": model,
            "tags": tags or []
        }

        # 添加到索引
        index = SessionIndex(self._get_index_path(user_id))
        index.add_session(session)

        logger.info(f"✅ 新会话已创建: {session_id}")
        return session

    def append_message(self,
                      user_id: int,
                      session_id: str,
                      role: str,
                      content: str,
                      message_type: str = "message",
                      usage: Dict[str, int] = None) -> None:
        """追加消息到会话"""
        log_file = self._get_session_file(user_id, session_id)
        log = MessageLog(log_file)

        entry = {
            "type": message_type,
            "role": role,
            "content": content
        }

        if usage:
            entry["usage"] = usage

        log.append(entry)

        # 更新索引
        index = SessionIndex(self._get_index_path(user_id))
        session = index.get_session(session_id)
        if session:
            session["updated_at"] = datetime.now().isoformat()
            session["message_count"] = log.count()
            index.add_session(session)

    def append_memory(self,
                     user_id: int,
                     session_id: str,
                     content: str,
                     memory_type: str = "learned",
                     importance: int = 1) -> None:
        """追加记忆到会话"""
        log_file = self._get_session_file(user_id, session_id)
        log = MessageLog(log_file)

        entry = {
            "type": "memory",
            "memory_type": memory_type,
            "content": content,
            "importance": importance
        }

        log.append(entry)

        # 更新索引
        index = SessionIndex(self._get_index_path(user_id))
        session = index.get_session(session_id)
        if session:
            session["updated_at"] = datetime.now().isoformat()
            session["memory_count"] = len(log.read_by_type("memory"))
            index.add_session(session)

    def get_session_messages(self,
                            user_id: int,
                            session_id: str,
                            limit: int = None) -> List[Dict[str, Any]]:
        """获取会话消息"""
        log_file = self._get_session_file(user_id, session_id)
        log = MessageLog(log_file)
        messages = log.read_all()

        if limit:
            return messages[-limit:]
        return messages

    def get_session_memories(self,
                            user_id: int,
                            session_id: str) -> List[Dict[str, Any]]:
        """获取会话记忆"""
        log_file = self._get_session_file(user_id, session_id)
        log = MessageLog(log_file)
        return log.read_by_type("memory")

    def get_session(self, user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        index = SessionIndex(self._get_index_path(user_id))
        return index.get_session(session_id)

    def get_conversation_session(self, user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
        """获取对话对应的会话"""
        index = SessionIndex(self._get_index_path(user_id))
        return index.get_by_conversation(conversation_id)

    def update_session(self, user_id: int, session_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新会话信息"""
        index = SessionIndex(self._get_index_path(user_id))
        session = index.get_session(session_id)
        
        if not session:
            return None
        
        # 更新会话字段
        session.update(kwargs)
        session["updated_at"] = datetime.now().isoformat()
        index.add_session(session)
        
        logger.info(f"✅ 会话已更新: {session_id}")
        return session

    def delete_session(self, user_id: int, session_id: str) -> bool:
        """删除会话"""
        # 删除日志文件
        log_file = self._get_session_file(user_id, session_id)
        log = MessageLog(log_file)
        log.delete()

        # 从索引删除
        index = SessionIndex(self._get_index_path(user_id))
        return index.delete_session(session_id)

    def get_user_sessions(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """获取用户的所有会话"""
        index = SessionIndex(self._get_index_path(user_id))
        return index.list_sessions(limit)

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """获取用户统计"""
        index = SessionIndex(self._get_index_path(user_id))
        return index.get_stats()