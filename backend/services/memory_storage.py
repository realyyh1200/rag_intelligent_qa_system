"""
结构化JSON记忆存储系统

长期记忆以JSON格式存储在文件中，使用index.json索引加速查询。
"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from core.logger import logger


class MemoryStorage:
    """
    结构化JSON记忆存储
    
    文件结构:
    data_dir/
    ├── memories/
    │   ├── user_1/
    │   │   ├── index.json          # 索引文件
    │   │   ├── memory_001.json     # 单条记忆
    │   │   ├── memory_002.json
    │   │   └── ...
    │   └── user_2/
    │       └── ...
    """
    
    def __init__(self, data_dir: str = "data/memories"):
        self.data_dir = Path(data_dir)
        self.index_file = "index.json"
    
    def _get_user_dir(self, user_id: int) -> Path:
        """获取用户记忆目录"""
        return self.data_dir / f"user_{user_id}"
    
    def _get_index_path(self, user_id: int) -> Path:
        """获取索引文件路径"""
        return self._get_user_dir(user_id) / self.index_file
    
    def _ensure_user_dir(self, user_id: int) -> None:
        """确保用户目录存在"""
        user_dir = self._get_user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self, user_id: int) -> Dict[str, Any]:
        """加载索引文件"""
        index_path = self._get_index_path(user_id)
        
        if not index_path.exists():
            return {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "memory_count": 0,
                "memories": {},  # memory_id -> memory_index_entry
                "by_type": {},    # memory_type -> [memory_ids]
                "by_importance": {},  # importance -> [memory_ids]
                "by_conversation": {}  # conversation_id -> [memory_ids]
            }
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载索引失败: {e}")
            return {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "memory_count": 0,
                "memories": {},
                "by_type": {},
                "by_importance": {},
                "by_conversation": {}
            }
    
    def _save_index(self, user_id: int, index: Dict[str, Any]) -> None:
        """保存索引文件"""
        index_path = self._get_index_path(user_id)
        index["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 保存索引失败: {e}")
            raise
    
    def _load_memory(self, user_id: int, memory_id: str) -> Optional[Dict[str, Any]]:
        """加载单条记忆"""
        user_dir = self._get_user_dir(user_id)
        memory_path = user_dir / f"{memory_id}.json"
        
        if not memory_path.exists():
            return None
        
        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载记忆失败 {memory_id}: {e}")
            return None
    
    def _save_memory(self, user_id: int, memory: Dict[str, Any]) -> None:
        """保存单条记忆"""
        memory_id = memory.get('id')
        if not memory_id:
            raise ValueError("Memory must have an 'id' field")
        
        user_dir = self._get_user_dir(user_id)
        self._ensure_user_dir(user_id)
        
        memory_path = user_dir / f"{memory_id}.json"
        
        try:
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 保存记忆失败 {memory_id}: {e}")
            raise
    
    def _delete_memory_file(self, user_id: int, memory_id: str) -> None:
        """删除记忆文件"""
        user_dir = self._get_user_dir(user_id)
        memory_path = user_dir / f"{memory_id}.json"
        
        if memory_path.exists():
            memory_path.unlink()
    
    # ========== 公共API ==========
    
    def add(self, 
           user_id: int, 
           content: str, 
           memory_type: str = "general",
           importance: int = 1,
           conversation_id: Optional[int] = None,
           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加一条新记忆
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            memory_type: 记忆类型 (general, preference, fact, etc.)
            importance: 重要性 1-10
            conversation_id: 关联的对话ID
            metadata: 额外元数据
        
        Returns:
            创建的记忆对象
        """
        self._ensure_user_dir(user_id)
        index = self._load_index(user_id)
        
        # 生成记忆ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_{index['memory_count'] + 1:04d}"
        
        # 创建记忆对象
        memory = {
            "id": memory_id,
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "conversation_id": conversation_id,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None,
            "keywords": self._extract_keywords(content),  # 提取关键词用于快速检索
            "content_hash": str(hash(content))  # 用于去重
        }
        
        # 保存记忆文件
        self._save_memory(user_id, memory)
        
        # 更新索引
        index["memory_count"] += 1
        
        # 更新 memories 索引
        index["memories"][memory_id] = {
            "memory_type": memory_type,
            "importance": importance,
            "conversation_id": conversation_id,
            "created_at": memory["created_at"],
            "content_hash": memory["content_hash"],
            "keyword_count": len(memory["keywords"])
        }
        
        # 更新 by_type 索引
        if memory_type not in index["by_type"]:
            index["by_type"][memory_type] = []
        index["by_type"][memory_type].append(memory_id)
        
        # 更新 by_importance 索引
        importance_key = f"imp_{importance // 2}"  # 分组: 1-2, 3-4, 5-6, 7-8, 9-10
        if importance_key not in index["by_importance"]:
            index["by_importance"][importance_key] = []
        index["by_importance"][importance_key].append(memory_id)
        
        # 更新 by_conversation 索引
        if conversation_id:
            conv_key = str(conversation_id)
            if conv_key not in index["by_conversation"]:
                index["by_conversation"][conv_key] = []
            index["by_conversation"][conv_key].append(memory_id)
        
        # 保存索引
        self._save_index(user_id, index)
        
        logger.info(f"💾 记忆已保存: {memory_id} (type={memory_type}, importance={importance})")
        return memory
    
    def _extract_keywords(self, content: str) -> List[str]:
        """从内容中提取关键词（简单实现）"""
        # 简单分词 - 提取长度>=2的词
        words = []
        current_word = ""
        
        for char in content:
            if char.isalnum():
                current_word += char
            else:
                if len(current_word) >= 2:
                    words.append(current_word.lower())
                current_word = ""
        
        if len(current_word) >= 2:
            words.append(current_word.lower())
        
        # 去重并返回前20个
        unique_words = list(set(words))[:20]
        return unique_words
    
    def get(self, user_id: int, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆"""
        memory = self._load_memory(user_id, memory_id)
        
        if memory:
            # 更新访问统计
            self._update_access(user_id, memory_id)
        
        return memory
    
    def _update_access(self, user_id: int, memory_id: str) -> None:
        """更新记忆访问统计"""
        index = self._load_index(user_id)
        
        if memory_id in index["memories"]:
            index["memories"][memory_id]["access_count"] = index["memories"][memory_id].get("access_count", 0) + 1
            index["memories"][memory_id]["last_accessed"] = datetime.now().isoformat()
            self._save_index(user_id, index)
    
    def get_all(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户所有记忆"""
        index = self._load_index(user_id)
        
        memories = []
        for memory_id in index["memories"]:
            memory = self._load_memory(user_id, memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def search_by_keyword(self, user_id: int, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        通过关键词快速搜索记忆（使用索引加速）
        """
        index = self._load_index(user_id)
        keyword_lower = keyword.lower()
        
        matching_ids = []
        
        # 遍历所有记忆，查找关键词匹配
        for memory_id, mem_info in index["memories"].items():
            memory = self._load_memory(user_id, memory_id)
            if memory and keyword_lower in memory.get('keywords', []):
                matching_ids.append(memory_id)
                if len(matching_ids) >= limit:
                    break
        
        # 加载匹配的记忆
        results = []
        for memory_id in matching_ids:
            memory = self._load_memory(user_id, memory_id)
            if memory:
                results.append(memory)
        
        return results
    
    def search_by_type(self, user_id: int, memory_type: str) -> List[Dict[str, Any]]:
        """按类型获取记忆"""
        index = self._load_index(user_id)
        
        memory_ids = index["by_type"].get(memory_type, [])
        
        results = []
        for memory_id in memory_ids:
            memory = self._load_memory(user_id, memory_id)
            if memory:
                results.append(memory)
        
        return results
    
    def search_by_conversation(self, user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        """获取特定对话相关的记忆"""
        index = self._load_index(user_id)
        
        memory_ids = index["by_conversation"].get(str(conversation_id), [])
        
        results = []
        for memory_id in memory_ids:
            memory = self._load_memory(user_id, memory_id)
            if memory:
                results.append(memory)
        
        return results
    
    def get_recent(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的记忆"""
        index = self._load_index(user_id)
        
        # 按创建时间排序
        sorted_mems = sorted(
            index["memories"].items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        results = []
        for memory_id, _ in sorted_mems[:limit]:
            memory = self._load_memory(user_id, memory_id)
            if memory:
                results.append(memory)
        
        return results
    
    def get_important(self, user_id: int, min_importance: int = 7) -> List[Dict[str, Any]]:
        """获取重要记忆（重要性 >= min_importance）"""
        index = self._load_index(user_id)
        
        # 找出所有高重要性记忆
        memory_ids = []
        for imp_key, ids in index["by_importance"].items():
            try:
                imp_level = int(imp_key.split("_")[1])
                if imp_level * 2 >= min_importance:  # 近似匹配
                    memory_ids.extend(ids)
            except:
                pass
        
        results = []
        seen = set()
        for memory_id in memory_ids:
            if memory_id not in seen:
                seen.add(memory_id)
                memory = self._load_memory(user_id, memory_id)
                if memory and memory.get('importance', 0) >= min_importance:
                    results.append(memory)
        
        # 按重要性排序
        results.sort(key=lambda x: x.get('importance', 0), reverse=True)
        return results
    
    def delete(self, user_id: int, memory_id: str) -> bool:
        """删除记忆"""
        index = self._load_index(user_id)
        
        if memory_id not in index["memories"]:
            return False
        
        # 获取记忆信息
        mem_info = index["memories"][memory_id]
        
        # 删除记忆文件
        self._delete_memory_file(user_id, memory_id)
        
        # 更新索引 - memories
        del index["memories"][memory_id]
        index["memory_count"] -= 1
        
        # 更新索引 - by_type
        memory_type = mem_info.get("memory_type")
        if memory_type and memory_type in index["by_type"]:
            if memory_id in index["by_type"][memory_type]:
                index["by_type"][memory_type].remove(memory_id)
        
        # 更新索引 - by_conversation
        conversation_id = mem_info.get("conversation_id")
        if conversation_id:
            conv_key = str(conversation_id)
            if conv_key in index["by_conversation"] and memory_id in index["by_conversation"][conv_key]:
                index["by_conversation"][conv_key].remove(memory_id)
        
        # 保存索引
        self._save_index(user_id, index)
        
        logger.info(f"🗑️ 记忆已删除: {memory_id}")
        return True
    
    def update(self, user_id: int, memory_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新记忆"""
        memory = self._load_memory(user_id, memory_id)
        
        if not memory:
            return None
        
        # 更新字段
        allowed_fields = ['content', 'memory_type', 'importance', 'metadata']
        for key, value in updates.items():
            if key in allowed_fields:
                memory[key] = value
        
        memory['updated_at'] = datetime.now().isoformat()
        
        # 如果内容更新，重新提取关键词
        if 'content' in updates:
            memory['keywords'] = self._extract_keywords(updates['content'])
            memory['content_hash'] = str(hash(updates['content']))
        
        # 保存
        self._save_memory(user_id, memory)
        
        # 更新索引
        index = self._load_index(user_id)
        if memory_id in index["memories"]:
            for key in ['memory_type', 'importance', 'conversation_id']:
                if key in updates:
                    index["memories"][memory_id][key] = updates[key]
            if 'content' in updates:
                index["memories"][memory_id]["content_hash"] = memory['content_hash']
                index["memories"][memory_id]["keyword_count"] = len(memory['keywords'])
            self._save_index(user_id, index)
        
        logger.info(f"✏️ 记忆已更新: {memory_id}")
        return memory
    
    def get_stats(self, user_id: int) -> Dict[str, Any]:
        """获取记忆统计信息"""
        index = self._load_index(user_id)
        
        stats = {
            "total_memories": index["memory_count"],
            "by_type": {k: len(v) for k, v in index["by_type"].items()},
            "memory_types": list(index["by_type"].keys()),
        }
        
        return stats
    
    def optimize_index(self, user_id: int) -> None:
        """
        优化索引文件，重新构建索引
        用于修复可能的索引不一致问题
        """
        user_dir = self._get_user_dir(user_id)
        
        if not user_dir.exists():
            return
        
        # 收集所有记忆文件
        memory_files = list(user_dir.glob("mem_*.json"))
        
        # 重建索引
        new_index = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "memory_count": 0,
            "memories": {},
            "by_type": {},
            "by_importance": {},
            "by_conversation": {}
        }
        
        for mem_file in memory_files:
            try:
                with open(mem_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                
                memory_id = memory.get('id')
                if not memory_id:
                    continue
                
                memory_type = memory.get('memory_type', 'general')
                importance = memory.get('importance', 1)
                conversation_id = memory.get('conversation_id')
                
                new_index["memory_count"] += 1
                
                # memories
                new_index["memories"][memory_id] = {
                    "memory_type": memory_type,
                    "importance": importance,
                    "conversation_id": conversation_id,
                    "created_at": memory.get("created_at", ""),
                    "content_hash": memory.get("content_hash", ""),
                    "keyword_count": len(memory.get("keywords", []))
                }
                
                # by_type
                if memory_type not in new_index["by_type"]:
                    new_index["by_type"][memory_type] = []
                new_index["by_type"][memory_type].append(memory_id)
                
                # by_importance
                importance_key = f"imp_{importance // 2}"
                if importance_key not in new_index["by_importance"]:
                    new_index["by_importance"][importance_key] = []
                new_index["by_importance"][importance_key].append(memory_id)
                
                # by_conversation
                if conversation_id:
                    conv_key = str(conversation_id)
                    if conv_key not in new_index["by_conversation"]:
                        new_index["by_conversation"][conv_key] = []
                    new_index["by_conversation"][conv_key].append(memory_id)
                
            except Exception as e:
                logger.warning(f"⚠️ 处理记忆文件失败 {mem_file}: {e}")
        
        # 保存新索引
        self._save_index(user_id, new_index)
        logger.info(f"🔧 索引已优化，用户 {user_id}，共 {new_index['memory_count']} 条记忆")


# 全局实例
memory_storage = MemoryStorage()
