"""
GSSC上下文管理流水线
Generate-Score-Select-Compress

四个阶段：
1. Gather（汇集）- 从多个来源收集候选信息
2. Select（选择）- 相关性+新近性评分，贪心选择
3. Structure（结构化）- 组织成固定模板结构
4. Compress（压缩）- 兜底的智能摘要压缩
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import tiktoken
from core.logger import logger


class ContextSource(Enum):
    """上下文来源类型"""
    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    HISTORY = "history"
    RAG = "rag"
    MEMORY = "memory"
    CUSTOM = "custom"


@dataclass
class ContextItem:
    """单个上下文项"""
    content: str
    source: ContextSource
    priority: int = 5  # 优先级 1-10, 10最高
    relevance_score: float = 0.0  # 相关性评分 0-1
    recency_score: float = 0.0  # 新近性评分 0-1
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self.estimate_tokens(self.content)
    
    @staticmethod
    def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
        """估算token数量"""
        try:
            enc = tiktoken.get_encoding(model)
            return len(enc.encode(text))
        except Exception:
            # 粗略估算：中文约2字符/token，英文约4字符/token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 2 + other_chars / 4)


@dataclass
class StructuredContext:
    """结构化上下文模板"""
    role_policies: str = ""  # 角色与策略
    task: str = ""  # 任务
    state: str = ""  # 状态
    evidence: str = ""  # 证据（RAG内容）
    context: str = ""  # 上下文（历史对话、记忆）
    output_requirements: str = ""  # 输出要求
    
    def to_prompt(self) -> str:
        """转换为prompt字符串"""
        parts = []
        
        if self.role_policies:
            parts.append(f"【角色与策略】\n{self.role_policies}")
        
        if self.task:
            parts.append(f"【任务】\n{self.task}")
        
        if self.state:
            parts.append(f"【状态】\n{self.state}")
        
        if self.evidence:
            parts.append(f"【证据】\n{self.evidence}")
        
        if self.context:
            parts.append(f"【上下文】\n{self.context}")
        
        if self.output_requirements:
            parts.append(f"【输出要求】\n{self.output_requirements}")
        
        return "\n\n".join(parts)
    
    def estimate_tokens(self) -> int:
        """估算总token数"""
        return ContextItem.estimate_tokens(self.to_prompt())


class GatherStage:
    """阶段1：汇集（Gather）- 从多个来源收集信息"""
    
    def __init__(self):
        self.sources: List[ContextSource] = []
        self.items: List[ContextItem] = []
    
    def add_system_prompt(self, prompt: str, priority: int = 10) -> 'GatherStage':
        """添加系统指令"""
        self.items.append(ContextItem(
            content=prompt,
            source=ContextSource.SYSTEM_PROMPT,
            priority=priority,
            relevance_score=1.0,
            recency_score=1.0,
            metadata={"type": "system"}
        ))
        return self
    
    def add_user_message(self, message: str, priority: int = 10) -> 'GatherStage':
        """添加用户消息"""
        self.items.append(ContextItem(
            content=message,
            source=ContextSource.USER_MESSAGE,
            priority=priority,
            relevance_score=1.0,
            recency_score=1.0,
            metadata={"type": "user_message"}
        ))
        return self
    
    def add_history(self, messages: List[Dict[str, str]], 
                   max_count: int = 20,
                   priority: int = 5) -> 'GatherStage':
        """添加历史对话"""
        for i, msg in enumerate(messages[-max_count:]):
            # 越新的消息，新近性分数越高
            recency = (i + 1) / max_count if max_count > 0 else 0
            self.items.append(ContextItem(
                content=f"{msg.get('role', 'user')}: {msg.get('content', '')}",
                source=ContextSource.HISTORY,
                priority=priority,
                relevance_score=0.5,  # 历史消息相关性稍低
                recency_score=recency,
                metadata={
                    "type": "history",
                    "role": msg.get('role'),
                    "index": len(messages) - max_count + i
                }
            ))
        return self
    
    def add_rag_results(self, results: List[Dict[str, Any]], 
                       priority: int = 7) -> 'GatherStage':
        """添加RAG召回结果"""
        for result in results:
            self.items.append(ContextItem(
                content=f"【文档: {result.get('file_name', 'unknown')}】\n{result.get('content', '')}",
                source=ContextSource.RAG,
                priority=priority,
                relevance_score=result.get('final_rrf_score', result.get('rrf_score', 0.5)),
                recency_score=0.5,  # RAG结果不考虑新近性
                metadata={
                    "type": "rag",
                    "file_name": result.get('file_name'),
                    "file_path": result.get('file_path'),
                    "chunk_index": result.get('chunk_index')
                }
            ))
        return self
    
    def add_memory(self, memories: List[Dict[str, Any]], 
                  priority: int = 6) -> 'GatherStage':
        """添加记忆内容"""
        for memory in memories:
            self.items.append(ContextItem(
                content=memory.get('content', ''),
                source=ContextSource.MEMORY,
                priority=priority,
                relevance_score=memory.get('importance', 1) / 10.0,
                recency_score=0.5,
                metadata={
                    "type": "memory",
                    "memory_type": memory.get('memory_type'),
                    "importance": memory.get('importance', 1)
                }
            ))
        return self
    
    def add_custom(self, content: str, label: str, 
                  priority: int = 5, 
                  relevance_score: float = 0.5) -> 'GatherStage':
        """添加自定义内容"""
        self.items.append(ContextItem(
            content=f"【{label}】\n{content}",
            source=ContextSource.CUSTOM,
            priority=priority,
            relevance_score=relevance_score,
            recency_score=0.5,
            metadata={"type": "custom", "label": label}
        ))
        return self
    
    def gather(self) -> List[ContextItem]:
        """执行汇集，返回所有候选上下文项"""
        logger.info(f"📥 Gather阶段完成，收集到 {len(self.items)} 个候选上下文项")
        for item in self.items:
            logger.debug(f"   - [{item.source.value}] (priority={item.priority}, tokens={item.token_count})")
        return self.items


class SelectStage:
    """阶段2：选择（Select）- 评分和贪心选择"""
    
    def __init__(self, 
                 relevance_weight: float = 0.7,
                 recency_weight: float = 0.3,
                 min_relevance_threshold: float = 0.2,
                 max_token_budget: int = 6000):
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.min_relevance_threshold = min_relevance_threshold
        self.max_token_budget = max_token_budget
    
    def _calculate_composite_score(self, item: ContextItem) -> float:
        """计算综合评分 = 相关性×权重 + 新近性×权重"""
        return (
            self.relevance_weight * item.relevance_score +
            self.recency_weight * item.recency_score
        ) * (item.priority / 10.0)  # 乘以优先级归一化
    
    def select(self, items: List[ContextItem], 
              query: str = "") -> Tuple[List[ContextItem], int]:
        """
        执行选择，返回选中的上下文项和总token数
        
        使用贪心算法：
        1. 按综合评分降序排序
        2. 依次选择，直到Token预算耗尽
        3. 跳过低于相关性阈值的项
        """
        # 计算所有项的综合评分
        for item in items:
            item.relevance_score = self._calculate_composite_score(item)
        
        # 按综合评分排序
        sorted_items = sorted(items, 
                            key=lambda x: x.relevance_score, 
                            reverse=True)
        
        selected = []
        total_tokens = 0
        
        for item in sorted_items:
            # 跳过低于相关性阈值的项
            if item.relevance_score < self.min_relevance_threshold:
                logger.debug(f"   ⏭️ 跳过 [{item.source.value}] (score={item.relevance_score:.3f} < {self.min_relevance_threshold})")
                continue
            
            # 检查Token预算
            if total_tokens + item.token_count > self.max_token_budget:
                logger.debug(f"   ⏭️ Token预算耗尽 [{item.source.value}] (need {item.token_count}, have {self.max_token_budget - total_tokens})")
                continue
            
            selected.append(item)
            total_tokens += item.token_count
            logger.debug(f"   ✅ 选中 [{item.source.value}] (score={item.relevance_score:.3f}, tokens={item.token_count})")
        
        logger.info(f"🔍 Select阶段完成，从 {len(items)} 个候选中选中 {len(selected)} 个，消耗 {total_tokens} tokens")
        
        return selected, total_tokens


class StructureStage:
    """阶段3：结构化（Structure）- 组织成固定模板"""
    
    def __init__(self):
        self.role_policies_template = """你是一个专业的RAG智能问答系统。
核心原则：
1. 始终基于提供的【证据】内容回答问题
2. 如果证据不足以回答，诚实地说明并基于你的知识补充
3. 不要编造信息，引用时要标注来源
4. 回答要清晰、有条理，适当使用列表或代码块格式化"""
        
        self.output_requirements_template = """【输出要求】
1. 直接回答问题，不要重复问题
2. 引用【证据】时使用 [文档名] 格式
3. 保持回答简洁，突出关键信息"""
    
    def structure(self, 
                items: List[ContextItem],
                user_query: str,
                system_prompt: str = "") -> StructuredContext:
        """
        将选中的上下文项组织成结构化格式
        """
        context = StructuredContext()
        
        # 1. 角色与策略
        context.role_policies = self.role_policies_template
        if system_prompt:
            context.role_policies += f"\n\n【自定义系统指令】\n{system_prompt}"
        
        # 2. 任务（用户问题）
        context.task = f"【用户问题】\n{user_query}"
        
        # 3. 状态（空，暂不使用）
        context.state = ""
        
        # 4. 证据（RAG结果）
        rag_items = [item for item in items if item.source == ContextSource.RAG]
        if rag_items:
            context.evidence = "\n\n".join([item.content for item in rag_items])
        
        # 5. 上下文（历史对话 + 记忆）
        other_items = [item for item in items if item.source != ContextSource.RAG]
        if other_items:
            # 按来源分组
            history_items = [item for item in other_items if item.source == ContextSource.HISTORY]
            memory_items = [item for item in other_items if item.source == ContextSource.MEMORY]
            system_items = [item for item in other_items if item.source == ContextSource.SYSTEM_PROMPT]
            user_items = [item for item in other_items if item.source == ContextSource.USER_MESSAGE]
            
            context_parts = []
            
            if history_items:
                context_parts.append("【历史对话】\n" + "\n".join([item.content for item in history_items]))
            
            if memory_items:
                context_parts.append("【相关记忆】\n" + "\n".join([item.content for item in memory_items]))
            
            if user_items:
                context_parts.append("【当前用户输入】\n" + user_items[0].content)
            
            context.context = "\n\n".join(context_parts)
        
        # 6. 输出要求
        context.output_requirements = self.output_requirements_template
        
        return context


class CompressStage:
    """阶段4：压缩（Compress）- 兜底的智能摘要"""
    
    def __init__(self):
        self.min_summary_ratio = 0.3  # 最小压缩到30%
        self.max_compression_ratio = 0.7  # 最大压缩到70%
    
    def _smart_summary(self, content: str, target_ratio: float = 0.5) -> str:
        """
        智能摘要 - 保留关键信息
        """
        # 估算当前token数
        current_tokens = ContextItem.estimate_tokens(content)
        target_tokens = int(current_tokens * target_ratio)
        
        if current_tokens <= target_tokens:
            return content
        
        # 简单的截断摘要 - 保留开头和结尾（通常最重要）
        lines = content.split('\n')
        
        if len(lines) <= 3:
            # 内容太少，直接截断
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(content)
            tokens = tokens[:target_tokens]
            return enc.decode(tokens)
        
        # 保留第一段和最后一段，中间部分截断
        first_lines = []
        last_lines = []
        middle_lines = []
        
        for i, line in enumerate(lines):
            if i < len(lines) // 3:
                first_lines.append(line)
            elif i >= 2 * len(lines) // 3:
                last_lines.append(line)
            else:
                middle_lines.append(line)
        
        # 构建摘要
        summary_parts = []
        
        # 开头
        if first_lines:
            enc = tiktoken.get_encoding("cl100k_base")
            first_text = '\n'.join(first_lines)
            first_tokens = enc.encode(first_text)
            if len(first_tokens) > target_tokens // 3:
                first_tokens = first_tokens[:target_tokens // 3]
            summary_parts.append(enc.decode(first_tokens))
        
        # 中间（摘要）
        if middle_lines:
            middle_text = f"\n【内容摘要，已压缩】\n" + self._compress_middle(middle_lines, target_tokens // 3)
            enc = tiktoken.get_encoding("cl100k_base")
            middle_tokens = enc.encode(middle_text)
            if len(middle_tokens) > target_tokens // 3:
                middle_tokens = middle_tokens[:target_tokens // 3]
            summary_parts.append(enc.decode(middle_tokens))
        
        # 结尾
        if last_lines:
            enc = tiktoken.get_encoding("cl100k_base")
            last_text = '\n'.join(last_lines)
            last_tokens = enc.encode(last_text)
            if len(last_tokens) > target_tokens // 3:
                last_tokens = last_tokens[:target_tokens // 3]
            summary_parts.append(enc.decode(last_tokens))
        
        return '\n'.join(summary_parts)
    
    def _compress_middle(self, lines: List[str], max_tokens: int) -> str:
        """压缩中间部分 - 提取关键句子"""
        # 简单策略：每隔一行取一行
        key_lines = []
        for i, line in enumerate(lines):
            if i % 2 == 0:  # 保留偶数行
                key_lines.append(line)
        
        result = '\n'.join(key_lines)
        
        # 如果还是太长，进一步截断
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(result)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return enc.decode(tokens)
        
        return result
    
    def compress(self, 
                context: StructuredContext,
                max_token_budget: int = 4000) -> Tuple[StructuredContext, bool]:
        """
        执行压缩，如果需要的话
        
        返回：(压缩后的上下文, 是否进行了压缩)
        """
        current_tokens = context.estimate_tokens()
        
        if current_tokens <= max_token_budget:
            logger.info(f"📦 Compress阶段：无需压缩 (current={current_tokens}, budget={max_token_budget})")
            return context, False
        
        logger.info(f"📦 Compress阶段：开始压缩 (current={current_tokens}, budget={max_token_budget})")
        
        # 计算需要压缩的比例
        ratio = max_token_budget / current_tokens
        ratio = max(ratio, self.min_summary_ratio)  # 至少保留30%
        ratio = min(ratio, self.max_compression_ratio)  # 最多压缩到70%
        
        compressed = StructuredContext()
        
        # 系统指令不压缩
        compressed.role_policies = context.role_policies
        
        # 任务不压缩
        compressed.task = context.task
        
        # 状态不压缩
        compressed.state = context.state
        
        # 证据压缩
        if context.evidence:
            compressed.evidence = self._smart_summary(context.evidence, ratio)
        
        # 上下文压缩
        if context.context:
            compressed.context = self._smart_summary(context.context, ratio)
        
        # 输出要求不压缩
        compressed.output_requirements = context.output_requirements
        
        new_tokens = compressed.estimate_tokens()
        logger.info(f"📦 Compress阶段：压缩完成 (before={current_tokens}, after={new_tokens}, ratio={new_tokens/current_tokens:.2%})")
        
        return compressed, True


class GSSCPipeline:
    """
    GSSC上下文管理流水线
    
    使用示例:
    pipeline = GSSCPipeline(
        max_token_budget=6000,
        relevance_weight=0.7,
        recency_weight=0.3
    )
    
    context = pipeline.run(
        user_query="如何学习Python?",
        system_prompt="你是一个Python教学助手",
        history=[...],
        rag_results=[...],
        memories=[...]
    )
    """
    
    def __init__(self,
                 max_token_budget: int = 6000,
                 relevance_weight: float = 0.7,
                 recency_weight: float = 0.3,
                 min_relevance_threshold: float = 0.2):
        self.max_token_budget = max_token_budget
        self.gather = GatherStage()
        self.select = SelectStage(
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            min_relevance_threshold=min_relevance_threshold,
            max_token_budget=max_token_budget
        )
        self.structure = StructureStage()
        self.compress = CompressStage()
    
    def run(self,
           user_query: str,
           system_prompt: str = "",
           history: List[Dict[str, str]] = None,
           rag_results: List[Dict[str, Any]] = None,
           memories: List[Dict[str, Any]] = None,
           custom_contexts: List[Tuple[str, str]] = None) -> str:
        """
        执行完整的GSSC流水线
        
        Args:
            user_query: 用户问题
            system_prompt: 系统提示词
            history: 历史对话记录
            rag_results: RAG召回结果
            memories: 记忆内容
            custom_contexts: 自定义上下文 [(label, content), ...]
        
        Returns:
            构建好的prompt字符串
        """
        history = history or []
        rag_results = rag_results or []
        memories = memories or []
        custom_contexts = custom_contexts or []
        
        # ========== 阶段1: Gather ==========
        logger.info("=" * 60)
        logger.info("🟢 GSSC Pipeline - Stage 1: Gather")
        logger.info("=" * 60)
        
        self.gather = GatherStage()  # 重置
        self.gather.add_user_message(user_query, priority=10)
        self.gather.add_system_prompt(system_prompt, priority=10)
        self.gather.add_history(history, priority=5)
        
        if rag_results:
            self.gather.add_rag_results(rag_results, priority=7)
        
        if memories:
            self.gather.add_memory(memories, priority=6)
        
        for label, content in custom_contexts:
            self.gather.add_custom(content, label, priority=5)
        
        gathered_items = self.gather.gather()
        
        # ========== 阶段2: Select ==========
        logger.info("=" * 60)
        logger.info("🔵 GSSC Pipeline - Stage 2: Select")
        logger.info("=" * 60)
        
        # 重置选择器
        self.select = SelectStage(
            relevance_weight=self.select.relevance_weight,
            recency_weight=self.select.recency_weight,
            min_relevance_threshold=self.select.min_relevance_threshold,
            max_token_budget=self.max_token_budget
        )
        
        selected_items, used_tokens = self.select.select(gathered_items, user_query)
        
        # ========== 阶段3: Structure ==========
        logger.info("=" * 60)
        logger.info("🟡 GSSC Pipeline - Stage 3: Structure")
        logger.info("=" * 60)
        
        structured = self.structure.structure(
            items=selected_items,
            user_query=user_query,
            system_prompt=system_prompt
        )
        
        logger.info(f"📋 Structure阶段完成，生成结构化上下文 ({structured.estimate_tokens()} tokens)")
        
        # ========== 阶段4: Compress ==========
        logger.info("=" * 60)
        logger.info("🔴 GSSC Pipeline - Stage 4: Compress")
        logger.info("=" * 60)
        
        # 设置压缩预算为总预算的70%
        compress_budget = int(self.max_token_budget * 0.7)
        final_context, was_compressed = self.compress.compress(structured, compress_budget)
        
        # ========== 输出最终Prompt ==========
        final_prompt = final_context.to_prompt()
        
        logger.info("=" * 60)
        logger.info(f"✅ GSSC Pipeline 完成!")
        logger.info(f"   候选项: {len(gathered_items)}")
        logger.info(f"   选中项: {len(selected_items)}")
        logger.info(f"   Token使用: {ContextItem.estimate_tokens(final_prompt)}")
        logger.info(f"   压缩: {'是' if was_compressed else '否'}")
        logger.info("=" * 60)
        
        return final_prompt
