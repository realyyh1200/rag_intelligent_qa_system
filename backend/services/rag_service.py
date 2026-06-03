import os
import hashlib
import re
import jieba
from typing import List, Dict, Any, Tuple
from services.bge_service import BGEEmbeddingService as BgeService
from services.qdrant_service import QdrantService
from services.cross_encoder_service import CrossEncoderService
from services.bm25_service import BM25Service
from services.rrf_fusion import RRFusion
from core.config import settings
from core.logger import logger
from collections import Counter
import math

# 排除的文件扩展名（二进制、视频、音频等）
EXCLUDED_EXTENSIONS = {
    # 二进制文件
    '.exe', '.dll', '.so', '.dylib', '.obj', '.lib', '.class', '.jar',
    # 视频
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
    # 音频
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    # 图像
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp',
    # 压缩文件
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',
    # 二进制文档
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
    # 数据库
    '.db', '.sqlite', '.mdb',
    # 其他
    '.iso', '.bin', '.dat', '.tmp', '.lock', '.swp', '.bak'
}

# 支持的文本文件扩展名
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.json', '.xml', '.html', '.css', '.js', '.ts', '.py',
    '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.lua', '.yml',
    '.yaml', '.toml', '.ini', '.cfg', '.conf', '.log', '.csv', '.sql'
}


class RAGService:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.bge_service = BgeService()
        self.qdrant_service = QdrantService()
        self.cross_encoder = CrossEncoderService()
        self.bm25_service = BM25Service()
        self.rrf_fusion = RRFusion(k=60)  # RRF融合器，k=60是常用默认值
        self.chunk_size = 512
        self.chunk_overlap = 64

    def _is_text_file(self, filename: str) -> bool:
        """判断是否为文本文件"""
        _, ext = os.path.splitext(filename.lower())
        return ext in SUPPORTED_EXTENSIONS

    def _calculate_chunk_hash(self, content: str) -> str:
        """计算chunk内容的哈希值，用于去重"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _split_text_into_chunks(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """将文本切分为chunk"""
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + size
            chunk = text[start:end]

            # 如果不是最后一个chunk，尝试在句子边界处分割
            if end < text_length:
                last_punctuation = max(
                    chunk.rfind('。'), chunk.rfind('.'), chunk.rfind('?'),
                    chunk.rfind('!'), chunk.rfind('\n'), chunk.rfind('？'),
                    chunk.rfind('！')
                )
                if last_punctuation != -1 and last_punctuation > size // 2:
                    end = start + last_punctuation + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

            if start >= text_length:
                break

        return chunks

    def _process_file(self, file_path: str, file_name: str = None) -> List[Dict[str, Any]]:
        """处理单个文件，返回chunk列表"""
        if not file_name:
            file_name = os.path.basename(file_path)

        logger.info(f"📄 处理文件: {file_name}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"⚠️ 文件内容为空: {file_name}")
                return []

            chunks = self._split_text_into_chunks(content)
            logger.info(f"✅ 文件 {file_name} 切分为 {len(chunks)} 个chunk")

            chunk_results = []
            for idx, chunk in enumerate(chunks):
                chunk_hash = self._calculate_chunk_hash(chunk)
                chunk_results.append({
                    'content': chunk,
                    'index': idx,
                    'hash': chunk_hash,
                    'file_name': file_name,
                    'file_path': file_path
                })

            return chunk_results
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {file_name}: {str(e)}")
            return []

    def _process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """处理目录，遍历所有文本文件"""
        logger.info(f"📁 处理目录: {dir_path}")

        all_chunks = []
        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                if self._is_text_file(filename):
                    file_path = os.path.join(root, filename)
                    try:
                        chunks = self._process_file(file_path, filename)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"❌ 处理文件失败 {file_path}: {str(e)}")

        logger.info(f"✅ 目录 {dir_path} 处理完成，共 {len(all_chunks)} 个chunk")
        return all_chunks

    def _store_chunks_to_qdrant(self, chunks: List[Dict[str, Any]]) -> int:
        """将chunks存储到Qdrant向量数据库"""
        if not chunks:
            return 0

        vectors = []
        payloads = []
        chunk_hashes = set()

        for chunk in chunks:
            # 去重检查
            if chunk['hash'] in chunk_hashes:
                continue
            chunk_hashes.add(chunk['hash'])

            embedding = self.bge_service.encode([chunk['content']])[0]
            vectors.append(embedding)
            payloads.append({
                'user_id': self.user_id,
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'chunk_index': chunk['index'],
                'content': chunk['content'],
                'chunk_hash': chunk['hash']
            })

        success_count = self.qdrant_service.store_vectors(vectors, payloads, settings.QDRANT_RAG_COLLECTION)
        logger.info(f"✅ Qdrant存储完成，成功 {success_count} 条")
        return success_count

    def _process_file_content(self, file_name: str, content: str) -> List[Dict[str, Any]]:
        """处理直接传入的文件内容，返回chunk列表"""
        logger.info(f"📄 处理文件内容: {file_name}")

        try:
            if not content.strip():
                logger.warning(f"⚠️ 文件内容为空: {file_name}")
                return []

            chunks = self._split_text_into_chunks(content)
            logger.info(f"✅ 文件 {file_name} 切分为 {len(chunks)} 个chunk")

            chunk_results = []
            for idx, chunk in enumerate(chunks):
                chunk_hash = self._calculate_chunk_hash(chunk)
                chunk_results.append({
                    'content': chunk,
                    'index': idx,
                    'hash': chunk_hash,
                    'file_name': file_name,
                    'file_path': f'/uploaded/{file_name}'
                })

            return chunk_results
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {file_name}: {str(e)}")
            return []

    def process_files(self, paths: List[str]) -> Dict[str, Any]:
        """处理文件或文件夹列表"""
        all_chunks = []

        for path in paths:
            if os.path.isfile(path):
                if self._is_text_file(path):
                    chunks = self._process_file(path)
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"⚠️ 跳过非文本文件: {path}")
            elif os.path.isdir(path):
                chunks = self._process_directory(path)
                all_chunks.extend(chunks)
            else:
                logger.error(f"❌ 路径不存在: {path}")

        if not all_chunks:
            return {
                'success': True,
                'message': '没有找到可处理的文本文件',
                'total_files': 0,
                'total_chunks': 0,
                'stored_chunks': 0,
                'skipped_chunks': 0
            }

        # 直接存储到Qdrant（已包含去重逻辑）
        stored_count = self._store_chunks_to_qdrant(all_chunks)

        file_names = list(set(ch['file_name'] for ch in all_chunks))

        return {
            'success': True,
            'message': f'处理完成！',
            'total_files': len(file_names),
            'total_chunks': len(all_chunks),
            'stored_chunks': stored_count,
            'skipped_chunks': len(all_chunks) - stored_count,
            'files': file_names
        }

    def process_uploaded_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理前端上传的文件内容列表"""
        all_chunks = []

        for file in files:
            file_name = file.get('name', 'unknown')
            content = file.get('content', '')

            if not self._is_text_file(file_name):
                logger.warning(f"⚠️ 跳过非文本文件: {file_name}")
                continue

            chunks = self._process_file_content(file_name, content)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {
                'success': True,
                'message': '没有找到可处理的文本文件',
                'total_files': 0,
                'total_chunks': 0,
                'stored_chunks': 0,
                'skipped_chunks': 0
            }

        # 直接存储到Qdrant（已包含去重逻辑）
        stored_count = self._store_chunks_to_qdrant(all_chunks)

        file_names = list(set(ch['file_name'] for ch in all_chunks))

        return {
            'success': True,
            'message': f'处理完成！',
            'total_files': len(file_names),
            'total_chunks': len(all_chunks),
            'stored_chunks': stored_count,
            'skipped_chunks': len(all_chunks) - stored_count,
            'files': file_names
        }

    def _keyword_matching_score(self, query: str, content: str) -> float:
        """关键词精确匹配得分 - 用于增强短关键词的召回能力"""
        score = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 精确匹配整个查询
        if query_lower in content_lower:
            score += 5.0
        
        # 分词匹配
        query_terms = [t for t in jieba.cut(query_lower) if len(t) > 1]
        for term in query_terms:
            if term in content_lower:
                score += 1.0
        
        return score

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """使用RRF融合向量检索、BM25和关键词检索，加入cross-encoder精排"""
        logger.info(f"🔍 RAG召回开始，查询: {query[:50]}...")

        # 1. 向量检索（Qdrant）
        query_embedding = self.bge_service.encode([query])[0]
        qdrant_results = self.qdrant_service.search_vectors_for_rag(
            query_embedding,
            user_id=self.user_id,
            limit=top_k * 3
        )
        logger.info(f"📥 Qdrant向量检索返回 {len(qdrant_results)} 条结果")

        if not qdrant_results:
            logger.info("⚠️ Qdrant未找到匹配结果")
            return []

        # 2. BM25检索
        all_contents = self.qdrant_service.get_all_rag_contents_for_user(self.user_id)
        self.bm25_service.initialize(all_contents)
        bm25_results = self.bm25_service.search(query, top_k * 3)
        logger.info(f"📥 BM25检索返回 {len(bm25_results)} 条结果")

        # 3. 关键词检索（用于增强短关键词召回）
        keyword_results = self._keyword_search(query, all_contents, top_k * 3)
        logger.info(f"📥 关键词检索返回 {len(keyword_results)} 条结果")

        # 4. RRF融合三个检索结果
        logger.info(f"🔄 开始RRF融合...")
        fused_results = self.rrf_fusion.fuse_with_scores(
            vector_results=qdrant_results,
            bm25_results=bm25_results,
            keyword_results=keyword_results if keyword_results else None,
            id_field='id'
        )

        # 记录排名信息
        for result in fused_results:
            ranks = []
            if 'vector_rank' in result:
                ranks.append(f"V:{result['vector_rank']}")
            if 'bm25_rank' in result:
                ranks.append(f"B:{result['bm25_rank']}")
            if 'keyword_rank' in result:
                ranks.append(f"K:{result['keyword_rank']}")
            result['ranks_info'] = ','.join(ranks) if ranks else 'N/A'

        logger.info(f"📊 RRF融合完成，取前20条进入精排")
        for i, result in enumerate(fused_results[:5], 1):
            logger.info(f"   粗排[{i}] {result.get('file_name', 'unknown')} "
                       f"(RRF={result['rrf_score']:.4f}, ranks=[{result['ranks_info']}])")

        # 5. Cross-encoder精排：取前20条进行精排
        top_candidates = fused_results[:20]
        if top_candidates:
            contents = [r.get('content', '') for r in top_candidates]
            ce_scores = self.cross_encoder.score(query, contents)
            
            logger.info(f"🔄 Cross-encoder精排开始，处理 {len(top_candidates)} 条候选")
            logger.info(f"   CE分数范围: [{min(ce_scores):.4f}, {max(ce_scores):.4f}]")
            
            # 使用RRF融合精排分数
            ce_ranked = sorted(enumerate(ce_scores), key=lambda x: x[1], reverse=True)
            ce_ranks = [(i, rank+1) for rank, (i, _) in enumerate(ce_ranked)]
            
            # 重新计算RRF分数
            final_scores = []
            for result in top_candidates:
                # 原始RRF分数
                rrf_base = result['rrf_score']
                # 精排RRF
                ce_rank = next((rank for i, rank in ce_ranks if i == top_candidates.index(result)), 999)
                ce_rrf = 1.0 / (60 + ce_rank)
                # 最终RRF分数 = 原始RRF + 精排RRF
                final_rrf = rrf_base + ce_rrf
                result['ce_score'] = ce_scores[top_candidates.index(result)]
                result['ce_rank'] = ce_rank
                result['final_rrf_score'] = final_rrf
                final_scores.append((result, final_rrf))
            
            # 按最终RRF分数排序
            final_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [r for r, _ in final_scores]
            
            logger.info(f"✅ Cross-encoder精排完成")
            logger.info(f"🔍 精排前后对比:")
            for i, (result, score) in enumerate(final_scores[:5], 1):
                logger.info(f"   精排[{i}] {result.get('file_name', 'unknown')} "
                           f"(RRF={result['rrf_score']:.4f}, ce={result['ce_score']:.4f}, "
                           f"final={result['final_rrf_score']:.4f})")
        else:
            for result in fused_results:
                result['ce_score'] = 0.0
                result['final_rrf_score'] = result['rrf_score']

        # 6. 文件多样性控制
        file_chunk_count = {}
        final_results = []
        max_chunks_per_file = 3

        for result in top_candidates:
            if len(final_results) >= top_k:
                break

            file_key = result.get('file_name', '')
            current_count = file_chunk_count.get(file_key, 0)

            if current_count < max_chunks_per_file:
                file_chunk_count[file_key] = current_count + 1
                final_results.append(result)

        logger.info(f"✅ RAG召回完成，找到 {len(final_results)} 条相关结果（来自 {len(file_chunk_count)} 个文件）:")
        for i, result in enumerate(final_results, 1):
            logger.info(f"   [{i}] {result.get('file_name', 'unknown')} "
                       f"(chunk_{result.get('chunk_index', 0)}) "
                       f"(RRF={result['rrf_score']:.4f}, "
                       f"ce={result.get('ce_score', 0):.4f}, "
                       f"final={result.get('final_rrf_score', result['rrf_score']):.4f})")
            logger.info(f"       内容预览: {result.get('content', '')[:100]}...")

        return final_results

    def _keyword_search(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """关键词精确匹配检索"""
        if not documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            score = self._keyword_matching_score(query_lower, content)
            if score > 0:
                results.append({
                    'id': doc.get('id', 0),
                    'content': doc.get('content', ''),
                    'file_name': doc.get('file_name', ''),
                    'file_path': doc.get('file_path', ''),
                    'chunk_index': doc.get('chunk_index', 0),
                    'keyword_score': score
                })
        
        # 按关键词分数排序
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:top_k]

    def get_user_files(self) -> List[Dict[str, Any]]:
        """获取用户的RAG文件列表"""
        files = self.qdrant_service.list_user_files(self.user_id)
        return [{
            'file_name': f['file_name'],
            'file_path': f['file_path'],
            'chunk_count': f['chunk_count']
        } for f in files]

    def delete_file(self, file_path: str) -> bool:
        """删除用户的RAG文件"""
        success = self.qdrant_service.delete_vectors_by_file(file_path, self.user_id)
        
        if success:
            logger.info(f"✅ 删除文件成功: {file_path}")
        else:
            logger.warning(f"❌ 删除文件失败: {file_path}")
        
        return success

    def delete_all_files(self) -> bool:
        """删除用户的所有RAG数据"""
        success = self.qdrant_service.delete_all_user_rag(self.user_id)
        
        if success:
            logger.info(f"✅ 删除所有RAG文件成功")
        else:
            logger.warning(f"❌ 删除所有RAG文件失败")
        
        return success
