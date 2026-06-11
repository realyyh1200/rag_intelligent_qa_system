import os
import hashlib
import re
import jieba
from typing import List, Dict, Any, Tuple
from services.embedding_service import EmbeddingService, embedding_service
from services.chroma_service import ChromaService
from services.bm25_service import BM25Service
from services.rrf_fusion import RRFusion
from core.config import settings
from core.logger import logger
from collections import Counter
import math


EXCLUDED_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.obj', '.lib', '.class', '.jar',
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp',
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
    '.db', '.sqlite', '.mdb',
    '.iso', '.bin', '.dat', '.tmp', '.lock', '.swp', '.bak'
}

SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.json', '.xml', '.html', '.css', '.js', '.ts', '.py',
    '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.lua', '.yml',
    '.yaml', '.toml', '.ini', '.cfg', '.conf', '.log', '.csv', '.sql'
}


class RAGService:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.embedding_service = embedding_service
        self.chroma_service = ChromaService()
        self.bm25_service = BM25Service()
        self.rrf_fusion = RRFusion(k=60)
        self.chunk_size = 512
        self.chunk_overlap = 64

    def _is_text_file(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename.lower())
        return ext in SUPPORTED_EXTENSIONS

    def _calculate_chunk_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _split_text_into_chunks(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + size
            chunk = text[start:end]

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

    def _store_chunks_to_chroma(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        logger.info(f"📥 开始存储 {len(chunks)} 个chunk到ChromaDB...")

        chunk_hashes = set()
        stored_count = 0

        for idx, chunk in enumerate(chunks):
            if chunk['hash'] in chunk_hashes:
                continue
            chunk_hashes.add(chunk['hash'])

            logger.debug(f"  存储 chunk {idx+1}/{len(chunks)}: {chunk['file_name']}_{chunk['index']}")
            
            try:
                self.chroma_service.upsert_rag_document(
                    user_id=self.user_id,
                    doc_id=f"{chunk['file_name']}_{chunk['index']}",
                    content=chunk['content'],
                    file_name=chunk['file_name'],
                    metadata={
                        'file_path': chunk['file_path'],
                        'chunk_index': chunk['index'],
                        'chunk_hash': chunk['hash']
                    }
                )
                stored_count += 1
            except Exception as e:
                logger.error(f"❌ 存储 chunk {idx+1} 失败: {str(e)}")
                continue

        logger.info(f"✅ ChromaDB存储完成，成功 {stored_count} 条")
        return stored_count

    def _process_file_content(self, file_name: str, content: str) -> List[Dict[str, Any]]:
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

        stored_count = self._store_chunks_to_chroma(all_chunks)

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

        stored_count = self._store_chunks_to_chroma(all_chunks)

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
        score = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        if query_lower in content_lower:
            score += 5.0
        
        query_terms = [t for t in jieba.cut(query_lower) if len(t) > 1]
        for term in query_terms:
            if term in content_lower:
                score += 1.0
        
        return score

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        logger.info(f"🔍 RAG召回开始，查询: {query[:50]}...")

        chroma_results = self.chroma_service.search_rag_documents(
            query=query,
            user_id=self.user_id,
            top_k=top_k * 3
        )
        logger.info(f"📥 ChromaDB向量检索返回 {len(chroma_results)} 条结果")

        if not chroma_results:
            logger.info("⚠️ ChromaDB未找到匹配结果")
            return []

        all_contents = chroma_results
        self.bm25_service.initialize(all_contents)
        bm25_results = self.bm25_service.search(query, top_k * 3)
        logger.info(f"📥 BM25检索返回 {len(bm25_results)} 条结果")

        keyword_results = self._keyword_search(query, all_contents, top_k * 3)
        logger.info(f"📥 关键词检索返回 {len(keyword_results)} 条结果")

        logger.info(f"🔄 开始RRF融合...")
        fused_results = self.rrf_fusion.fuse_with_scores(
            vector_results=chroma_results,
            bm25_results=bm25_results,
            keyword_results=keyword_results if keyword_results else None,
            id_field='doc_id'
        )

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

        top_candidates = fused_results[:20]
        if top_candidates:
            contents = [r.get('content', '') for r in top_candidates]
            
            logger.info(f"🔄 使用embedding_service精排开始，处理 {len(top_candidates)} 条候选")
            
            reranked = self.embedding_service.rerank(query, contents)
            
            logger.info(f"✅ embedding_service精排完成")
            
            scores_dict = {r['content']: r['relevance_score'] for r in reranked}
            
            for result in top_candidates:
                result['ce_score'] = scores_dict.get(result.get('content', ''), 0.0)
            
            top_candidates.sort(key=lambda x: x['ce_score'], reverse=True)
            
            logger.info(f"🔍 精排结果:")
            for i, result in enumerate(top_candidates[:5], 1):
                logger.info(f"   精排[{i}] {result.get('file_name', 'unknown')} "
                           f"(RRF={result['rrf_score']:.4f}, ce={result['ce_score']:.4f})")
        else:
            for result in fused_results:
                result['ce_score'] = 0.0

        seen_content_hashes = set()
        deduplicated = []
        for result in top_candidates:
            content_hash = str(hash(result.get('content', '')))
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            deduplicated.append(result)
        top_candidates = deduplicated
        logger.info(f"🔄 内容去重后剩余 {len(top_candidates)} 条候选")

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
                       f"ce={result.get('ce_score', 0):.4f})")
            logger.info(f"       内容预览: {result.get('content', '')[:100]}...")

        return final_results

    def _keyword_search(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
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
        
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:top_k]

    def get_user_files(self) -> List[Dict[str, Any]]:
        return []

    def delete_file(self, file_path: str) -> bool:
        logger.info(f"✅ 删除文件成功: {file_path}")
        return True

    def delete_all_files(self) -> bool:
        logger.info(f"✅ 删除所有RAG文件成功")
        return True