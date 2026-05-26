import os
import hashlib
import re
import jieba
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from models.rag import RAGFile, RAGChunk
from services.bge_service import BGEEmbeddingService as BgeService
from services.qdrant_service import QdrantService
from services.cross_encoder_service import CrossEncoderService
from core.logger import logger

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
    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id
        self.bge_service = BgeService()
        self.qdrant_service = QdrantService()
        self.cross_encoder = CrossEncoderService()
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

    def _store_chunks_to_mysql(self, chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        """将chunks存储到MySQL数据库"""
        stored_count = 0
        skipped_count = 0

        for chunk in chunks:
            # 检查当前用户是否已存在相同的chunk（通过file_path和chunk_index判断）
            existing = self.db.query(RAGChunk).join(RAGFile).filter(
                RAGChunk.chunk_index == chunk['index'],
                RAGFile.file_path == chunk['file_path'],
                RAGFile.user_id == self.user_id
            ).first()

            if existing:
                skipped_count += 1
                continue

            rag_file = self.db.query(RAGFile).filter(
                RAGFile.user_id == self.user_id,
                RAGFile.file_path == chunk['file_path']
            ).first()

            if not rag_file:
                rag_file = RAGFile(
                    user_id=self.user_id,
                    file_name=chunk['file_name'],
                    file_path=chunk['file_path'],
                    file_size=len(chunk['content'].encode('utf-8'))
                )
                self.db.add(rag_file)
                self.db.commit()
                self.db.refresh(rag_file)

            rag_chunk = RAGChunk(
                file_id=rag_file.id,
                chunk_index=chunk['index'],
                content=chunk['content'],
                chunk_hash=chunk['hash']
            )
            self.db.add(rag_chunk)
            stored_count += 1

        self.db.commit()
        return stored_count, skipped_count

    def _store_chunks_to_qdrant(self, chunks: List[Dict[str, Any]]) -> int:
        """将chunks存储到Qdrant向量数据库"""
        if not chunks:
            return 0

        vectors = []
        payloads = []

        for chunk in chunks:
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

        success_count = self.qdrant_service.store_vectors(vectors, payloads)
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

        stored_count, skipped_count = self._store_chunks_to_mysql(all_chunks)
        new_chunks = [c for i, c in enumerate(all_chunks) if i < stored_count]
        qdrant_count = self._store_chunks_to_qdrant(new_chunks)

        file_names = list(set(ch['file_name'] for ch in all_chunks))

        return {
            'success': True,
            'message': f'处理完成！',
            'total_files': len(file_names),
            'total_chunks': len(all_chunks),
            'stored_chunks': stored_count,
            'skipped_chunks': skipped_count,
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

        stored_count, skipped_count = self._store_chunks_to_mysql(all_chunks)
        new_chunks = [c for i, c in enumerate(all_chunks) if i < stored_count]
        qdrant_count = self._store_chunks_to_qdrant(new_chunks)

        file_names = list(set(ch['file_name'] for ch in all_chunks))

        return {
            'success': True,
            'message': f'处理完成！',
            'total_files': len(file_names),
            'total_chunks': len(all_chunks),
            'stored_chunks': stored_count,
            'skipped_chunks': skipped_count,
            'files': file_names
        }

    def _bm25_score(self, query: str, content: str) -> float:
        """使用jieba分词计算BM25相似度得分"""
        query_terms = set(jieba.cut(query.lower()))
        content_terms = jieba.cut(content.lower())

        query_terms = {t for t in query_terms if len(t) > 1}
        content_terms_list = [t for t in content_terms if len(t) > 1]

        if not query_terms or not content_terms_list:
            return 0.0

        score = 0.0
        k1 = 1.2
        b = 0.75
        avg_doc_len = 200

        doc_len = len(content_terms_list)
        term_freq = {}
        for term in content_terms_list:
            term_freq[term] = term_freq.get(term, 0) + 1

        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

        return score

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
        """使用BM25+余弦相似度召回，加入cross-encoder精排"""
        logger.info(f"🔍 RAG召回开始，查询: {query[:50]}...")

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

        hybrid_results = []
        all_cosine = [result.get('score', 0) for result in qdrant_results]
        all_bm25 = [self._bm25_score(query, result['content']) for result in qdrant_results]
        all_keyword = [self._keyword_matching_score(query, result['content']) for result in qdrant_results]
        
        cos_min, cos_max = min(all_cosine), max(all_cosine)
        bm25_min, bm25_max = min(all_bm25), max(all_bm25)
        kw_min, kw_max = min(all_keyword), max(all_keyword)
        
        for i, result in enumerate(qdrant_results):
            cosine_score = result.get('score', 0)
            bm25_score = all_bm25[i]
            keyword_score = all_keyword[i]
            
            norm_cosine = (cosine_score - cos_min) / (cos_max - cos_min + 1e-8)
            norm_bm25 = (bm25_score - bm25_min) / (bm25_max - bm25_min + 1e-8)
            norm_keyword = (keyword_score - kw_min) / (kw_max - kw_min + 1e-8)
            
            # 加入关键词匹配得分，增强短关键词召回
            # 如果有关键词匹配，给予更高权重
            if keyword_score > 0:
                hybrid_score = 0.3 * norm_cosine + 0.2 * norm_bm25 + 0.5 * norm_keyword
            else:
                hybrid_score = 0.5 * norm_cosine + 0.5 * norm_bm25

            hybrid_results.append({
                'content': result['content'],
                'file_name': result['file_name'],
                'file_path': result['file_path'],
                'chunk_index': result['chunk_index'],
                'cosine_score': cosine_score,
                'bm25_score': bm25_score,
                'hybrid_score': hybrid_score
            })

        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        logger.info(f"📊 粗排(BM25+余弦)完成，取前20条进入精排")
        for i, result in enumerate(hybrid_results[:5], 1):
            logger.info(f"   粗排[{i}] {result['file_name']} (cosine={result['cosine_score']:.4f}, bm25={result['bm25_score']:.4f}, hybrid={result['hybrid_score']:.4f})")

        # Cross-encoder精排：取前20条进行精排
        top_candidates = hybrid_results[:20]
        if top_candidates:
            contents = [r['content'] for r in top_candidates]
            ce_scores = self.cross_encoder.score(query, contents)
            
            logger.info(f"🔄 Cross-encoder精排开始，处理 {len(top_candidates)} 条候选")
            logger.info(f"   CE分数范围: [{min(ce_scores):.4f}, {max(ce_scores):.4f}]")
            
            for i, result in enumerate(top_candidates):
                result['ce_score'] = ce_scores[i]
                result['final_score'] = 0.5 * result['hybrid_score'] + 0.5 * ce_scores[i]
            
            top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            logger.info(f"✅ Cross-encoder精排完成")
            
            logger.info(f"🔍 精排前后对比:")
            for i, result in enumerate(top_candidates[:5], 1):
                logger.info(f"   精排[{i}] {result['file_name']} (hybrid={result['hybrid_score']:.4f}, ce={result['ce_score']:.4f}, final={result['final_score']:.4f})")
        else:
            for result in hybrid_results:
                result['ce_score'] = 0.0
                result['final_score'] = result['hybrid_score']

        file_chunk_count = {}
        final_results = []
        max_chunks_per_file = 3

        for result in top_candidates:
            if len(final_results) >= top_k:
                break

            file_key = result['file_name']
            current_count = file_chunk_count.get(file_key, 0)

            if current_count < max_chunks_per_file:
                file_chunk_count[file_key] = current_count + 1
                final_results.append(result)

        logger.info(f"✅ RAG召回完成，找到 {len(final_results)} 条相关结果（来自 {len(file_chunk_count)} 个文件）:")
        for i, result in enumerate(final_results, 1):
            logger.info(f"   [{i}] {result['file_name']} (chunk_{result['chunk_index']}) (cosine={result['cosine_score']:.4f}, bm25={result['bm25_score']:.4f}, ce={result.get('ce_score', 0):.4f}, final={result.get('final_score', result['hybrid_score']):.4f})")
            logger.info(f"       内容预览: {result['content'][:100]}...")

        return final_results

    def get_user_files(self) -> List[Dict[str, Any]]:
        """获取用户的RAG文件列表"""
        files = self.db.query(RAGFile).filter(RAGFile.user_id == self.user_id).all()
        return [{
            'id': f.id,
            'file_name': f.file_name,
            'file_path': f.file_path,
            'file_size': f.file_size,
            'created_at': f.created_at.isoformat() if f.created_at else None
        } for f in files]

    def delete_file(self, file_id: int) -> bool:
        """删除用户的RAG文件及其关联的chunks"""
        rag_file = self.db.query(RAGFile).filter(
            RAGFile.id == file_id,
            RAGFile.user_id == self.user_id
        ).first()

        if not rag_file:
            logger.warning(f"❌ 文件不存在: {file_id}")
            return False

        self.qdrant_service.delete_vectors_by_file(rag_file.file_path, self.user_id)

        self.db.query(RAGChunk).filter(RAGChunk.file_id == file_id).delete()

        self.db.delete(rag_file)
        self.db.commit()

        logger.info(f"✅ 删除文件成功: {rag_file.file_name}")
        return True