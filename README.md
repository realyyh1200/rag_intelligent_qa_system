# RAG智能问答系统

基于FastAPI后端和Vue3前端的AI智能问答平台，支持**自定义RAG构建**和**结构化上下文管理**。

## 核心功能

- 📚 **自定义RAG构建** - 上传文档，自动构建知识库
- 📝 **文件内容总结** - 自动分析并总结上传文件内容
- 💬 **智能问答** - 基于RAG知识库进行问答
- 🔍 **闲聊检测** - 智能区分闲聊和专业问题
- 🔄 **RRF融合检索** - 倒数排名融合多检索器结果
- 🎯 **精排优化** - 轻量级精排提升检索精度
- 📊 **GSSC上下文管理** - Generate-Score-Select-Compress流水线
- 🧠 **会话记忆** - JSONL格式存储，支持会话级记忆
- 📱 **流式响应** - 实时流式输出AI回答

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 前端 | Vue 3 + Element Plus | 现代化UI框架 |
| 后端 | FastAPI + Python | 高性能API服务 |
| 向量存储 | ChromaDB | 本地向量数据库 |
| 嵌入模型 | 轻量级本地嵌入 | 词频+哈希向量表示 |
| 精排模型 | 轻量级Jaccard相似度 | 无需外部模型依赖 |
| 记忆存储 | SessionStorage (JSONL) | 会话级记忆存储 |
| AI模型 | OpenAI API | 支持多种模型 |

## 项目结构

```
ai_coding_website/
├── backend/
│   ├── api/              # API路由
│   │   ├── chat.py       # 聊天相关API
│   │   ├── rag.py        # RAG管理API
│   │   └── user.py       # 用户信息API
│   ├── core/             # 核心配置
│   │   ├── config.py     # 应用配置
│   │   └── logger.py     # 日志配置
│   ├── db/               # 数据库
│   │   └── database.py   # 数据库连接
│   ├── models/           # 数据模型
│   │   ├── conversation.py  # 会话模型
│   │   ├── memory.py        # 记忆模型
│   │   └── rag.py           # RAG文档模型
│   ├── schemas/          # Pydantic模式
│   │   └── schema.py     # 请求/响应模式
│   ├── services/         # 业务逻辑
│   │   ├── openai_service.py    # OpenAI模型服务
│   │   ├── embedding_service.py # 轻量级嵌入服务
│   │   ├── bm25_service.py      # BM25全文检索服务
│   │   ├── chroma_service.py    # ChromaDB向量服务
│   │   ├── session_storage.py   # JSONL会话存储
│   │   ├── memory_service.py    # 记忆服务
│   │   ├── rag_service.py       # RAG检索服务（RRF融合）
│   │   ├── rrf_fusion.py        # RRF倒数排名融合
│   │   ├── gsc_pipeline.py      # GSSC上下文管理流水线
│   │   ├── context_manager.py   # 上下文管理器
│   │   └── chit_chat_detector.py # 闲聊检测服务
│   ├── test/             # 测试脚本
│   │   ├── test_rag_refactor.py    # RAG重构测试
│   │   ├── test_rrf_fusion.py      # RRF融合测试
│   │   └── test_e2e.py             # 端到端测试
│   ├── data/             # 数据存储
│   │   ├── chroma/       # ChromaDB向量数据
│   │   └── memories/     # 会话记忆JSONL文件
│   ├── main.py           # FastAPI应用入口
│   └── pyproject.toml    # Python依赖配置
│
└── frontend/
    ├── src/
    │   ├── api/          # API调用
    │   ├── components/   # Vue组件
    │   ├── router/       # 路由配置
    │   ├── store/        # Pinia状态管理
    │   ├── views/        # 页面视图
    │   ├── App.vue
    │   └── main.js
    ├── index.html
    └── package.json
    └── vite.config.js    # Vite配置（含代理）
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- uv（推荐的Python包管理器）

### 后端启动

```bash
cd backend

# 安装依赖
uv sync

# 复制环境变量文件并配置
cp .env.example .env
# 编辑.env文件，填入你的API密钥等配置

# 启动服务器
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**API文档地址**: http://localhost:8080/docs

### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

**访问地址**: http://localhost:5173

## 核心架构

### RRF（倒数排名融合）

替代传统的加权求和，实现多检索器的鲁棒融合：

```
RRF_score(d) = Σ 1/(k + rank(d))

其中:
- d 是文档
- rank(d) 是文档在检索器中的排名
- k = 60 (平滑参数)
```

**优势**：
- 无需分数归一化
- 对排名敏感而非分数值
- 单检索器异常不影响整体
- 无需调参

### GSSC上下文管理流水线

四阶段上下文构建流程：

```
┌─────────────────────────────────────────┐
│  1️⃣ GATHER（汇集）                      │
│  从多源收集候选信息：                     │
│  • 系统指令（优先级10）                  │
│  • 用户消息（优先级10）                  │
│  • 历史对话（优先级5）                   │
│  • RAG召回（优先级7）                   │
│  • 记忆（优先级6）                       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  2️⃣ SELECT（选择）                      │
│  Score = (0.7×相关性 + 0.3×新近性) × 优先级 │
│  → 贪心算法在Token预算内选择             │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  3️⃣ STRUCTURE（结构化）                 │
│  【角色与策略】【任务】【状态】           │
│  【证据(RAG)】【上下文】【输出要求】     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  4️⃣ COMPRESS（压缩）                    │
│  兜底智能摘要，保留结构完整性            │
└─────────────────────────────────────────┘
```

### 会话记忆存储 (SessionStorage)

使用JSONL格式存储会话级记忆：

```
data/memories/
└── user_1/
    ├── sessions.json          # 会话索引文件
    ├── sess_20260611_222750_1.jsonl
    ├── sess_20260611_223203_2.jsonl
    └── ...
```

**特点**：
- 每个会话独立文件
- 支持追加写入
- 自动索引管理
- 会话级记忆隔离

## RAG工作流程

```
用户上传文档
    │
    ▼
┌─────────────────────────────────────────┐
│  文档解析                                │
│  • 支持TXT、DOCX、PDF等                 │
│  • 智能chunk切分 (512 tokens, 64 overlap)│
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  并行处理                                │
│  • 轻量级向量嵌入 → ChromaDB            │
│  • BM25索引构建                          │
└─────────────────────────────────────────┘

用户提问
    │
    ▼
┌─────────────────────────────────────────┐
│  三路并行检索                            │
│  • 向量检索 (ChromaDB)                  │
│  • BM25检索                             │
│  • 关键词检索                           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  RRF融合                                │
│  融合多检索器排名，结果更鲁棒            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  轻量级精排                              │
│  Jaccard相似度精排，无需外部模型         │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  GSSC上下文构建                          │
│  动态构建最优上下文                      │
└─────────────────────────────────────────┘
                    │
                    ▼
               AI模型回答
```

## 功能特性

- ✅ 流式AI响应输出
- ✅ 多对话管理
- ✅ 对话历史持久化
- ✅ 自定义RAG知识库构建
- ✅ 文件内容自动总结
- ✅ 闲聊/专业问题智能区分
- ✅ RRF融合多检索器
- ✅ 轻量级精排优化
- ✅ GSSC上下文管理流水线
- ✅ 会话级记忆存储

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `API_KEY` | OpenAI API密钥 | - |
| `API_BASE` | OpenAI API地址 | https://api.openai.com/v1 |
| `MODEL` | 使用的模型名称 | gpt-4o |
| `SECRET_KEY` | JWT加密密钥 | - |
| `CHROMA_DB_PATH` | ChromaDB数据路径 | ./data/chroma |

## 使用示例

### 1. 构建知识库

1. 打开系统
2. 上传文档文件
3. 系统自动构建RAG索引（向量化 + BM25）

### 2. 智能问答

```
用户：如何学习Python？
AI：根据参考文档，学习Python需要...
【参考文献】
- Python入门指南.pdf
```

### 3. 文件总结

```
用户：总结一下这份文档
AI：这份文档主要讲述了...
```

## 开发说明

### 常用命令

| 命令 | 说明 |
|------|------|
| `uv sync` | 同步并安装所有依赖 |
| `uv add package` | 添加新依赖 |
| `uv remove package` | 移除依赖 |
| `uv lock` | 更新锁定文件 |
| `uv run command` | 在虚拟环境中运行命令 |

### 测试脚本

```bash
# 测试RRF融合
cd backend
uv run python test/test_rrf_fusion.py

# 测试RAG重构
uv run python test/test_rag_refactor.py

# 端到端测试
uv run python test/test_e2e.py
```

## 架构演进

| 版本 | 变化 |
|------|------|
| v1.0 | 基础RAG + MySQL存储 |
| v2.0 | 引入Qdrant向量数据库 |
| v3.0 | RRF替代加权求和 |
| v4.0 | GSSC上下文管理流水线 |
| v5.0 | 结构化JSON记忆存储 |
| v6.0 | Anthropic → OpenAI，移除认证系统 |
| v7.0 | ChromaDB替代Qdrant，轻量级嵌入方案 |
| v8.0 | SessionStorage JSONL记忆存储 |

## License

MIT