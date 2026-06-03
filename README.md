# RAG智能问答系统

基于FastAPI后端和Vue3前端的AI智能问答平台，支持**自定义RAG构建**和**结构化上下文管理**。

## 核心功能

- 📚 **自定义RAG构建** - 上传文档，自动构建知识库
- 📝 **文件内容总结** - 自动分析并总结上传文件内容
- 💬 **智能问答** - 基于RAG知识库进行问答
- 🔍 **闲聊检测** - 智能区分闲聊和专业问题
- 🔄 **RRF融合检索** - 倒数排名融合多检索器结果
- 🎯 **精排优化** - Cross-encoder精排提升检索精度
- 📊 **GSSC上下文管理** - Generate-Score-Select-Compress流水线
- 🧠 **结构化记忆** - JSON存储 + 索引加速查询
- 📱 **流式响应** - 实时流式输出AI回答

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 前端 | Vue 3 + Element Plus | 现代化UI框架 |
| 后端 | FastAPI + Python | 高性能API服务 |
| 向量存储 | Qdrant | 本地向量数据库 |
| 嵌入模型 | BGE | 中文语义嵌入 |
| 精排模型 | Cross-encoder | 文档相关性精排 |
| 记忆存储 | JSON文件系统 | 结构化JSON + 索引加速 |

## 项目结构

```
ai_coding_website/
├── backend/
│   ├── api/              # API路由
│   │   ├── auth.py       # 认证相关API
│   │   ├── chat.py       # 聊天相关API
│   │   └── rag.py        # RAG管理API
│   ├── core/             # 核心配置
│   │   ├── config.py     # 应用配置
│   │   ├── logger.py     # 日志配置
│   │   └── security.py   # JWT认证和安全工具
│   ├── db/               # 数据库
│   │   └── database.py   # 数据库连接
│   ├── models/           # 数据模型
│   │   └── user.py       # User, Conversation, Message模型
│   ├── schemas/          # Pydantic模式
│   │   └── schema.py     # 请求/响应模式
│   ├── services/         # 业务逻辑
│   │   ├── anthropic_service.py   # AI模型服务
│   │   ├── bge_service.py         # BGE嵌入服务
│   │   ├── bm25_service.py        # BM25全文检索服务
│   │   ├── cross_encoder_service.py # Cross-encoder精排服务
│   │   ├── memory_service.py      # 记忆服务（Qdrant存储）
│   │   ├── memory_storage.py      # 结构化JSON记忆存储
│   │   ├── qdrant_service.py      # Qdrant向量服务
│   │   ├── rag_service.py         # RAG检索服务（RRF融合）
│   │   ├── rrf_fusion.py           # RRF倒数排名融合
│   │   ├── gsc_pipeline.py        # GSSC上下文管理流水线
│   │   ├── context_manager.py      # 上下文管理器
│   │   └── chit_chat_detector.py   # 闲聊检测服务
│   ├── test/              # 测试脚本
│   │   ├── test_rag_refactor.py    # RAG重构测试
│   │   └── test_rrf_fusion.py      # RRF融合测试
│   ├── main.py            # FastAPI应用入口
│   └── requirements.txt   # Python依赖
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
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- uv（推荐的Python包管理器）
- Qdrant向量数据库（不再依赖MySQL存储RAG/记忆）
- BAAI/bge-small-zh-v1.5 BGE嵌入模型
- ms-marco-MiniLM-L-6-v2 Cross-encoder精排模型

### 后端启动

```bash
cd backend

# 安装依赖
uv sync

# 复制环境变量文件并配置
cp .env.example .env
# 编辑.env文件，填入你的API密钥等配置

# 启动服务器
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API文档地址**: http://localhost:8000/docs

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

### 结构化JSON记忆存储

长期记忆以JSON格式存储，使用index.json索引加速查询：

```
data/memories/
└── user_1/
    ├── index.json          # 索引文件
    ├── mem_20240101_0001.json
    ├── mem_20240101_0002.json
    └── ...
```

**查询加速**：
- 关键词搜索
- 类型查询
- 重要性查询
- 对话关联查询

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
│  • BGE向量嵌入 → Qdrant                 │
│  • BM25索引构建                          │
└─────────────────────────────────────────┘

用户提问
    │
    ▼
┌─────────────────────────────────────────┐
│  三路并行检索                            │
│  • 向量检索 (Qdrant)                    │
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
│  Cross-encoder精排                      │
│  取Top20候选，结果更精准                │
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

- ✅ 用户注册和登录
- ✅ JWT令牌认证（Access Token + Refresh Token）
- ✅ 流式AI响应输出
- ✅ 多对话管理
- ✅ 对话历史持久化
- ✅ 用户信息管理
- ✅ 自定义RAG知识库构建
- ✅ 文件内容自动总结
- ✅ 闲聊/专业问题智能区分
- ✅ RRF融合多检索器
- ✅ Cross-encoder精排优化
- ✅ GSSC上下文管理流水线
- ✅ 结构化JSON记忆存储

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API密钥 | - |
| `SECRET_KEY` | JWT加密密钥 | - |
| `QDRANT_HOST` | Qdrant服务地址 | localhost |
| `QDRANT_PORT` | Qdrant服务端口 | 6333 |

## 使用示例

### 1. 构建知识库

1. 登录系统
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
python test/test_rrf_fusion.py

# 测试RAG重构
python test/test_rag_refactor.py
```

## 架构演进

| 版本 | 变化 |
|------|------|
| v1.0 | 基础RAG + MySQL存储 |
| v2.0 | 引入Qdrant向量数据库 |
| v3.0 | RRF替代加权求和 |
| v4.0 | GSSC上下文管理流水线 |
| v5.0 | 结构化JSON记忆存储 |

## License

MIT
