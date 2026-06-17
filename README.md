# power-rag

**面向电力行业的智能知识库 RAG 问答系统**

基于 Spring Boot 3 / Spring AI / PgVector 构建，集成多代理架构与混合检索引擎

[![Java](https://img.shields.io/badge/Java-21-blue?logo=openjdk&logoColor=white)](https://adoptium.net/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![Vue](https://img.shields.io/badge/Vue-3.4-4FC08D?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PgVector-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-6+-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.8-FF6600?logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 项目简介

AgentHub 是一个专为电力行业设计的智能知识库管理与问答系统，支持 PDF / Excel / Word / 图片等多种文件格式的上传、解析、向量化存储，并基于 RAG（检索增强生成）技术提供智能问答服务。

系统采用**双模型路由 + 多代理**架构，支持深度思考模式、工具调用（联网搜索、知识库检索、偏差计算等）和混合检索（向量 + BM25），针对真实业务场景的核心痛点进行了系统性优化：

| 痛点 | 解决方案 |
|:---|:---|
| 文档格式多样，解析不完整 | Tika + PDFBox + OCR 多模态混合解析 |
| 单一检索方式召回率不足 | 向量检索 + BM25 双路召回 + RRF 融合排序 |
| 固定切分导致语义断裂 | 递归中文分块器 + EvidenceBlock 动态组装 |
| 上下文过长导致 Token 溢出 | GSC 结构化+压缩流水线 |
| 数据权限管理混乱 | 基于角色的细粒度数据隔离 |
| 单一模型能力有限 | 双模型路由（基座 + 微调）+ 辅助角色协作 |

---

## 系统架构

```text
┌───────────────────────────────────────────────────────────┐
│                      Vue 3 前端                            │
│              Element Plus + TypeScript                     │
└──────────────────────────┬────────────────────────────────┘
                           │ HTTPS / JWT
                           ▼
┌───────────────────────────────────────────────────────────┐
│                   Spring Boot 3.2                          │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ 用户认证  │ │ 知识库管理 │ │  聊天模块 │ │  代理引擎    │ │
│  │ Security │ │ Knowledge │ │   Chat   │ │ Agent Engine │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────┬───────┘ │
│                                                 │         │
│                 双模型路由                        │         │
│           ┌────────────┬────────────┐            │         │
│           │ 基座模型    │ 微调模型    │            │         │
│           │ (CHAT/工具) │ (KB_QA)    │            │         │
│           └─────┬──────┴─────┬──────┘            │         │
└─────────────────┼────────────┼───────────────────┼─────────┘
                  │            │                    │
       ┌──────────┐    ┌──────────────┐   ┌───────────────┐
       │  Ollama  │    │  DashScope   │   │   PostgreSQL  │
       │ (Worker) │    │(辅助角色:     │   │  + PgVector   │
       │ qwen3.5  │    │ Judge/Intent │   │   Redis/RMQ   │
       │          │    │ Reader/Rewr) │   │   阿里云 OSS   │
       └──────────┘    └──────────────┘   └───────────────┘
```

---

## 核心功能

### 1. 混合检索：向量 + BM25 + RRF 融合

**问题**：单一检索方式容易漏召回
- 向量检索：语义强但关键词精确度差
- BM25 检索：关键词精确但语义理解弱

**方案**：双路并行召回 + RRF 融合排序

```text
Query
  │
  ├──▶ 向量检索 (PgVector)  ──┐
  │                           │
  └──▶ BM25 + Jieba 检索  ──┤
                              │
                       RRF 融合排序
                              │
                        Top-K 结果
```

**RRF 融合公式**：`RRF(d) = Σ 1/(k + rank_i(d))`

**工程优化**：
- 并行检索（CompletableFuture + 专用线程池）
- 批量查询避免 N+1
- 超时控制（20s）+ 自动降级

详见：`HybridSearchServiceImpl.java:101-153`、`RRFFusion.java`

---

### 2. EvidenceBlock：动态证据链组装

**问题**：固定长度切分导致"半句话"，语义不完整

**方案**：基于元数据动态合并相邻 chunk

**合并条件**：
- 同文件（filename）
- 页码相邻（±2）
- chunkIndex 连续

**证据类型**：
| 类型 | 说明 |
|:---|:---|
| SINGLE | 单个 chunk，未合并 |
| MERGED | 多个连续 chunk 合并 |
| CROSS_PAGE | 跨页合并 |
| OCR_CONTENT | OCR 提取内容 |

**效果**：回答完整性提升，可精确溯源至页码 + chunk 偏移

详见：`EvidenceAssembly.java`、`EvidenceBlock.java`

---

### 3. 智能中文分块器

**问题**：按字符数固定切分会截断句子

**方案**：递归分割策略

```
优先级：段落 → 换行 → 句号 → 标点 → 空格
大小：450-600 字符/块
重叠：150 字符（保证上下文连续）
```

**效果**：语义完整性提升，避免"半句话"

详见：`VectorStoreHelper.java:113-200`

---

### 4. 多模态文档解析 Pipeline

**方案**：混合解析 + 页面级并行 OCR

```text
PDF 文档
    │
    ├──▶ 第一遍：快速文本提取（串行）
    │       └──▶ 成功 → 直接使用
    │
    └──▶ 第二遍：并行 OCR 处理
            └──▶ 文本提取失败 / 扫描件
```

**技术栈**：
| 技术 | 用途 |
|:---|:---|
| Apache Tika 2.9 | 通用文档解析（Word、Excel 等） |
| Apache PDFBox 3.0 | 结构化 PDF 解析 |
| Qwen OCR | 扫描件文字识别 |

**工程优化**：
- 页面级并行 OCR（速度提升 3-5 倍）
- 自动降级机制
- Semaphore 并发控制

详见：`VectorStoreHelper.java:293-357`、`QwenOcrDocumentReader.java`

---

### 5. Rerank 精排 + 自动降级

**模型**：gte-rerank-v2（阿里云 DashScope）

**工程保障**：
- 10 秒超时控制
- 异常自动降级（返回原始排序结果）
- 保留原始分数

详见：`DashScopeRerankerConfig.java`

---

### 6. 双模型路由引擎（Agent Engine）

**架构**：本地 Ollama 双模型 + DashScope 云端辅助角色

```text
用户提问
    │
    ▼
意图识别 (qwen-plus via DashScope)
    │
    ├── KB_QA ──▶ 微调模型 (qwen3.5-agenthub) + 知识库检索
    │
    └── CHAT ───▶ 基座模型 (qwen3.5) + 联网搜索/计算等工具
    │
    ▼
Judge 审计 (glm-4.7 via DashScope, 异步不阻塞)
```

| 角色 | 模型 | 职责 |
|:---|:---|:---|
| **Worker (基座)** | qwen3.5 (Ollama 本地) | CHAT 闲聊 + 通用工具调用 |
| **Worker (微调)** | qwen3.5-agenthub (Ollama 本地) | KB_QA 知识库问答 |
| **Intent** | qwen-plus (DashScope) | 意图识别：LLM 优先，规则兜底 |
| **Judge** | glm-4.7 (DashScope) | 合规审计，回答质量评估 |
| **Reader** | kimi-k2-thinking (DashScope) | 超长文档阅读 |
| **QueryRewrite** | qwen-plus (DashScope) | 口语化查询改写 |

**可用工具**：
| 工具 | 说明 |
|:---|:---|
| web_search | Tavily 互联网搜索 |
| knowledge_search | 知识库 RAG 检索 |
| deviation_calculation | 偏差电费计算 |
| get_current_time | 获取当前时间 |

- 意图识别策略：LLM（qwen-plus）优先 → 规则关键词兜底
- 支持 qwen3.5 深度思考模式（thinking 输出）
- GSC 流水线：证据结构化 + Token 预算压缩
- Case 快照冻结 + Judge 异步审计
- 对话记忆滑动窗口

---

### 7. 细粒度权限与数据隔离

| 角色 | 全局知识库 | 私有知识库 |
|:---|:---:|:---:|
| 管理员 (admin) | 读写 | 读写全部 |
| 普通用户 (user) | 只读 | 仅读写自己的 |

---

## 技术栈

### 后端

| 模块 | 技术 |
|:---|:---|
| 框架 | Java 21, Spring Boot 3.2 |
| 安全 | Spring Security + JWT 0.12 |
| 数据库 | PostgreSQL + PgVector |
| ORM | MyBatis-Plus 3.5.5 |
| AI 编排 | Spring AI 1.1 + Ollama + 阿里云 DashScope |
| 文档解析 | Apache Tika 2.9 + PDFBox 3.0 |
| 缓存 | Redis + Lettuce 连接池 |
| 消息队列 | RabbitMQ |
| 对象存储 | 阿里云 OSS（前端直传） |
| 中文分词 | Jieba-analysis 1.0.2 |
| API 文档 | Knife4j 4.4 (Swagger) |

### 前端

| 模块 | 技术 |
|:---|:---|
| 框架 | Vue 3.4 (Composition API) |
| 语言 | TypeScript 5.0+ |
| 构建 | Vite 5.0+ |
| UI | Element Plus 2.5+ |
| 状态管理 | Pinia 2.1+ |
| HTTP | Axios 1.6+ |

---

## 项目结构

```text
com.agenthub
├── common/                          # 通用模块
│   ├── base/                        #   BaseEntity, BaseController
│   ├── core/                        #   AjaxResult, PageQuery, PageResult
│   ├── enums/                       #   UserRole 角色枚举
│   └── exception/                   #   全局异常处理
│
├── framework/                       # 框架配置
│   ├── config/                      #   CORS, MyBatisPlus, ThreadPool
│   └── security/                    #   JWT 过滤器, Token 服务, 认证处理
│
├── system/                          # 系统管理
│   ├── controller/                  #   AuthController, SysUserController
│   ├── domain/                      #   SysUser, LoginUser, LoginBody
│   └── service/                     #   用户服务实现
│
├── search/                          # 检索模块 ⭐
│   ├── service/impl/                #   HybridSearchServiceImpl, Bm25SearchServiceImpl
│   ├── util/                        #   RRFFusion, ChineseTokenizer
│   └── mapper/                      #   数据访问层
│
├── ai/                              # AI 模块 ⭐
│   ├── service/                     #   PowerKnowledgeService, EvidenceAssembly
│   ├── utils/                       #   VectorStoreHelper, QwenOcrDocumentReader
│   └── config/                      #   DashScopeRerankerConfig
│
├── knowledge/                       # 知识库模块
│   ├── controller/                  #   KnowledgeBaseController, ChatController
│   ├── domain/                      #   KnowledgeBase, ChatHistory, ChatSession, VO
│   ├── mapper/                      #   数据访问层
│   └── service/                     #   知识库 & 聊天服务接口与实现
│
├── prompt/                          # Prompt 模板管理
│   ├── service/                     #   Prompt 服务
│   └── interceptor/                 #   Prompt 上下文拦截器
│
└── agent_engine/                    # 双模型路由引擎
    ├── config/                      #   LLMService, OllamaLLMService, AgentModelFactory
    ├── controller/                  #   AgentV2Controller, EvalDashboardController
    ├── core/                        #   ChatAgent 接口, OllamaChatAgent, SinglePassExecutor
    ├── model/                       #   AgentContext, ToolCall, IntentResult
    ├── service/                     #   IntentRecognitionService, ReflectionService
    └── tool/                        #   WebSearchTool, KnowledgeSearchTool, DeviationCalculationTool
```

---

## 快速开始

### 环境要求

- JDK 21+
- PostgreSQL 14+（需安装 pgvector 扩展）
- Redis 6+
- RabbitMQ 3.8+
- Maven 3.8+
- Node.js 18+（前端）

### 1. 数据库初始化

```bash
createdb agenthub
psql -d agenthub -c "CREATE EXTENSION vector;"
psql -d agenthub -f src/main/resources/sql/schema.sql
```

### 2. 后端配置

复制 `application.yml` 并创建 `application-local.yml`（已被 gitignore，不会提交）：

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/agenthub
    username: your_username
    password: your_password

  data:
    redis:
      host: localhost
      port: 6379

  rabbitmq:
    host: localhost
    port: 5672

  # DashScope 云端模型（辅助角色：Judge / Intent / Reader / QueryRewrite）
  ai:
    dashscope:
      api-key: your_dashscope_api_key
    # Ollama 本地模型（Worker 角色：基座 + 微调）
    ollama:
      base-url: http://localhost:11434
      chat:
        options:
          model: qwen3.5:latest

# 双模型路由配置
app:
  llm:
    provider: ollama              # ollama (本地) | dashscope (云端)
    base-model: qwen3.5:latest           # 基座模型 (CHAT + 工具调用)
    finetuned-model: qwen3.5-agenthub:latest  # 微调模型 (KB_QA 知识库问答)

aliyun:
  oss:
    endpoint: oss-cn-shanghai.aliyuncs.com
    access-key-id: your_key
    access-key-secret: your_secret
    bucket-name: your_bucket

# Agent 工具配置
agent:
  tavily:
    api-key: your_tavily_api_key     # https://tavily.com 注册获取

knowledge:
  retrieval:
    enable-hybrid: true    # 启用混合检索
    top-k: 5
```

### 3. 启动后端

```bash
mvn clean install
mvn spring-boot:run
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 5. 访问系统

| 地址 | 说明 |
|:---|:---|
| `http://localhost:8080` | 后端 API |
| `http://localhost:8080/doc.html` | Swagger 接口文档 |
| `http://localhost:5173` | 前端页面 |

### 默认账号

| 角色 | 用户名 | 密码 |
|:---|:---|:---|
| 管理员 | `admin` | `admin123` |
| 普通用户 | `user` | `user123` |

---

## API 概览

### 认证

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| POST | `/auth/login` | 用户登录 |
| POST | `/auth/register` | 用户注册 |

### 知识库

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| GET | `/knowledge/base/upload/policy` | 获取 OSS 上传凭证 |
| POST | `/knowledge/base/upload/callback` | 上传回调（创建记录） |
| GET | `/knowledge/base/list` | 分页查询知识库 |
| DELETE | `/knowledge/base/{ids}` | 删除知识库 |

### 智能问答

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| POST | `/knowledge/chat/ask` | 普通问答 |
| POST | `/knowledge/chat/stream` | 流式问答（SSE） |
| POST | `/api/v2/agent/chat` | V2 代理流式问答（深度思考 + 工具调用） |
| GET | `/knowledge/chat/sessions` | 获取会话列表 |
| GET | `/knowledge/chat/history/{sessionId}` | 获取聊天历史 |
| DELETE | `/knowledge/chat/history/{sessionId}` | 清空聊天历史 |

---

## 部署

### Docker Compose

```bash
docker-compose up -d        # 构建并启动
docker-compose logs -f       # 查看日志
docker-compose down          # 停止服务
```

### 生产环境变量

```bash
# 数据库
SPRING_DATASOURCE_URL=jdbc:postgresql://host:5432/agenthub
SPRING_DATASOURCE_USERNAME=username
SPRING_DATASOURCE_PASSWORD=password

# Redis
SPRING_DATA_REDIS_HOST=redis-host
SPRING_DATA_REDIS_PASSWORD=redis-password

# RabbitMQ
SPRING_RABBITMQ_HOST=mq-host
SPRING_RABBITMQ_USERNAME=username
SPRING_RABBITMQ_PASSWORD=password

# DashScope（辅助角色）
AI_DASHSCOPE_API_KEY=your_api_key

# Ollama（Worker 角色）
SPRING_AI_OLLAMA_BASE_URL=http://ollama-host:11434

# 双模型路由
APP_LLM_PROVIDER=ollama
APP_LLM_BASE_MODEL=qwen3.5:latest
APP_LLM_FINETUNED_MODEL=qwen3.5-agenthub:latest

# 阿里云 OSS
ALIYUN_OSS_ENDPOINT=oss-cn-shanghai.aliyuncs.com
ALIYUN_OSS_ACCESSKEYID=your_key
ALIYUN_OSS_ACCESSKEYSECRET=your_secret
ALIYUN_OSS_BUCKETNAME=your_bucket

# Tavily 搜索
AGENT_TAVILY_API_KEY=your_tavily_key
```

---

## 性能优化

| 优化点 | 方案 | 效果 |
|:---|:---|:---|
| 并行检索 | CompletableFuture + 专用线程池 | 召回耗时降低 50% |
| 批量查询 | 避免 N+1，一次 SQL 查完 | BM25 检索 2-3s 内完成 |
| 并行 OCR | 页面级并行处理 | OCR 速度提升 3-5 倍 |
| 降级策略 | Rerank 失败自动降级 | 服务可用性 99.9%+ |

---

## 适用场景

- 企业内部知识库问答系统
- 电力行业文档检索与智能问答
- 技术/合规文档解析与检索
- PDF 扫描件智能识别
- AI 助手 / 多代理系统集成

---

## License

[MIT License](LICENSE)
