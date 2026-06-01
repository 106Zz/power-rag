package com.agenthub.api.agent_engine.core;

import com.agenthub.api.agent_engine.capability.ToolRegistry;
import com.agenthub.api.agent_engine.config.LLMService;
import com.agenthub.api.agent_engine.model.*;
import com.agenthub.api.agent_engine.service.IntentRecognitionService;
import com.agenthub.api.agent_engine.service.ReflectionService;
import com.agenthub.api.agent_engine.tool.AgentTool;
import com.agenthub.api.ai.domain.llm.StreamCallback;
import com.agenthub.api.ai.service.PowerKnowledgeService;
import com.agenthub.api.ai.service.LLMCacheService;
import com.agenthub.api.ai.service.gssc.GSSCService;
import com.agenthub.api.prompt.builder.CaseSnapshotBuilder;
import com.agenthub.api.prompt.enums.CaseStatus;
import com.agenthub.api.prompt.enums.Scenario;
import com.agenthub.api.prompt.service.ICaseSnapshotService;
import com.agenthub.api.prompt.service.ISysPromptService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.memory.ChatMemoryRepository;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * 单次执行器（双模型路由版）
 * <p>实现 "意图识别 → 预检索 → 模型分流 → LLM输出 → Judge事后审计" 的单次流程</p>
 *
 * <h3>模型分流策略：</h3>
 * <ul>
 *   <li>KB_QA → 微调模型 (qwen3.5-agenthub) + 只给 knowledge_search 工具</li>
 *   <li>CHAT → 基座模型 (qwen3.5) + 给所有工具，允许按需调用 knowledge_search 兜底</li>
 * </ul>
 *
 * <h3>执行流程：</h3>
 * <pre>
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  1. 意图识别 (IntentRecognition)                                    │
 * │     └─ 识别用户意图: KB_QA / CHAT / UNKNOWN                       │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  2. 预检索 (PreRetrieval) - 仅 KB_QA                                │
 * │     └─ QueryRewrite → 多路召回 → Reranker → EvidenceAssembly       │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  3. 构建消息 (BuildMessages)                                        │
 * │     ├─ SystemMessage: 系统提示词                                    │
 * │     ├─ HistoryMessages: 滑动窗口(最近N条)                           │
 * │     ├─ EvidenceMessage: 证据文本（GSC 结构化+压缩）                  │
 * │     └─ UserMessage: 用户当前问题                                    │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  4. LLM 生成 (Stream) - 按意图分流模型                              │
 * │     ├─ KB_QA → 微调模型 + knowledge_search                         │
 * │     └─ CHAT → 基座模型 + 其他工具                                   │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  5. 保存记忆 (SaveMemory)                                           │
 * │     └─ 保存 User + Assistant → sys_ai_memory                        │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  6. Judge 审计 (Async) - 不阻塞响应                                 │
 * │     └─ 异步评估回答质量，用于事后分析                               │
 * └─────────────────────────────────────────────────────────────────────┘
 * </pre>
 *
 * @author AgentHub
 * @since 2026-02-07
 */
@Slf4j
public class SinglePassExecutor {

    // ==================== 依赖服务 ====================

    private final IntentRecognitionService intentRecognition;
    private final PowerKnowledgeService powerKnowledgeService;
    private final ToolRegistry toolRegistry;
    private final ReflectionService reflectionService;
    private final ChatMemoryRepository chatMemoryRepository;
    private final ISysPromptService sysPromptService;
    private final ICaseSnapshotService caseSnapshotService;
    private final ObjectMapper objectMapper;
    private final LLMService llmService;
    private final GSSCService gscService;
    private final LLMCacheService llmCacheService;

    // ==================== 线程池 ====================

    private final Executor judgeExecutor;
    private final Executor agentWorkerExecutor;

    // ==================== 常量配置 ====================

    /**
     * 滑动窗口大小 (保留最近 N 条历史消息)
     */
    private static final int MEMORY_WINDOW_SIZE = 5;

    /** KB_QA 知识库问答提示词 */
    private static final String SYSTEM_PROMPT_KBQA = "SYSTEM-RAG-LITE";

    /** CHAT 闲聊提示词 */
    private static final String SYSTEM_PROMPT_CHAT = "SYSTEM-CHAT-v1.0";

    // ==================== 双模型配置 ====================

    /** 基座模型：CHAT 闲聊 + 通用工具调用 */
    private final String baseModel;

    /** 微调模型：KB_QA 知识库问答 */
    private final String finetunedModel;

    // ==================== 构造器 ====================

    public SinglePassExecutor(
            IntentRecognitionService intentRecognition,
            PowerKnowledgeService powerKnowledgeService,
            ToolRegistry toolRegistry,
            ReflectionService reflectionService,
            ChatMemoryRepository chatMemoryRepository,
            ISysPromptService sysPromptService,
            ICaseSnapshotService caseSnapshotService,
            ObjectMapper objectMapper,
            LLMService llmService,
            GSSCService gscService,
            LLMCacheService llmCacheService,
            Executor judgeExecutor,
            Executor agentWorkerExecutor,
            String baseModel,
            String finetunedModel) {
        this.intentRecognition = intentRecognition;
        this.powerKnowledgeService = powerKnowledgeService;
        this.toolRegistry = toolRegistry;
        this.reflectionService = reflectionService;
        this.chatMemoryRepository = chatMemoryRepository;
        this.sysPromptService = sysPromptService;
        this.caseSnapshotService = caseSnapshotService;
        this.objectMapper = objectMapper;
        this.llmService = llmService;
        this.gscService = gscService;
        this.llmCacheService = llmCacheService;
        this.judgeExecutor = judgeExecutor;
        this.agentWorkerExecutor = agentWorkerExecutor;
        this.baseModel = baseModel;
        this.finetunedModel = finetunedModel;
    }

    // ==================== 核心执行方法 ====================

    /**
     * 流式执行单次流程
     */
    public Flux<String> executeStream(AgentContext context) {
        log.info("[SinglePass] 开始执行: sessionId={}, query={}",
                context.getSessionId(), context.getQuery());

        StringBuilder fullAnswer = new StringBuilder();
        StringBuilder thinkingContent = new StringBuilder();
        final boolean[] collectingThinking = new boolean[]{false};

        return Flux.create(sink -> {
            try {
                // ==================== 1. 意图识别 ====================
                IntentResult intentResult = intentRecognition.recognizeIntent(context.getQuery());
                context.setIntent(intentResult.intent());
                context.setIntentConfidence(intentResult.confidence());

                log.info("[SinglePass] 意图识别结果: intent={}, confidence={}, needsPreRetrieval={}",
                        intentResult.intent(), intentResult.confidence(), intentResult.needsPreRetrieval());

                // ==================== 2. 预检索 (仅 KB_QA) ====================
                String evidenceContext = "";
                if (intentResult.needsPreRetrieval()) {
                    log.info("[SinglePass] 触发预检索: query={}", context.getQuery());

                    sink.next("__TOOL_CALL__:knowledge_search\n");

                    long retrieveStart = System.currentTimeMillis();
                    try {
                        var knowledgeResult = powerKnowledgeService.retrieve(
                                new com.agenthub.api.ai.domain.knowledge.PowerKnowledgeQuery(
                                        context.getQuery(),
                                        5, null, null
                                )
                        );

                        evidenceContext = formatKnowledgeResult(knowledgeResult);
                        context.setPreRetrievedContent(evidenceContext);
                        context.setPreRetrievalDone(true);

                        // 保存 sources 到 context，供前端渲染
                        if (knowledgeResult.sources() != null && !knowledgeResult.sources().isEmpty()) {
                            List<AgentContext.SourceDocument> sources = knowledgeResult.sources().stream()
                                    .map(src -> AgentContext.SourceDocument.builder()
                                            .filename(src.filename())
                                            .downloadUrl(src.downloadUrl())
                                            .build())
                                    .toList();
                            context.setSources(sources);
                            log.info("[SinglePass] 保存 sources 到 context: {} 个文件", sources.size());
                        }

                        log.info("[SinglePass] 预检索完成: 耗时={}ms, evidenceBlocks={}",
                                System.currentTimeMillis() - retrieveStart,
                                knowledgeResult.getEvidenceBlockCount());

                    } catch (Exception e) {
                        log.error("[SinglePass] 预检索失败", e);
                        evidenceContext = "【检索失败】知识库暂时无法访问，请稍后重试。";
                    }
                }

                // ==================== 3. 构建消息 ====================
                List<Message> messages = buildMessages(context, evidenceContext);
                log.debug("[SinglePass] 消息构建完成: 数量={}", messages.size());

                // ==================== 4. LLM 生成 (流式) ====================
                doStreamChat(messages, context)
                        .doOnNext(chunk -> {
                            if (!chunk.isEmpty()) {
                                appendChunkForPersistence(chunk, fullAnswer, thinkingContent, collectingThinking);
                                sink.next(chunk);
                            }
                        })
                        .doOnComplete(() -> {
                            log.info("[SinglePass] LLM 生成完成: 回答长度={}", fullAnswer.length());

                            // ==================== 5. 保存记忆 ====================
                            saveToMemory(context, thinkingContent.toString(), fullAnswer.toString());

                            // ==================== 6. 异步 Judge 审计 ====================
                            log.info("[SinglePass] 开始异步 Judge 审计: sessionId={}, answerLength={}",
                                    context.getSessionId(), fullAnswer.length());
                            asyncJudge(context, fullAnswer.toString());

                            sink.complete();
                        })
                        .doOnError(sink::error)
                        .subscribe();

            } catch (Exception e) {
                log.error("[SinglePass] 执行异常", e);
                sink.error(e);
            }
        });
    }

    // ==================== 消息构建 ====================

    /**
     * 统一构建消息列表
     * <p>
     * 消息结构保持角色分离（System/User/Assistant），
     * GSC 只负责证据部分的结构化 + Token 预算压缩
     * </p>
     *
     * @param context        Agent 上下文
     * @param evidenceContext 预检索得到的证据文本（可能为空）
     * @return 消息列表
     */
    private List<Message> buildMessages(AgentContext context, String evidenceContext) {
        List<Message> messages = new ArrayList<>();

        // 1. SystemMessage（独立角色）
        String systemPrompt = buildSystemPrompt(context);
        messages.add(new SystemMessage(systemPrompt));

        // 2. 历史消息（滑动窗口，保持 User/Assistant 角色分离）
        List<Message> history = loadRecentHistory(context.getSessionId());
        if (!history.isEmpty()) {
            messages.addAll(history);
            log.debug("[SinglePass] 加载历史记录: 数量={}", history.size());
        }

        // 3. 证据上下文（如果有，通过 GSC 结构化+压缩后作为 UserMessage）
        if (StringUtils.hasText(evidenceContext)) {
            String processedEvidence = processEvidence(evidenceContext, context.getQuery());
            messages.add(new UserMessage(processedEvidence));
        }

        // 4. 用户当前问题（独立 UserMessage，让 LLM 明确知道要回答什么）
        messages.add(new UserMessage(context.getQuery()));

        return messages;
    }

    /**
     * 处理证据文本：结构化 + 压缩
     * <p>
     * 证据已经过 Reranker 精排 + EvidenceAssembly 组装，
     * 这里只做格式化和 Token 预算控制
     * </p>
     */
    private String processEvidence(String evidenceContext, String userQuery) {
        if (gscService.isEnabled()) {
            return gscService.processEvidence(evidenceContext, userQuery);
        }
        // 未启用 GSC：直接返回原始格式
        return evidenceContext;
    }

    /**
     * 加载最近的历史消息（统一使用滑动窗口）
     * <p>
     * 从 ChatMemoryRepository 加载，过滤思考标签，滑动窗口控制数量。
     * 不再使用 Jaccard 词法匹配重排——对于多轮对话，最近的消息本身就是最相关的。
     * </p>
     *
     * @param sessionId 会话ID
     * @return 最近的历史消息列表
     */
    private List<Message> loadRecentHistory(String sessionId) {
        try {
            List<Message> allHistory = chatMemoryRepository.findByConversationId(sessionId);
            if (allHistory == null || allHistory.isEmpty()) {
                return new ArrayList<>();
            }

            // 过滤掉思考标签，防止模型模仿内部标记
            List<Message> filteredHistory = new ArrayList<>();
            for (Message msg : allHistory) {
                if (msg instanceof AssistantMessage assistantMsg) {
                    String originalContent = assistantMsg.getText();
                    if (originalContent != null) {
                        String filteredContent = originalContent
                                .replaceAll("@@THINK_START@@.*?@@THINK_END@@", "")
                                .trim();
                        filteredHistory.add(new AssistantMessage(filteredContent));
                    } else {
                        filteredHistory.add(assistantMsg);
                    }
                } else {
                    filteredHistory.add(msg);
                }
            }

            // 滑动窗口：只保留最近 N 条
            if (filteredHistory.size() <= MEMORY_WINDOW_SIZE) {
                return filteredHistory;
            }

            List<Message> windowed = filteredHistory.subList(
                    filteredHistory.size() - MEMORY_WINDOW_SIZE,
                    filteredHistory.size()
            );
            log.debug("[SinglePass] 滑动窗口控制历史消息: {} -> {}",
                    filteredHistory.size(), windowed.size());
            return windowed;

        } catch (Exception e) {
            log.error("[SinglePass] 加载历史记录失败: sessionId={}", sessionId, e);
            return new ArrayList<>();
        }
    }

    // ==================== 记忆保存 ====================

    /**
     * 保存到记忆
     */
    private void saveToMemory(AgentContext context, String thinking, String answer) {
        try {
            List<Message> history = chatMemoryRepository.findByConversationId(context.getSessionId());
            if (history == null) {
                history = new ArrayList<>();
            }

            history.add(new UserMessage(context.getQuery()));

            String fullResponse = buildFullResponse(thinking, answer);
            history.add(new AssistantMessage(fullResponse));

            // 滑动窗口控制 (保存前截取)
            if (history.size() > MEMORY_WINDOW_SIZE) {
                history = history.subList(history.size() - MEMORY_WINDOW_SIZE, history.size());
            }

            chatMemoryRepository.saveAll(context.getSessionId(), history);
            log.debug("[SinglePass] 记忆已保存: sessionId={}, 消息数={}",
                    context.getSessionId(), history.size());

        } catch (Exception e) {
            log.error("[SinglePass] 保存记忆失败: sessionId={}", context.getSessionId(), e);
        }
    }

    private String buildFullResponse(String thinking, String answer) {
        StringBuilder sb = new StringBuilder();
        if (StringUtils.hasText(thinking)) {
            sb.append("```\n").append(thinking).append("\n```\n\n");
        }
        sb.append(answer != null ? answer : "");
        return sb.toString();
    }

    // ==================== 工具调用 ====================

    /**
     * 执行工具调用并基于结果继续生成
     */
    private void executeToolsAndContinue(
            List<ToolCall> toolCalls,
            List<Message> originalMessages,
            AgentContext context,
            reactor.core.publisher.FluxSink<String> sink) {

        log.info("[SinglePass] 开始执行工具，数量: {}", toolCalls.size());

        SecurityContext mainThreadSecurityContext = SecurityContextHolder.getContext();

        CompletableFuture.runAsync(() -> {
            SecurityContextHolder.setContext(mainThreadSecurityContext);

            Map<String, String> toolResults = new LinkedHashMap<>();
            boolean hasFailure = false;

            try {
                for (ToolCall tc : toolCalls) {
                    if (tc == null || tc.toolName() == null) {
                        continue;
                    }

                    String toolName = tc.toolName();
                    String parameters = tc.parameters() != null ? tc.parameters() : "{}";

                    try {
                        AgentTool tool = toolRegistry.getTool(toolName);
                        if (tool == null) {
                            String errorMsg = String.format("工具 '%s' 不存在", toolName);
                            sink.next("\n\n" + errorMsg + "\n");
                            toolResults.put(toolName, errorMsg);
                            hasFailure = true;
                            continue;
                        }

                        Map<String, Object> argsMap = parseParameters(parameters);

                        ToolExecutionRequest request = ToolExecutionRequest.builder()
                                .toolName(toolName)
                                .arguments(argsMap)
                                .originalCallId(tc.callId())
                                .build();

                        long execStart = System.currentTimeMillis();
                        ToolExecutionResult result = tool.execute(request, context);
                        long execDuration = System.currentTimeMillis() - execStart;

                        if (result.isSuccess()) {
                            toolResults.put(toolName, result.getOutput() != null ? result.getOutput() : "");
                            context.addToolCallRecord(ToolCallRecord.of(
                                    tc,
                                    ToolResult.success(toolName, result.getOutput(), execDuration)
                            ));
                        } else {
                            String errorMsg = String.format("工具执行失败: %s",
                                    result.getErrorMessage() != null ? result.getErrorMessage() : "未知错误");
                            sink.next("\n\n" + errorMsg + "\n");
                            toolResults.put(toolName, errorMsg);
                            hasFailure = true;
                            context.addToolCallRecord(ToolCallRecord.of(
                                    tc,
                                    ToolResult.failure(toolName, result.getErrorMessage(), execDuration, tc.callId())
                            ));
                        }

                    } catch (Exception e) {
                        log.error("[SinglePass] 工具执行异常: toolName={}", toolName, e);
                        String errorMsg = String.format("工具执行异常: %s", e.getMessage());
                        sink.next("\n\n" + errorMsg + "\n");
                        toolResults.put(toolName, errorMsg);
                        hasFailure = true;
                        context.addToolCallRecord(ToolCallRecord.of(
                                tc,
                                ToolResult.failure(toolName, e.getMessage(), 0, tc.callId())
                        ));
                    }
                }

                if (hasFailure) {
                    sink.next("\n\n【系统提示】工具执行失败，请稍后重试或联系管理员。\n");
                    sink.complete();
                    return;
                }

                continueAfterTools(originalMessages, toolResults, context, sink);

            } finally {
                SecurityContextHolder.clearContext();
            }

        }, agentWorkerExecutor).exceptionally(e -> {
            log.error("[SinglePass] 工具执行异步异常", e);
            sink.next("\n\n【系统错误】工具执行过程中发生异常\n");
            sink.complete();
            return null;
        });
    }

    /**
     * 工具执行完成后，构建消息并继续调用 LLM
     */
    private void continueAfterTools(
            List<Message> originalMessages,
            Map<String, String> toolResults,
            AgentContext context,
            reactor.core.publisher.FluxSink<String> sink) {

        log.info("[SinglePass] 工具执行完成，生成最终回答");

        List<Message> newMessages = new ArrayList<>(originalMessages);

        // 构建工具结果消息
        StringBuilder toolResultsMsg = new StringBuilder("【工具执行结果】\n");
        for (Map.Entry<String, String> entry : toolResults.entrySet()) {
            String result = entry.getValue();
            if (result.length() > 5000) {
                result = result.substring(0, 5000) + "\n\n...(结果过长，已截断)";
            }
            toolResultsMsg.append(String.format("**%s**: %s\n\n", entry.getKey(), result));
        }

        // 如果 GSC 启用，对工具结果做结构化+压缩
        if (gscService.isEnabled()) {
            String compressed = gscService.process(
                    null,  // 无新证据
                    null,  // 历史已在 originalMessages 中
                    toolResultsMsg.toString(),
                    null,  // 系统提示词已在 originalMessages 中
                    context.getQuery()
            );
            newMessages.add(new UserMessage(compressed));
        } else {
            newMessages.add(new UserMessage(toolResultsMsg.toString()));
        }

        final boolean[] newState = new boolean[]{false, false};
        final StringBuilder secondRoundAnswer = new StringBuilder();

        String secondRoundModel = context.getIntent() == IntentType.KB_QA ? finetunedModel : baseModel;
        llmService.deepThinkStream(secondRoundModel, newMessages, List.of(), new StreamCallback() {
            @Override
            public void onReasoning(String reasoning) {
                if (reasoning != null && !reasoning.isEmpty()) {
                    if (!newState[0] || newState[1]) {
                        sink.next("@@THINK_START@@");
                        newState[0] = true;
                        newState[1] = false;
                    }
                    sink.next(reasoning);
                }
            }

            @Override
            public void onContent(String content) {
                if (content != null && !content.isEmpty()) {
                    if (newState[0] && !newState[1]) {
                        sink.next("@@THINK_END@@");
                        newState[1] = true;
                        newState[0] = false;
                    }
                    sink.next(content);
                    secondRoundAnswer.append(content);
                }
            }

            @Override
            public void onToolCall(List<ToolCall> toolCalls) {
                log.warn("[SinglePass] 第二轮 LLM 不应再调用工具");
            }

            @Override
            public void onComplete() {
                if (newState[0] && !newState[1]) {
                    sink.next("@@THINK_END@@");
                }

                log.info("[SinglePass] 第二轮 LLM 生成完成，回答长度: {}", secondRoundAnswer.length());
                saveToMemory(context, "", secondRoundAnswer.toString());
                asyncJudge(context, secondRoundAnswer.toString());
                sink.complete();
            }

            @Override
            public void onError(Throwable e) {
                log.error("[SinglePass] 第二轮 LLM 调用失败", e);
                sink.error(e);
            }
        });
    }

    // ==================== LLM 调用 ====================

    /**
     * 流式 LLM 调用（双模型路由）
     * <p>
     * 根据意图分流：
     * <ul>
     *   <li>KB_QA → 微调模型 + 只给 knowledge_search</li>
     *   <li>CHAT → 基座模型 + 所有工具，允许按需调用 knowledge_search 兜底</li>
     * </ul>
     */
    private Flux<String> doStreamChat(List<Message> messages, AgentContext context) {
        String activeModel;
        List<AgentTool> tools;

        if (context.getIntent() == IntentType.KB_QA) {
            activeModel = finetunedModel;
            if (context.isPreRetrievalDone()) {
                tools = List.of();
                log.debug("[SinglePass] KB_QA 预检索已完成，不传工具给微调模型");
            } else {
                tools = toolRegistry.getTools(Set.of("knowledge_search"));
                log.debug("[SinglePass] KB_QA → 微调模型 + knowledge_search, 工具数: {}", tools.size());
            }
        } else {
            activeModel = baseModel;
            tools = toolRegistry.getTools(null);
            log.debug("[SinglePass] CHAT → 基座模型 + 所有工具（含 knowledge_search 兜底）, 工具数: {}", tools.size());
        }

        final String model = activeModel;

        return Flux.create(sink -> {
            final boolean[] state = new boolean[]{false, false};
            final List<ToolCall> pendingToolCalls = new ArrayList<>();
            final boolean[] toolCallTriggered = new boolean[]{false};

            llmService.deepThinkStream(model, messages, tools, new StreamCallback() {
                @Override
                public void onReasoning(String reasoning) {
                    if (reasoning != null && !reasoning.isEmpty()) {
                        if (!state[0] || state[1]) {
                            sink.next("@@THINK_START@@");
                            state[0] = true;
                            state[1] = false;
                        }
                        sink.next(reasoning);
                    }
                }

                @Override
                public void onContent(String content) {
                    if (content != null && !content.isEmpty()) {
                        if (state[0] && !state[1]) {
                            sink.next("@@THINK_END@@");
                            state[1] = true;
                            state[0] = false;
                        }
                        sink.next(content);
                    }
                }

                @Override
                public void onToolCall(List<ToolCall> toolCalls) {
                    if (toolCalls == null || toolCalls.isEmpty()) {
                        return;
                    }
                    pendingToolCalls.addAll(toolCalls);
                    toolCallTriggered[0] = true;

                    for (ToolCall tc : toolCalls) {
                        if (tc != null && tc.toolName() != null) {
                            log.info("[SinglePass] LLM 调用工具: toolName={}, parameters={}",
                                    tc.toolName(), tc.parameters());
                            sink.next(String.format("\n\n> 🛠️ 调用工具: `%s`\n", tc.toolName()));
                        }
                    }
                }

                @Override
                public void onComplete() {
                    if (state[0] && !state[1]) {
                        sink.next("@@THINK_END@@");
                    }

                    if (toolCallTriggered[0] && !pendingToolCalls.isEmpty()) {
                        executeToolsAndContinue(pendingToolCalls, messages, context, sink);
                    } else {
                        sink.complete();
                    }
                }

                @Override
                public void onError(Throwable e) {
                    log.error("[SinglePass] LLM流式调用失败: sessionId={}", context.getSessionId(), e);
                    sink.error(e);
                }
            });
        });
    }

    // ==================== Judge 审计 ====================

    private void appendChunkForPersistence(
            String chunk,
            StringBuilder answer,
            StringBuilder thinking,
            boolean[] collectingThinking) {
        if (chunk == null || chunk.isEmpty()) {
            return;
        }

        if (chunk.startsWith("__TOOL_CALL__:")) {
            thinking.append("\n[调用工具: ")
                    .append(chunk.substring("__TOOL_CALL__:".length()).trim())
                    .append("]\n");
            return;
        }

        String remaining = chunk;
        while (!remaining.isEmpty()) {
            int start = remaining.indexOf("@@THINK_START@@");
            int end = remaining.indexOf("@@THINK_END@@");

            if (start < 0 && end < 0) {
                appendToCurrentBuffer(remaining, answer, thinking, collectingThinking[0]);
                return;
            }

            boolean nextIsStart = start >= 0 && (end < 0 || start < end);
            int markerIndex = nextIsStart ? start : end;
            String beforeMarker = remaining.substring(0, markerIndex);
            appendToCurrentBuffer(beforeMarker, answer, thinking, collectingThinking[0]);

            String marker = nextIsStart ? "@@THINK_START@@" : "@@THINK_END@@";
            collectingThinking[0] = nextIsStart;
            remaining = remaining.substring(markerIndex + marker.length());
        }
    }

    private void appendToCurrentBuffer(
            String text,
            StringBuilder answer,
            StringBuilder thinking,
            boolean collectingThinking) {
        if (text == null || text.isEmpty()) {
            return;
        }
        if (collectingThinking) {
            thinking.append(text);
        } else {
            answer.append(text);
        }
    }

    private void asyncJudge(AgentContext context, String answer) {
        CompletableFuture.runAsync(() -> {
            try {
                long startTime = System.currentTimeMillis();
                log.info("[Judge] ========== 开始审计 ==========");

                Map<String, Object> ragContext = buildRagContextForJudge(context);

                log.info("[Judge] 审计上下文构建完成: toolCalls={}, hasPreRetrieved={}, hasHistory={}",
                        context.hasToolCallRecords() ? context.getToolCallRecords().size() : 0,
                        ragContext.containsKey("pre_retrieved_content"),
                        ragContext.containsKey("conversation_history"));

                EvaluationResult evalResult = reflectionService.evaluate(
                        context.getQuery(), answer, ragContext
                );

                log.info("[Judge] 审计结果: sessionId={}, passed={}, reason='{}'",
                        context.getSessionId(), evalResult.isPassed(), evalResult.getReason());

                freezeCaseSnapshot(context, answer, evalResult, startTime);

                log.info("[Judge] ========== 审计完成 ==========");

            } catch (Exception e) {
                log.error("[Judge] 审计失败: sessionId={}", context.getSessionId(), e);
                try {
                    freezeFailedCaseSnapshot(context, answer, e);
                } catch (Exception ex) {
                    log.error("[Judge] 保存失败 CaseSnapshot 也失败了", ex);
                }
            }
        }, judgeExecutor);
    }

    private Map<String, Object> buildRagContextForJudge(AgentContext context) {
        Map<String, Object> ragContext = new HashMap<>();

        if (context.getPreRetrievedContent() != null) {
            ragContext.put("pre_retrieved_content", context.getPreRetrievedContent());
        }

        if (context.hasToolCallRecords()) {
            List<Map<String, Object>> toolCallsList = new ArrayList<>();
            for (ToolCallRecord record : context.getToolCallRecords()) {
                Map<String, Object> tcMap = new HashMap<>();
                if (record.toolCall() != null) {
                    tcMap.put("tool_name", record.toolCall().toolName());
                    tcMap.put("parameters", record.toolCall().parameters());
                }
                if (record.toolResult() != null) {
                    tcMap.put("success", record.toolResult().success());
                    tcMap.put("result", record.toolResult().result());
                    tcMap.put("error", record.toolResult().errorMessage());
                    tcMap.put("duration_ms", record.toolResult().durationMs());
                }
                toolCallsList.add(tcMap);
            }
            ragContext.put("tool_calls", toolCallsList);
        }

        if (context.getIntent() != null) {
            ragContext.put("intent", context.getIntent().name());
        }
        if (context.getIntentConfidence() != null) {
            ragContext.put("intent_confidence", context.getIntentConfidence());
        }

        try {
            List<Message> history = chatMemoryRepository.findByConversationId(context.getSessionId());
            if (history != null && !history.isEmpty()) {
                int maxMessages = Math.min(6, history.size());
                List<Message> recentHistory = history.subList(
                        Math.max(0, history.size() - maxMessages), history.size()
                );
                StringBuilder historySummary = new StringBuilder();
                for (Message msg : recentHistory) {
                    String role = msg instanceof UserMessage ? "用户" : "助手";
                    String content = msg.getText();
                    if (content != null && content.length() > 200) {
                        content = content.substring(0, 200) + "...";
                    }
                    historySummary.append(String.format("[%s]: %s\n", role, content));
                }
                ragContext.put("conversation_history", historySummary.toString());
            }
        } catch (Exception e) {
            log.warn("[SinglePass] 获取历史记录失败: {}", e.getMessage());
        }

        return ragContext;
    }

    private void freezeCaseSnapshot(AgentContext context, String answer,
                                     EvaluationResult evalResult, long startTime) {
        try {
            log.info("[CaseSnapshot] 开始冻结: sessionId={}, passed={}",
                    context.getSessionId(), evalResult.isPassed());

            var snapshot = CaseSnapshotBuilder.create()
                    .scenario(Scenario.CHAT)
                    .intent(context.getIntent() != null ? context.getIntent().name() : "UNKNOWN")
                    .input(context.getQuery(),
                            context.getUserId() != null ? Long.parseLong(context.getUserId()) : null,
                            context.getSessionId())
                    .outputData(answer, System.currentTimeMillis() - startTime, null)
                    .status(CaseStatus.COMPLETED)
                    .durationMs((int) (System.currentTimeMillis() - startTime))
                    .toolCallRecords(context.getToolCallRecords())
                    .aiJudgeResult(evalResult.isPassed(), evalResult.getReason(), "SYSTEM-JUDGE-v1.0")
                    .promptData("SYSTEM-JUDGE-v1.0")
                    .metadata("intent_confidence",
                            context.getIntentConfidence() != null ?
                                    String.valueOf(context.getIntentConfidence()) : "N/A")
                    .build();

            caseSnapshotService.freezeAsync(snapshot);
            log.info("[CaseSnapshot] 已提交冻结: sessionId={}", context.getSessionId());

        } catch (Exception e) {
            log.error("[CaseSnapshot] 构建失败", e);
        }
    }

    private void freezeFailedCaseSnapshot(AgentContext context, String answer, Exception error) {
        var snapshot = CaseSnapshotBuilder.create()
                .scenario(Scenario.CHAT)
                .intent(context.getIntent() != null ? context.getIntent().name() : "UNKNOWN")
                .input(context.getQuery(),
                        context.getUserId() != null ? Long.parseLong(context.getUserId()) : null,
                        context.getSessionId())
                .outputData(answer, null, null)
                .status(CaseStatus.FAILED)
                .errorMessage("Judge审计失败: " + error.getMessage())
                .toolCallRecords(context.getToolCallRecords())
                .build();

        caseSnapshotService.freezeAsync(snapshot);
    }

    // ==================== 辅助方法 ====================

    private String formatKnowledgeResult(com.agenthub.api.ai.domain.knowledge.PowerKnowledgeResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("【知识库检索结果】");

        List<com.agenthub.api.ai.domain.knowledge.EvidenceBlock> blocks = result.evidenceBlocks();
        if (blocks != null && !blocks.isEmpty()) {
            sb.append("共找到 ").append(blocks.size())
                    .append(" 个证据块（").append(result.rawContentSnippets().size())
                    .append(" 个文档片段）\n\n");

            for (int i = 0; i < blocks.size(); i++) {
                com.agenthub.api.ai.domain.knowledge.EvidenceBlock block = blocks.get(i);
                sb.append(String.format("[证据 %d] %s | 支持度: %.2f\n",
                        i + 1, block.getSourceReference(), block.supportScore()));
                sb.append(block.content()).append("\n\n");
            }
        } else {
            sb.append("共找到 ").append(result.rawContentSnippets().size())
                    .append(" 条相关内容\n\n");
            for (String snippet : result.rawContentSnippets()) {
                sb.append(snippet).append("\n\n---\n\n");
            }
        }

        if (result.sources() != null && !result.sources().isEmpty()) {
            sb.append("【来源文件】\n");
            for (var source : result.sources()) {
                sb.append(String.format("- %s\n", source.filename()));
            }
        }

        return sb.toString();
    }

    private String buildSystemPrompt(AgentContext context) {
        try {
            String promptCode;
            if (context.getIntent() == IntentType.KB_QA) {
                promptCode = SYSTEM_PROMPT_KBQA;
            } else {
                promptCode = SYSTEM_PROMPT_CHAT;
            }

            String toolsDesc = "";
            Map<String, Object> vars = new HashMap<>();
            vars.put("tools_desc", toolsDesc);

            String systemPromptText = sysPromptService.render(promptCode, vars);

            if (systemPromptText != null && !systemPromptText.isEmpty()) {
                return systemPromptText;
            }
            return "你是一个电力行业智能助手。";

        } catch (Exception e) {
            log.warn("[SinglePass] 构建系统提示词失败", e);
            return "你是一个智能助手。";
        }
    }

    private Map<String, Object> parseParameters(String parameters) {
        try {
            if (parameters == null || parameters.isBlank()) {
                return new HashMap<>();
            }
            return objectMapper.readValue(parameters,
                    new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            log.warn("[SinglePass] 解析工具参数失败: {}, error={}", parameters, e.getMessage());
            return new HashMap<>();
        }
    }

}
