package com.agenthub.api.prompt.builder;

import com.agenthub.api.agent_engine.model.ToolCallRecord;
import com.agenthub.api.prompt.context.PromptContext;
import com.agenthub.api.prompt.context.PromptContextHolder;
import com.agenthub.api.prompt.domain.entity.CaseSnapshot;
import com.agenthub.api.prompt.domain.entity.SysPrompt;
import com.agenthub.api.prompt.enums.CaseStatus;
import com.agenthub.api.prompt.enums.Scenario;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Case 快照构建器
 * <p>简化 CaseSnapshot 的创建过程</p>
 *
 * @author AgentHub
 * @since 2026-01-27
 */
@Slf4j
public class CaseSnapshotBuilder {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    private final CaseSnapshot snapshot;

    private CaseSnapshotBuilder() {
        this.snapshot = new CaseSnapshot();
    }

    public static CaseSnapshotBuilder create() {
        return new CaseSnapshotBuilder();
    }

    public CaseSnapshotBuilder scenario(String scenario) {
        try {
            this.snapshot.setScenario(Scenario.valueOf(scenario.toUpperCase()));
        } catch (IllegalArgumentException e) {
            this.snapshot.setScenario(Scenario.OTHER);
        }
        return this;
    }

    public CaseSnapshotBuilder scenario(Scenario scenario) {
        this.snapshot.setScenario(scenario);
        return this;
    }

    public CaseSnapshotBuilder intent(String intent) {
        this.snapshot.setIntent(intent);
        return this;
    }

    public CaseSnapshotBuilder input(String query, Long userId, String sessionId) {
        ObjectNode inputNode = objectMapper.createObjectNode();
        inputNode.put("query", query);
        if (userId != null) {
            inputNode.put("user_id", userId);
        }
        if (sessionId != null) {
            inputNode.put("session_id", sessionId);
        }
        this.snapshot.setInputData(inputNode);
        return this;
    }

    public CaseSnapshotBuilder inputData(JsonNode inputData) {
        this.snapshot.setInputData(inputData);
        return this;
    }

    public CaseSnapshotBuilder contextData(JsonNode contextData) {
        this.snapshot.setContextData(contextData);
        return this;
    }

    public CaseSnapshotBuilder ragContext(String query, JsonNode documents, Integer totalResults) {
        ObjectNode contextNode = objectMapper.createObjectNode();
        contextNode.put("rag_query", query);
        if (documents != null) {
            contextNode.set("rag_documents", documents);
        }
        if (totalResults != null) {
            contextNode.put("rag_total_results", totalResults);
        }
        this.snapshot.setContextData(contextNode);
        return this;
    }

    public CaseSnapshotBuilder outputData(String rawResponse, Long responseTimeMs, JsonNode tokenUsage) {
        ObjectNode outputNode = objectMapper.createObjectNode();
        outputNode.put("raw_response", rawResponse);
        if (responseTimeMs != null) {
            outputNode.put("response_time_ms", responseTimeMs);
        }
        if (tokenUsage != null) {
            outputNode.set("token_usage", tokenUsage);
        }
        outputNode.put("finish_reason", "stop");
        this.snapshot.setOutputData(outputNode);
        return this;
    }

    public CaseSnapshotBuilder outputData(JsonNode outputData) {
        this.snapshot.setOutputData(outputData);
        return this;
    }

    public CaseSnapshotBuilder modelData(String provider, String modelName, Map<String, Object> parameters) {
        ObjectNode modelNode = objectMapper.createObjectNode();
        modelNode.put("provider", provider);
        modelNode.put("name", modelName);
        if (parameters != null && !parameters.isEmpty()) {
            modelNode.set("parameters", objectMapper.valueToTree(parameters));
        }
        this.snapshot.setModelData(modelNode);
        return this;
    }

    public CaseSnapshotBuilder modelData(JsonNode modelData) {
        this.snapshot.setModelData(modelData);
        return this;
    }

    public CaseSnapshotBuilder status(CaseStatus status) {
        this.snapshot.setStatus(status);
        return this;
    }

    public CaseSnapshotBuilder status(String status) {
        try {
            this.snapshot.setStatus(CaseStatus.valueOf(status.toUpperCase()));
        } catch (IllegalArgumentException e) {
            this.snapshot.setStatus(CaseStatus.COMPLETED);
        }
        return this;
    }

    public CaseSnapshotBuilder errorMessage(String errorMessage) {
        this.snapshot.setErrorMessage(errorMessage);
        this.snapshot.setStatus(CaseStatus.FAILED);
        return this;
    }

    public CaseSnapshotBuilder durationMs(Integer durationMs) {
        this.snapshot.setDurationMs(durationMs);
        return this;
    }

    public CaseSnapshotBuilder requestTime(LocalDateTime requestTime) {
        this.snapshot.setRequestTime(requestTime);
        return this;
    }

    public CaseSnapshotBuilder responseTime(LocalDateTime responseTime) {
        this.snapshot.setResponseTime(responseTime);
        return this;
    }

    /**
     * 设置提示词数据（必须在主线程中调用，因为依赖 ThreadLocal）
     */
    public CaseSnapshotBuilder capturePromptData() {
        try {
            PromptContext context = PromptContextHolder.getContext();
            if (context != null) {
                // 复用 enrichPromptData 的逻辑，但在这里直接调用
                this.snapshot.setPromptData(collectPromptData(context));
            }
        } catch (Exception e) {
            log.warn("[CaseSnapshotBuilder] 捕获提示词数据失败: {}", e.getMessage());
        }
        return this;
    }

    /**
     * 从 PromptContext 收集提示词数据
     */
    private JsonNode collectPromptData(PromptContext context) {
        ObjectNode promptNode = objectMapper.createObjectNode();

        promptNode.put("timestamp", java.time.LocalDateTime.now().toString());
        promptNode.put("frozen_at", java.time.LocalDateTime.now()
                .format(java.time.format.DateTimeFormatter.ISO_LOCAL_DATE_TIME));

        // 系统 Prompts
        if (context.getSystemPrompts() != null && !context.getSystemPrompts().isEmpty()) {
            var systemArray = promptNode.putArray("system_prompts");
            for (SysPrompt prompt : context.getSystemPrompts()) {
                systemArray.add(buildPromptSummary(prompt));
            }
        }

        // Router Prompts
        if (context.getRouterPrompts() != null && !context.getRouterPrompts().isEmpty()) {
            var routerArray = promptNode.putArray("router_prompts");
            for (com.agenthub.api.prompt.domain.entity.SysPrompt prompt : context.getRouterPrompts()) {
                routerArray.add(buildPromptSummary(prompt));
            }
        }

        // Skill Prompts
        if (context.getSkillPrompts() != null && !context.getSkillPrompts().isEmpty()) {
            var skillArray = promptNode.putArray("skill_prompts");
            for (com.agenthub.api.prompt.domain.entity.SysPrompt prompt : context.getSkillPrompts()) {
                skillArray.add(buildPromptSummary(prompt));
            }
        }

        // Worker Prompts
        if (context.getWorkerPrompts() != null && !context.getWorkerPrompts().isEmpty()) {
            var workerArray = promptNode.putArray("worker_prompts");
            for (com.agenthub.api.prompt.domain.entity.SysPrompt prompt : context.getWorkerPrompts()) {
                workerArray.add(buildPromptSummary(prompt));
            }
        }

        // Tool Prompts
        if (context.getToolPrompts() != null && !context.getToolPrompts().isEmpty()) {
            var toolArray = promptNode.putArray("tool_prompts");
            for (com.agenthub.api.prompt.domain.entity.SysPrompt prompt : context.getToolPrompts()) {
                toolArray.add(buildPromptSummary(prompt));
            }
        }

        return promptNode;
    }

    /**
     * 构建提示词摘要
     */
    private com.fasterxml.jackson.databind.node.ObjectNode buildPromptSummary(
            com.agenthub.api.prompt.domain.entity.SysPrompt prompt) {
        com.fasterxml.jackson.databind.node.ObjectNode node = objectMapper.createObjectNode();
        node.put("code", prompt.getPromptCode());
        node.put("name", prompt.getPromptName());
        node.put("type", prompt.getPromptType().name());
        node.put("version", prompt.getVersion());
        node.put("template_type", prompt.getTemplateType().name());
        if (prompt.getContent() != null) {
            node.put("content_length", prompt.getContent().toString().length());
        }
        return node;
    }

    public CaseSnapshotBuilder metadata(JsonNode metadata) {
        this.snapshot.setMetadata(metadata);
        return this;
    }

    public CaseSnapshotBuilder metadata(String key, String value) {
        try {
            ObjectNode metaNode;
            if (this.snapshot.getMetadata() != null && this.snapshot.getMetadata().isObject()) {
                metaNode = (ObjectNode) this.snapshot.getMetadata();
            } else {
                metaNode = objectMapper.createObjectNode();
            }
            metaNode.put(key, value);
            this.snapshot.setMetadata(metaNode);
        } catch (Exception e) {
            log.warn("[CaseSnapshotBuilder] 设置元数据失败: {}", e.getMessage());
        }
        return this;
    }

    /**
     * 设置工具调用记录
     * <p>将工具调用记录序列化到 contextData 中</p>
     *
     * @param toolCallRecords 工具调用记录列表
     * @return this
     */
    public CaseSnapshotBuilder toolCallRecords(List<ToolCallRecord> toolCallRecords) {
        // 总是初始化 contextData（修复 NOT NULL 约束问题）
        ObjectNode contextNode;
        if (this.snapshot.getContextData() != null && this.snapshot.getContextData().isObject()) {
            contextNode = (ObjectNode) this.snapshot.getContextData();
        } else {
            contextNode = objectMapper.createObjectNode();
            this.snapshot.setContextData(contextNode);
        }

        // 只有在非空时才添加记录
        if (toolCallRecords != null && !toolCallRecords.isEmpty()) {
            try {
                var toolCallsArray = contextNode.putArray("tool_call_records");
                for (ToolCallRecord record : toolCallRecords) {
                    ObjectNode recordNode = objectMapper.createObjectNode();
                    if (record.toolCall() != null) {
                        recordNode.put("tool_name", record.toolCall().toolName());
                        recordNode.put("parameters", record.toolCall().parameters());
                        recordNode.put("call_id", record.toolCall().callId());
                    }
                    if (record.toolResult() != null) {
                        recordNode.put("success", record.toolResult().success());
                        recordNode.put("result", record.toolResult().result());
                        recordNode.put("error_message", record.toolResult().errorMessage());
                        recordNode.put("duration_ms", record.toolResult().durationMs());
                    }
                    toolCallsArray.add(recordNode);
                }
            } catch (Exception e) {
                log.warn("[CaseSnapshotBuilder] 设置工具调用记录失败: {}", e.getMessage());
            }
        }
        return this;
    }

    /**
     * 设置提示词数据（用于 prompt_data 字段）
     *
     * @param promptCode 提示词代码
     * @return this
     */
    public CaseSnapshotBuilder promptData(String promptCode) {
        try {
            ObjectNode promptDataNode = objectMapper.createObjectNode();
            promptDataNode.put("template_name", promptCode);
            promptDataNode.put("template_version", "v1.0");
            promptDataNode.put("system_prompt", "");
            promptDataNode.put("rendered_prompt", "");
            promptDataNode.put("variables", objectMapper.createObjectNode());
            this.snapshot.setPromptData(promptDataNode);
        } catch (Exception e) {
            log.warn("[CaseSnapshotBuilder] 设置 prompt_data 失败: {}", e.getMessage());
        }
        return this;
    }

    /**
     * 设置 AI Judge 结果
     *
     * @param passed    是否通过
     * @param reason    原因（失败时）
     * @param rawOutput Judge 模型原始输出
     * @return this
     */
    public CaseSnapshotBuilder aiJudgeResult(boolean passed, String reason, String rawOutput) {
        try {
            ObjectNode judgeNode = objectMapper.createObjectNode();
            judgeNode.put("passed", passed);
            judgeNode.put("reason", reason);
            judgeNode.put("raw_output", rawOutput);
            judgeNode.put("judged_at", LocalDateTime.now().toString());
            this.snapshot.setAiJudgeResult(judgeNode);
            this.snapshot.setIsEvaluated(true);
        } catch (Exception e) {
            log.warn("[CaseSnapshotBuilder] 设置 AI Judge 结果失败: {}", e.getMessage());
        }
        return this;
    }

    /**
     * 设置 AI Judge 结果（直接设置 JsonNode）
     *
     * @param aiJudgeResult Judge 结果 JsonNode
     * @return this
     */
    public CaseSnapshotBuilder aiJudgeResult(JsonNode aiJudgeResult) {
        this.snapshot.setAiJudgeResult(aiJudgeResult);
        return this;
    }

    public CaseSnapshot build() {
        // 设置默认时间
        if (this.snapshot.getRequestTime() == null) {
            this.snapshot.setRequestTime(LocalDateTime.now());
        }

        // 设置默认 model_data（修复 NOT NULL 约束问题）
        if (this.snapshot.getModelData() == null) {
            ObjectNode defaultModelData = objectMapper.createObjectNode();
            defaultModelData.put("provider", "ollama");
            defaultModelData.put("name", "unknown");
            this.snapshot.setModelData(defaultModelData);
        }

        return this.snapshot;
    }
}
