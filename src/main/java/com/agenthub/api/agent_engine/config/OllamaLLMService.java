package com.agenthub.api.agent_engine.config;

import com.agenthub.api.agent_engine.model.AgentToolDefinition;
import com.agenthub.api.agent_engine.model.ToolCall;
import com.agenthub.api.agent_engine.tool.AgentTool;
import com.agenthub.api.ai.domain.llm.DeepThinkResult;
import com.agenthub.api.ai.domain.llm.StreamCallback;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Ollama 本地模型服务
 * <p>实现 {@link LLMService} 接口，当 {@code app.llm.provider=ollama} 时激活</p>
 *
 * <h3>工具调用策略：</h3>
 * <ul>
 *   <li>无工具：使用 Spring AI {@link OllamaChatModel}（自动配置，简单高效）</li>
 *   <li>有工具：直接调用 Ollama REST API（绕过 Spring AI 自动执行，手动控制工具调用流程）</li>
 * </ul>
 *
 * <h3>思考内容提取：</h3>
 * <ul>
 *   <li>Spring AI 路径：从 ChatResponse.metadata["thinking"] 提取</li>
 *   <li>REST API 路径：从 message.thinking 字段提取</li>
 * </ul>
 *
 * @author AgentHub
 * @since 2026-04-25
 */
@Slf4j
@Service
@ConditionalOnProperty(name = "app.llm.provider", havingValue = "ollama")
public class OllamaLLMService implements LLMService {

    private final OllamaChatModel ollamaChatModel;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    @Value("${spring.ai.ollama.base-url:http://localhost:11434}")
    private String ollamaBaseUrl;

    public OllamaLLMService(OllamaChatModel ollamaChatModel, ObjectMapper objectMapper) {
        this.ollamaChatModel = ollamaChatModel;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newHttpClient();
        log.info("[Ollama] OllamaLLMService 已激活，baseUrl={}", ollamaBaseUrl);
    }

    // ========== LLMService 接口实现 ==========

    @Override
    public DeepThinkResult deepThink(String model, String prompt, String system) {
        List<Message> messages = new ArrayList<>();
        if (system != null) {
            messages.add(new SystemMessage(system));
        } else {
            messages.add(new SystemMessage("你是一个有用的助手。"));
        }
        messages.add(new UserMessage(prompt));
        return deepThink(model, messages);
    }

    @Override
    public DeepThinkResult deepThink(String model, List<Message> messages) {
        OllamaOptions options = OllamaOptions.builder()
                .model(model)
                .build();

        Prompt prompt = new Prompt(messages, options);

        try {
            ChatResponse response = ollamaChatModel.call(prompt);

            String rawContent = response.getResult() != null
                    ? response.getResult().getOutput().getText()
                    : "";

            // 从 metadata 提取思考内容
            String thinking = extractThinkingFromMetadata(response);

            return DeepThinkResult.builder()
                    .reasoningContent(thinking != null ? thinking : "")
                    .content(rawContent != null ? rawContent : "")
                    .model(model)
                    .build();
        } catch (Exception e) {
            log.error("[Ollama] 调用失败: {}", e.getMessage(), e);
            throw new RuntimeException("AI 调用失败: " + e.getMessage(), e);
        }
    }

    @Override
    public void deepThinkStream(String model, List<Message> messages, StreamCallback callback) {
        deepThinkStream(model, messages, null, callback);
    }

    @Override
    public void deepThinkStream(String model, List<Message> messages,
                                List<AgentTool> agentTools, StreamCallback callback) {
        if (agentTools != null && !agentTools.isEmpty()) {
            // 有工具 → 直接调用 Ollama REST API（手动控制工具调用流程）
            streamWithTools(model, messages, agentTools, callback);
        } else {
            // 无工具 → 使用 Spring AI OllamaChatModel
            streamWithoutTools(model, messages, callback);
        }
    }

    // ========== 无工具：Spring AI OllamaChatModel ==========

    private void streamWithoutTools(String model, List<Message> messages, StreamCallback callback) {
        OllamaOptions options = OllamaOptions.builder()
                .model(model)
                .temperature(0.7)
                .build();

        Prompt prompt = new Prompt(messages, options);

        try {
            ollamaChatModel.stream(prompt)
                    .subscribe(
                            response -> handleSpringAiChunk(response, callback),
                            error -> {
                                log.error("[Ollama] 流式调用失败: {}", error.getMessage(), error);
                                callback.onError(error);
                            },
                            callback::onComplete
                    );
        } catch (Exception e) {
            log.error("[Ollama] 流式调用失败: {}", e.getMessage(), e);
            callback.onError(e);
            throw new RuntimeException("AI 流式调用失败: " + e.getMessage(), e);
        }
    }

    private void handleSpringAiChunk(ChatResponse response, StreamCallback callback) {
        if (response == null || response.getResult() == null) {
            return;
        }

        // 思考内容（metadata）
        String thinking = extractThinkingFromMetadata(response);
        if (thinking != null && !thinking.isEmpty()) {
            callback.onReasoning(thinking);
        }

        // 正文内容
        String content = response.getResult().getOutput() != null
                ? response.getResult().getOutput().getText()
                : null;
        if (content != null && !content.isEmpty()) {
            callback.onContent(content);
        }
    }

    // ========== 有工具：Ollama REST API 直接调用 ==========

    /**
     * 直接调用 Ollama REST API（带工具）
     * <p>绕过 Spring AI 的自动工具执行，手动控制工具调用流程，
     * 以便通过 {@link StreamCallback#onToolCall} 回调给 SinglePassExecutor 处理</p>
     */
    private void streamWithTools(String model, List<Message> messages,
                                  List<AgentTool> agentTools, StreamCallback callback) {
        try {
            // 1. 构建 Ollama /api/chat 请求体
            ObjectNode request = buildOllamaRequest(model, messages, agentTools);

            log.debug("[Ollama] 工具调用请求: model={}, tools={}, messages={}",
                    model, agentTools.size(), messages.size());

            // 2. 发送 HTTP 请求（SSE 流式）
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(ollamaBaseUrl + "/api/chat"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(
                            objectMapper.writeValueAsString(request)))
                    .build();

            HttpResponse<Stream<String>> httpResponse = httpClient.send(
                    httpRequest, HttpResponse.BodyHandlers.ofLines());

            if (httpResponse.statusCode() != 200) {
                throw new RuntimeException("Ollama API 返回错误: " + httpResponse.statusCode());
            }

            // 3. 解析 SSE 流式响应
            parseSSEStream(httpResponse.body(), callback);

        } catch (Exception e) {
            log.error("[Ollama] 工具流式调用失败: {}", e.getMessage(), e);
            callback.onError(e);
        }
    }

    /**
     * 构建 Ollama /api/chat 请求体
     */
    private ObjectNode buildOllamaRequest(String model, List<Message> messages,
                                            List<AgentTool> agentTools) throws Exception {
        ObjectNode request = objectMapper.createObjectNode();
        request.put("model", model);
        request.put("stream", true);

        // messages 数组
        ArrayNode messagesArray = request.putArray("messages");
        for (Message msg : messages) {
            ObjectNode msgNode = messagesArray.addObject();
            msgNode.put("role", toOllamaRole(msg));
            msgNode.put("content", msg.getText());
        }

        // tools 数组
        ArrayNode toolsArray = request.putArray("tools");
        for (AgentTool tool : agentTools) {
            AgentToolDefinition def = tool.getDefinition();
            ObjectNode toolNode = toolsArray.addObject();
            toolNode.put("type", "function");
            ObjectNode functionNode = toolNode.putObject("function");
            functionNode.put("name", def.getName());
            functionNode.put("description", def.getDescription());

            // 参数 schema
            JsonNode params = objectMapper.readTree(def.getParameterSchema());
            functionNode.set("parameters", params);
        }

        return request;
    }

    /**
     * 解析 Ollama SSE 流式响应
     * <p>每行是一个 JSON 对象，格式：
     * <pre>
     * {"message":{"role":"assistant","thinking":"...","content":"...","tool_calls":[...]}, "done":false}
     * {"done":true, "total_duration":...}
     * </pre>
     */
    private void parseSSEStream(Stream<String> lines, StreamCallback callback) {
        // 工具调用累积器（Ollama 可能分多个 chunk 返回工具调用的参数）
        final StringBuilder[] toolCallAccumulator = new StringBuilder[1];
        final String[] currentToolName = new String[1];
        final boolean[] inlineThinking = new boolean[]{false};
        final StringBuilder inlineContentBuffer = new StringBuilder();

        lines.forEach(line -> {
            if (line == null || line.isBlank()) {
                return;
            }

            try {
                JsonNode chunk = objectMapper.readTree(line);
                JsonNode message = chunk.get("message");

                if (message == null) {
                    // 可能是最终的 done 消息
                    if (chunk.has("done") && chunk.get("done").asBoolean()) {
                        flushInlineContent(callback, inlineThinking, inlineContentBuffer);
                        callback.onComplete();
                    }
                    return;
                }

                // 1. 思考内容（message.thinking 字段）
                if (message.has("thinking") && !message.get("thinking").isNull()
                        && !message.get("thinking").asText().isEmpty()) {
                    flushInlineContent(callback, inlineThinking, inlineContentBuffer);
                    callback.onReasoning(message.get("thinking").asText());
                }

                // 2. 正文内容（message.content 字段）
                if (message.has("content") && !message.get("content").isNull()
                        && !message.get("content").asText().isEmpty()) {
                    routeContentWithThinkTags(message.get("content").asText(), callback, inlineThinking, inlineContentBuffer);
                }

                // 3. 工具调用（message.tool_calls 数组）
                if (message.has("tool_calls") && message.get("tool_calls").isArray()) {
                    flushInlineContent(callback, inlineThinking, inlineContentBuffer);
                    for (JsonNode tc : message.get("tool_calls")) {
                        JsonNode function = tc.get("function");
                        if (function == null) {
                            continue;
                        }

                        String name = function.has("name") ? function.get("name").asText() : null;
                        JsonNode argsNode = function.has("arguments") ? function.get("arguments") : null;

                        if (name != null) {
                            currentToolName[0] = name;
                        }

                        // 参数可能是增量传输的，需要累积
                        if (argsNode != null && !argsNode.isNull()) {
                            String argsStr = objectMapper.writeValueAsString(argsNode);
                            if (toolCallAccumulator[0] == null) {
                                toolCallAccumulator[0] = new StringBuilder(argsStr);
                            } else {
                                toolCallAccumulator[0].append(argsStr);
                            }
                        }

                        // 判断工具调用是否完成（看 finish_reason 或 done 标志）
                        // Ollama 通常一次性返回完整工具调用参数
                        if (toolCallAccumulator[0] != null && currentToolName[0] != null) {
                            String finalArgs = toolCallAccumulator[0].toString();

                            // 尝试提取有效的 JSON 对象
                            String cleanArgs = extractJsonObject(finalArgs);

                            log.info("[Ollama] 工具调用完成: tool={}, args={}",
                                    currentToolName[0],
                                    cleanArgs.length() > 100 ? cleanArgs.substring(0, 100) + "..." : cleanArgs);

                            List<ToolCall> toolCalls = new ArrayList<>();
                            toolCalls.add(ToolCall.builder()
                                    .toolName(currentToolName[0])
                                    .parameters(cleanArgs)
                                    .callId("")
                                    .build());
                            callback.onToolCall(toolCalls);

                            // 重置累积器
                            toolCallAccumulator[0] = null;
                            currentToolName[0] = null;
                        }
                    }
                }

                // 4. 检查流是否结束
                if (chunk.has("done") && chunk.get("done").asBoolean()) {
                    // done 消息已在最外层处理
                }

            } catch (Exception e) {
                log.warn("[Ollama] 解析 SSE chunk 失败: {}", e.getMessage());
            }
        });

        // 流结束后确保 callback.onComplete 被调用
        // （如果 done 消息已经触发了 onComplete，这里不会重复，因为 onComplete 幂等）
        try {
            flushInlineContent(callback, inlineThinking, inlineContentBuffer);
            callback.onComplete();
        } catch (Exception ignored) {
            // 已完成则忽略
        }
    }

    // ========== 辅助方法 ==========

    /**
     * Route inline <think> tags from message.content into the reasoning callback.
     */
    private void routeContentWithThinkTags(
            String content,
            StreamCallback callback,
            boolean[] inlineThinking,
            StringBuilder buffer) {
        if (content == null || content.isEmpty()) {
            return;
        }

        buffer.append(content);
        while (buffer.length() > 0) {
            String text = buffer.toString();
            if (inlineThinking[0]) {
                int end = text.indexOf("</think>");
                if (end >= 0) {
                    String reasoning = text.substring(0, end);
                    if (!reasoning.isEmpty()) {
                        callback.onReasoning(reasoning);
                    }
                    buffer.delete(0, end + "</think>".length());
                    inlineThinking[0] = false;
                    continue;
                }

                int emitLength = Math.max(0, text.length() - ("</think>".length() - 1));
                if (emitLength > 0) {
                    callback.onReasoning(text.substring(0, emitLength));
                    buffer.delete(0, emitLength);
                }
                return;
            }

            int start = text.indexOf("<think>");
            if (start >= 0) {
                String normalContent = text.substring(0, start);
                if (!normalContent.isEmpty()) {
                    callback.onContent(normalContent);
                }
                buffer.delete(0, start + "<think>".length());
                inlineThinking[0] = true;
                continue;
            }

            int emitLength = Math.max(0, text.length() - ("<think>".length() - 1));
            if (emitLength > 0) {
                callback.onContent(text.substring(0, emitLength));
                buffer.delete(0, emitLength);
            }
            return;
        }
    }

    private void flushInlineContent(
            StreamCallback callback,
            boolean[] inlineThinking,
            StringBuilder buffer) {
        if (buffer.length() == 0) {
            return;
        }

        String remaining = buffer.toString();
        buffer.setLength(0);
        if (inlineThinking[0]) {
            callback.onReasoning(remaining);
            inlineThinking[0] = false;
        } else {
            callback.onContent(remaining);
        }
    }

    /**
     * Spring AI Message → Ollama role 字符串
     */
    private String toOllamaRole(Message msg) {
        if (msg instanceof SystemMessage) {
            return "system";
        } else if (msg instanceof UserMessage) {
            return "user";
        } else if (msg instanceof AssistantMessage) {
            return "assistant";
        }
        return "user";
    }

    /**
     * 从字符串中提取第一个完整的 JSON 对象
     */
    private String extractJsonObject(String input) {
        if (input == null || input.isEmpty()) {
            return "{}";
        }

        // 如果已经是完整 JSON，直接返回
        String trimmed = input.trim();
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            // 验证是否是有效 JSON
            try {
                objectMapper.readTree(trimmed);
                return trimmed;
            } catch (Exception ignored) {
            }
        }

        // 从字符串中找到第一个 { 和最后一个 }
        int start = input.indexOf('{');
        int end = input.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return input.substring(start, end + 1);
        }

        return "{}";
    }

    /**
     * 从 ChatResponse metadata 提取思考内容
     */
    private String extractThinkingFromMetadata(ChatResponse response) {
        if (response.getResult() != null && response.getResult().getMetadata() != null) {
            var metadata = response.getResult().getMetadata();
            Object thinking = metadata.get("thinking");
            if (thinking != null) {
                return thinking.toString();
            }
        }
        return null;
    }
}
