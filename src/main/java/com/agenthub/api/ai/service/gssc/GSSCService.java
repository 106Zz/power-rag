package com.agenthub.api.ai.service.gssc;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * GSC 流水线服务（原 GSSC，去掉了 Select 阶段）
 * <p>
 * 实现 Structure → Compress 流程
 * </p>
 *
 * <h3>设计原则：</h3>
 * <ul>
 *   <li>证据块已经过 Reranker 精排，不再重复筛选</li>
 *   <li>历史消息由滑动窗口控制，不再用词法匹配重排</li>
 *   <li>GSC 只负责：结构化组装 + Token 预算压缩</li>
 * </ul>
 *
 * <h3>流程：</h3>
 * <ul>
 *   <li>Structure: 按模板结构化输出，拆分为独立段落</li>
 *   <li>Compress: 按优先级逐段压缩（HISTORY→EVIDENCE→TOOLS，SYSTEM/TASK 永不压缩）</li>
 * </ul>
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class GSSCService {

    @Value("${gssc.max-tokens:3000}")
    private int maxTokens;

    @Value("${gssc.enabled:false}")
    private boolean enabled;

    /**
     * 段落类型 + 压缩优先级（数值越小越先被砍，MAX_VALUE = 受保护永不压缩）
     */
    private enum SectionType {
        SYSTEM(Integer.MAX_VALUE),
        EVIDENCE(2),
        TOOL_RESULT(3),
        HISTORY(1),
        TASK(Integer.MAX_VALUE),//用户问题
        OUTPUT(Integer.MAX_VALUE);

        final int compressPriority;

        SectionType(int compressPriority) {
            this.compressPriority = compressPriority;
        }

        boolean isProtected() {
            return compressPriority == Integer.MAX_VALUE;
        }
    }

    private record Section(SectionType type, String label, String content) {
        int tokens() {
            return estimateTokens(content);
        }
    }

    // ── 公共 API ──────────────────────────────────────────────

    public String process(
            String evidenceText,
            String historyText,
            String toolResultText,
            String systemPrompt,
            String userQuery) {

        if (!enabled) {
            return buildSimpleContext(evidenceText, historyText, toolResultText, systemPrompt, userQuery);
        }

        log.debug("【GSC】开始处理: evidenceLen={}, historyLen={}, toolsLen={}",
                evidenceText != null ? evidenceText.length() : 0,
                historyText != null ? historyText.length() : 0,
                toolResultText != null ? toolResultText.length() : 0);

        List<Section> sections = buildSections(evidenceText, historyText, toolResultText, systemPrompt, userQuery);
        sections = compressByPriority(sections);
        return renderSections(sections);
    }

    public String processEvidence(String evidenceText, String userQuery) {
        return process(evidenceText, null, null, null, userQuery);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public int getMaxTokens() {
        return maxTokens;
    }

    // ── Structure: 构建段落列表 ────────────────────────────────

    private List<Section> buildSections(
            String evidence, String history, String tools,
            String systemPrompt, String userQuery) {

        List<Section> sections = new ArrayList<>();

        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            sections.add(new Section(SectionType.SYSTEM, "[System]", systemPrompt));
        }
        if (evidence != null && !evidence.isEmpty()) {
            sections.add(new Section(SectionType.EVIDENCE, "[Evidence]", evidence));
        }
        if (tools != null && !tools.isEmpty()) {
            sections.add(new Section(SectionType.TOOL_RESULT, "[工具执行结果]", tools));
        }
        if (history != null && !history.isEmpty()) {
            sections.add(new Section(SectionType.HISTORY, "[Context]", history));
        }
        sections.add(new Section(SectionType.TASK, "[Task]", userQuery));
        sections.add(new Section(SectionType.OUTPUT, "[Output]", "请基于以上信息回答问题。"));

        return sections;
    }

    // ── Compress: 按优先级逐段压缩 ─────────────────────────────

    /**
     * 1. 计算总 Token，不超预算则直接返回
     * 2. 算出受保护段落占用的 Token，得到可压缩段落的预算空间
     * 3. 按优先级从低到高（HISTORY→EVIDENCE→TOOLS）逐段削减
     * 4. SYSTEM / TASK / OUTPUT 永不压缩
     */
    private List<Section> compressByPriority(List<Section> sections) {
        int totalTokens = sumTokens(sections);
        if (totalTokens <= maxTokens) {
            return sections;
        }

        log.info("【GSC Compress】超出预算: {} tokens > {} tokens，按优先级逐段压缩",
                totalTokens, maxTokens);

        // 受保护段落的 Token 占用
        int protectedTokens = sections.stream()
                .filter(s -> s.type.isProtected())
                .mapToInt(Section::tokens)
                .sum();

        int compressibleBudget = Math.max(0, maxTokens - protectedTokens);
        int compressibleTokens = totalTokens - protectedTokens;

        if (compressibleBudget <= 0) {
            log.warn("【GSC Compress】受保护内容已占满预算，清空所有可压缩段落");
            return clearAllCompressible(sections);
        }

        // 按优先级排序（priority 小的排在前面，先被砍）
        List<Section> sorted = sections.stream()
                .filter(s -> !s.type.isProtected())
                .sorted(Comparator.comparingInt(s -> s.type.compressPriority))
                .toList();

        int excess = compressibleTokens - compressibleBudget;

        for (Section section : sorted) {
            if (excess <= 0) break;

            int sectionTokens = section.tokens();
            if (sectionTokens == 0) continue;

            int cutTokens = Math.min(excess, sectionTokens);
            int targetTokens = sectionTokens - cutTokens;

            int idx = sections.indexOf(section);
            if (idx == -1) continue;

            String compressed;
            if (targetTokens <= 0) {
                compressed = "[本段内容已省略，聚焦核心信息]";
            } else {
                int maxChars = targetTokens * 2;
                compressed = section.content().substring(0, Math.min(maxChars, section.content().length()));
                if (compressed.length() < section.content().length()) {
                    compressed += "\n...(已截断)";
                }
            }

            int actualSaved = sectionTokens - estimateTokens(compressed);
            sections.set(idx, new Section(section.type, section.label, compressed));
            excess -= actualSaved;

            log.debug("【GSC Compress】压缩段落 {}: {} → {} tokens",
                    section.label, sectionTokens, estimateTokens(compressed));
        }

        log.info("【GSC Compress】压缩完成: {} → {} tokens", totalTokens, sumTokens(sections));
        return sections;
    }

    private List<Section> clearAllCompressible(List<Section> sections) {
        List<Section> result = new ArrayList<>();
        for (Section s : sections) {
            if (s.type.isProtected()) {
                result.add(s);
            } else {
                result.add(new Section(s.type, s.label, "[本段内容已省略]"));
            }
        }
        return result;
    }

    // ── Render: 段落列表 → 字符串 ──────────────────────────────

    private String renderSections(List<Section> sections) {
        StringBuilder sb = new StringBuilder();
        for (Section s : sections) {
            sb.append(s.label()).append("\n").append(s.content()).append("\n\n");
        }
        if (sb.length() >= 2) {
            sb.setLength(sb.length() - 2);
        }
        return sb.toString();
    }

    // ── 辅助 ───────────────────────────────────────────────────

    private int sumTokens(List<Section> sections) {
        return sections.stream().mapToInt(Section::tokens).sum();
    }

    private String buildSimpleContext(
            String evidence, String history, String tools,
            String systemPrompt, String userQuery) {

        StringBuilder sb = new StringBuilder();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            sb.append("[System]\n").append(systemPrompt).append("\n\n");
        }
        if (evidence != null && !evidence.isEmpty()) {
            sb.append("[Evidence]\n").append(evidence).append("\n\n");
        }
        if (tools != null && !tools.isEmpty()) {
            sb.append("[工具执行结果]\n").append(tools).append("\n\n");
        }
        if (history != null && !history.isEmpty()) {
            sb.append("[Context]\n").append(history).append("\n");
        }
        sb.append("[Task]\n").append(userQuery);
        return sb.toString();
    }

    /**
     * Token 估算：中文约 1 token ≈ 1.5 字符，英文约 1 token ≈ 4 字符，综合取 1 token ≈ 2 字符
     */
    public static int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        return text.length() / 2;
    }
}
