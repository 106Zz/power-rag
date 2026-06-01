package com.agenthub.api.agent_engine.service.impl;

import com.agenthub.api.agent_engine.model.EvalDashboardVO;
import com.agenthub.api.agent_engine.service.EvalDashboardService;
import com.agenthub.api.common.core.page.PageResult;
import com.agenthub.api.prompt.domain.entity.CaseSnapshot;
import com.agenthub.api.prompt.enums.CaseStatus;
import com.agenthub.api.prompt.mapper.CaseSnapshotMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 评估大盘服务实现
 * <p>直接从 case_snapshot 表聚合数据</p>
 *
 * @author AgentHub
 * @since 2026-04-26
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EvalDashboardServiceImpl implements EvalDashboardService {

    private final CaseSnapshotMapper caseSnapshotMapper;

    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter DATETIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    @Override
    public EvalDashboardVO.EvalSummary getSummary() {
        // 总数
        Long total = caseSnapshotMapper.selectCount(new LambdaQueryWrapper<>());

        // 查询所有有 ai_judge_result 的 Case（真正的评估数据）
        List<CaseSnapshot> judgedCases = caseSnapshotMapper.selectList(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .isNotNull(CaseSnapshot::getAiJudgeResult));

        long judgePassed = 0;
        long judgeFailed = 0;
        for (CaseSnapshot cs : judgedCases) {
            JsonNode judge = cs.getAiJudgeResult();
            if (judge != null && judge.has("passed")) {
                if (judge.get("passed").asBoolean(true)) {
                    judgePassed++;
                } else {
                    judgeFailed++;
                }
            }
        }

        long judgedTotal = judgePassed + judgeFailed;
        double passRate = judgedTotal > 0 ? (judgePassed * 100.0 / judgedTotal) : 0;
        double interceptRate = judgedTotal > 0 ? (judgeFailed * 100.0 / judgedTotal) : 0;

        // 系统层面的失败数（status = FAILED）
        Long systemFailed = caseSnapshotMapper.selectCount(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .eq(CaseSnapshot::getStatus, CaseStatus.FAILED));

        // 平均耗时
        double avgDuration = 0;
        List<CaseSnapshot> casesWithDuration = caseSnapshotMapper.selectList(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .isNotNull(CaseSnapshot::getDurationMs)
                        .select(CaseSnapshot::getDurationMs));
        if (!casesWithDuration.isEmpty()) {
            avgDuration = casesWithDuration.stream()
                    .mapToInt(CaseSnapshot::getDurationMs)
                    .average()
                    .orElse(0);
        }

        return EvalDashboardVO.EvalSummary.builder()
                .totalCases(total)
                .completedCases(judgePassed)
                .failedCases(judgeFailed)
                .passRate(Math.round(passRate * 10.0) / 10.0)
                .aiInterceptRate(Math.round(interceptRate * 10.0) / 10.0)
                .avgDurationMs(Math.round(avgDuration * 10.0) / 10.0)
                .evaluatedCount((long) judgedTotal)
                .build();
    }

    @Override
    public List<EvalDashboardVO.TrendPoint> getTrend(int days) {
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minusDays(days - 1);

        LocalDateTime startDateTime = startDate.atStartOfDay();
        LocalDateTime endDateTime = endDate.atTime(LocalTime.MAX);

        // 查询时间范围内有 Judge 结果的 Case
        List<CaseSnapshot> cases = caseSnapshotMapper.selectList(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .ge(CaseSnapshot::getRequestTime, startDateTime)
                        .le(CaseSnapshot::getRequestTime, endDateTime)
                        .isNotNull(CaseSnapshot::getAiJudgeResult));

        Map<LocalDate, List<CaseSnapshot>> grouped = cases.stream()
                .collect(Collectors.groupingBy(cs -> cs.getRequestTime().toLocalDate()));

        List<EvalDashboardVO.TrendPoint> trend = new ArrayList<>();
        for (int i = 0; i < days; i++) {
            LocalDate date = startDate.plusDays(i);
            List<CaseSnapshot> dayCases = grouped.getOrDefault(date, Collections.emptyList());

            long dayPassed = 0;
            long dayFailed = 0;
            for (CaseSnapshot cs : dayCases) {
                JsonNode judge = cs.getAiJudgeResult();
                if (judge != null && judge.has("passed")) {
                    if (judge.get("passed").asBoolean(true)) {
                        dayPassed++;
                    } else {
                        dayFailed++;
                    }
                }
            }
            long dayTotal = dayPassed + dayFailed;
            double dayPassRate = dayTotal > 0 ? (dayPassed * 100.0 / dayTotal) : 0;

            trend.add(EvalDashboardVO.TrendPoint.builder()
                    .date(date.format(DATE_FMT))
                    .count(dayTotal)
                    .passRate(Math.round(dayPassRate * 10.0) / 10.0)
                    .build());
        }

        return trend;
    }

    @Override
    public EvalDashboardVO.ErrorBreakdown getErrorBreakdown() {
        // 查询所有有错误信息的 Case
        List<CaseSnapshot> failedCases = caseSnapshotMapper.selectList(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .eq(CaseSnapshot::getStatus, CaseStatus.FAILED)
                        .isNotNull(CaseSnapshot::getErrorMessage));

        // 查询有 AI Judge 拦截的 Case
        List<CaseSnapshot> aiJudgedCases = caseSnapshotMapper.selectList(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .isNotNull(CaseSnapshot::getAiJudgeResult));

        // 错误分类
        Map<String, Long> errorTypeCount = new LinkedHashMap<>();
        errorTypeCount.put("幻觉/编造事实", 0L);
        errorTypeCount.put("越权回答", 0L);
        errorTypeCount.put("格式错误", 0L);
        errorTypeCount.put("检索失败", 0L);
        errorTypeCount.put("系统异常", 0L);
        errorTypeCount.put("其他", 0L);

        // 分析失败 Case
        for (CaseSnapshot cs : failedCases) {
            String error = cs.getErrorMessage() != null ? cs.getErrorMessage().toLowerCase() : "";
            String type = classifyError(error);
            errorTypeCount.merge(type, 1L, Long::sum);
        }

        // 分析 AI Judge 拦截的 Case
        for (CaseSnapshot cs : aiJudgedCases) {
            JsonNode judgeResult = cs.getAiJudgeResult();
            if (judgeResult != null && judgeResult.has("passed") && !judgeResult.get("passed").asBoolean(true)) {
                String reason = judgeResult.has("reason") ? judgeResult.get("reason").asText("").toLowerCase() : "";
                String type = classifyAiJudgeError(reason);
                errorTypeCount.merge(type, 1L, Long::sum);
            }
        }

        // 计算总数和占比
        long total = errorTypeCount.values().stream().mapToLong(Long::longValue).sum();
        List<EvalDashboardVO.ErrorItem> items = new ArrayList<>();

        for (Map.Entry<String, Long> entry : errorTypeCount.entrySet()) {
            if (entry.getValue() > 0) {
                double pct = total > 0 ? (entry.getValue() * 100.0 / total) : 0;
                items.add(EvalDashboardVO.ErrorItem.builder()
                        .type(entry.getKey())
                        .count(entry.getValue())
                        .percentage(Math.round(pct * 10.0) / 10.0)
                        .build());
            }
        }

        // 按数量降序
        items.sort((a, b) -> Long.compare(b.getCount(), a.getCount()));

        return EvalDashboardVO.ErrorBreakdown.builder().items(items).build();
    }

    @Override
    public PageResult<EvalDashboardVO.BadCaseItem> getBadCases(int pageNum, int pageSize) {
        // 查询系统失败 OR AI Judge 判定失败的 Case
        LambdaQueryWrapper<CaseSnapshot> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w
                .eq(CaseSnapshot::getStatus, CaseStatus.FAILED)
                .or()
                .apply("ai_judge_result->>'passed' = 'false'"));
        wrapper.orderByDesc(CaseSnapshot::getRequestTime);

        IPage<CaseSnapshot> page = caseSnapshotMapper.selectPage(
                new Page<>(pageNum, pageSize), wrapper);

        List<EvalDashboardVO.BadCaseItem> items = page.getRecords().stream()
                .map(this::toBadCaseItem)
                .collect(Collectors.toList());

        PageResult<EvalDashboardVO.BadCaseItem> result = new PageResult<>();
        result.setRows(items);
        result.setTotal(page.getTotal());
        result.setPageNum(page.getCurrent());
        result.setPageSize(page.getSize());
        return result;
    }

    @Override
    public EvalDashboardVO.CaseDetail getCaseDetail(String caseId) {
        CaseSnapshot snapshot = caseSnapshotMapper.selectOne(
                new LambdaQueryWrapper<CaseSnapshot>()
                        .eq(CaseSnapshot::getCaseId, caseId));

        if (snapshot == null) {
            return null;
        }

        String query = "";
        if (snapshot.getInputData() != null && snapshot.getInputData().has("query")) {
            query = snapshot.getInputData().get("query").asText("");
        }

        String rawResponse = "";
        if (snapshot.getOutputData() != null && snapshot.getOutputData().has("raw_response")) {
            rawResponse = snapshot.getOutputData().get("raw_response").asText("");
        }

        return EvalDashboardVO.CaseDetail.builder()
                .caseId(snapshot.getCaseId())
                .scenario(snapshot.getScenario() != null ? snapshot.getScenario().name() : "")
                .intent(snapshot.getIntent())
                .status(snapshot.getStatus() != null ? snapshot.getStatus().name() : "")
                .query(query)
                .contextData(snapshot.getContextData())
                .promptData(snapshot.getPromptData())
                .rawResponse(rawResponse)
                .ruleJudgeResult(snapshot.getRuleJudgeResult())
                .aiJudgeResult(snapshot.getAiJudgeResult())
                .errorMessage(snapshot.getErrorMessage())
                .requestTime(snapshot.getRequestTime() != null ? snapshot.getRequestTime().format(DATETIME_FMT) : "")
                .responseTime(snapshot.getResponseTime() != null ? snapshot.getResponseTime().format(DATETIME_FMT) : "")
                .durationMs(snapshot.getDurationMs())
                .build();
    }

    private EvalDashboardVO.BadCaseItem toBadCaseItem(CaseSnapshot cs) {
        String query = "";
        if (cs.getInputData() != null && cs.getInputData().has("query")) {
            query = cs.getInputData().get("query").asText("");
        }

        String errorType = "系统异常";
        String errorReason = cs.getErrorMessage();

        // 优先从 AI Judge 结果获取错误信息
        if (cs.getAiJudgeResult() != null) {
            JsonNode judge = cs.getAiJudgeResult();
            if (judge.has("passed") && !judge.get("passed").asBoolean(true)) {
                errorType = "AI Judge 拦截";
                errorReason = judge.has("reason") ? judge.get("reason").asText("") : "未通过 AI 评估";
            }
        }

        return EvalDashboardVO.BadCaseItem.builder()
                .caseId(cs.getCaseId())
                .query(query)
                .scenario(cs.getScenario() != null ? cs.getScenario().name() : "")
                .errorType(errorType)
                .errorReason(errorReason)
                .requestTime(cs.getRequestTime() != null ? cs.getRequestTime().format(DATETIME_FMT) : "")
                .durationMs(cs.getDurationMs())
                .build();
    }

    private String classifyError(String errorMsg) {
        if (errorMsg == null) return "其他";
        if (errorMsg.contains("幻觉") || errorMsg.contains("编造") || errorMsg.contains("hallucin")) return "幻觉/编造事实";
        if (errorMsg.contains("越权") || errorMsg.contains("out of domain") || errorMsg.contains("超出范围")) return "越权回答";
        if (errorMsg.contains("格式") || errorMsg.contains("format") || errorMsg.contains("json")) return "格式错误";
        if (errorMsg.contains("检索") || errorMsg.contains("recall") || errorMsg.contains("召回")) return "检索失败";
        if (errorMsg.contains("exception") || errorMsg.contains("error") || errorMsg.contains("超时") || errorMsg.contains("timeout")) return "系统异常";
        return "其他";
    }

    private String classifyAiJudgeError(String reason) {
        if (reason == null) return "其他";
        if (reason.contains("幻觉") || reason.contains("编造") || reason.contains("虚假") || reason.contains("事实")) return "幻觉/编造事实";
        if (reason.contains("越权") || reason.contains("证据为空") || reason.contains("证据内容过少")) return "越权回答";
        if (reason.contains("格式")) return "格式错误";
        return "其他";
    }
}
