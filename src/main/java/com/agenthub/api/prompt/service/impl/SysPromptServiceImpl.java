package com.agenthub.api.prompt.service.impl;

import cn.hutool.core.util.StrUtil;
import com.agenthub.api.common.core.page.PageResult;
import com.agenthub.api.prompt.domain.entity.SysPrompt;
import com.agenthub.api.prompt.domain.entity.SysPromptCategory;
import com.agenthub.api.prompt.domain.entity.SysPromptTag;
import com.agenthub.api.prompt.domain.entity.SysPromptTagRelation;
import com.agenthub.api.prompt.domain.entity.SysPromptVersion;
import com.agenthub.api.prompt.domain.dto.request.PromptCreateRequest;
import com.agenthub.api.prompt.domain.dto.request.PromptQueryRequest;
import com.agenthub.api.prompt.domain.dto.request.PromptUpdateRequest;
import com.agenthub.api.prompt.domain.vo.PromptVO;
import com.agenthub.api.prompt.enums.ChangeType;
import com.agenthub.api.prompt.enums.PromptType;
import com.agenthub.api.prompt.enums.Scope;
import com.agenthub.api.prompt.enums.TemplateType;
import com.agenthub.api.prompt.mapper.SysPromptCategoryMapper;
import com.agenthub.api.prompt.mapper.SysPromptMapper;
import com.agenthub.api.prompt.mapper.SysPromptTagRelationMapper;
import com.agenthub.api.prompt.mapper.SysPromptVersionMapper;
import com.agenthub.api.prompt.service.ISysPromptService;
import com.agenthub.api.prompt.service.ISysPromptTagService;
import com.agenthub.api.prompt.service.ISysPromptVersionService;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.databind.JsonNode;
import freemarker.template.Configuration;
import freemarker.template.Template;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.ui.freemarker.FreeMarkerTemplateUtils;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 提示词管理 Service 实现
 *
 * @author AgentHub
 * @since 2026-01-27
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysPromptServiceImpl extends ServiceImpl<SysPromptMapper, SysPrompt> implements ISysPromptService {

    private final SysPromptMapper sysPromptMapper;
    private final SysPromptTagRelationMapper tagRelationMapper;
    private final SysPromptCategoryMapper categoryMapper;
    private final SysPromptVersionMapper versionMapper;
    private final ISysPromptTagService tagService;
    private final ISysPromptVersionService versionService;

    // ==================== Freemarker 模板引擎配置 ====================
    /**
     * Freemarker 配置实例
     *
     * <p>作用：用于渲染存储在数据库中的提示词模板。
     *
     * <p>场景示例：
     * <pre>
     * 数据库中存储的模板（Freemarker 格式）：
     * ┌─────────────────────────────────────────────────────────────────┐
     * │ 你是一个${role}，负责处理${industry}行业的业务。                 │
     * │ 当前时间是：${currentTime}                                       │
     * │ 用户问题：${userQuery}                                           │
     * │                                                                 │
     * │ 可用工具列表：                                                   │
     * │ &lt;#list tools as tool&gt;                                      │
     * │   - ${tool.name}: ${tool.description}                            │
     * │ &lt;/#list&gt;                                                  │
     * └─────────────────────────────────────────────────────────────────┘
     *
     * 调用 render("SYSTEM_PROMPT", variables):
     * ┌─────────────────────────────────────────────────────────────────┐
     * │ variables = {                                                   │
     * │   "role": "电力行业助手",                                        │
     * │   "industry": "电力",                                           │
     * │   "currentTime": "2026-01-27",                                  │
     * │   "userQuery": "帮我算一下电费",                                 │
     * │   "tools": [...]                                                │
     * │ }                                                               │
     * └─────────────────────────────────────────────────────────────────┘
     *                           ↓
     * 渲染后的提示词：
     * ┌─────────────────────────────────────────────────────────────────┐
     * │ 你是一个电力行业助手，负责处理电力行业的业务。                    │
     * │ 当前时间是：2026-01-27                                           │
     * │ 用户问题：帮我算一下电费                                         │
     * │                                                                 │
     * │ 可用工具列表：                                                   │
     * │   - CalculatorTool: 计算电费                                    │
     * │   - KnowledgeTool: 查询电力知识                                  │
     * └─────────────────────────────────────────────────────────────────┘
     * </pre>
     */
    private final Configuration freemarkerConfig = new Configuration(Configuration.VERSION_2_3_32);

    {
        // 初始化块（实例初始化器）：在构造函数执行后立即执行
        // 设置数字格式：最多保留 6 位小数，避免不必要的科学计数法
        // 例如：123.45 不会变成 123.450000，0.000001 不会被格式化为 1.0E-6
        freemarkerConfig.setNumberFormat("0.######");

        // 设置布尔值格式：使用英文 "true,false" 而不是默认的 "true,false"
        // 这确保模板中的布尔判断行为一致
        freemarkerConfig.setBooleanFormat("true,false");
    }

    /**
     * 渲染提示词模板
     * <p>结果缓存于 Redis，key 格式: prompt::render::{promptCode}
     * <p>注意：对于 TEXT 类型的模板，variables 参数不会被使用，因此可以安全缓存
     *
     * @param promptCode 提示词代码（如 "SYSTEM_RAG_v1.0"）
     * @param variables 模板变量（如 {"userName": "张三", "tools": [...]}）
     * @return 渲染后的提示词文本
     * @throws IllegalArgumentException 提示词不存在
     * @throws RuntimeException 渲染失败
     */
    @Override
    @Cacheable(value = "prompt", key = "'render::' + #promptCode")
    public String render(String promptCode, Map<String, Object> variables) {
        // Step 1: 从数据库获取提示词配置
        SysPrompt prompt = this.getByCode(promptCode);
        if (prompt == null) {
            log.warn("Prompt not found: {}", promptCode);
            throw new IllegalArgumentException("Prompt not found: " + promptCode);
        }

        // Step 2: 从 JSON 中提取模板字符串
        // 数据库存储格式: {"template": "你好 ${name}...", "otherField": "...}
        String templateContent = extractTemplate(prompt.getContent());
        if (StrUtil.isBlank(templateContent)) {
            return "";
        }

        // Step 3: 根据模板类型进行渲染
        try {
            if (TemplateType.FREEMARKER.equals(prompt.getTemplateType())) {
                // FREEMARKER 类型：使用 Freemarker 引擎解析模板
                // new Template(name, reader, config) - 从字符串创建模板
                Template template = new Template(promptCode, new StringReader(templateContent), freemarkerConfig);
                // 将变量注入模板，生成最终字符串
                return FreeMarkerTemplateUtils.processTemplateIntoString(template, variables);
            } else {
                // TEXT 类型：直接返回模板内容，不做任何解析
                // 注意：当前版本暂不支持 SPEL (Spring Expression Language)
                return templateContent;
            }
        } catch (Exception e) {
            log.error("Failed to render prompt [{}]: {}", promptCode, e.getMessage());
            throw new RuntimeException("Prompt rendering failed", e);
        }
    }

    @Override
    public PageResult<PromptVO> selectPage(PromptQueryRequest request) {
        LambdaQueryWrapper<SysPrompt> wrapper = new LambdaQueryWrapper<>();

        // 关键词搜索：代码或名称任一匹配即可 (OR)
        boolean hasKeyword = StrUtil.isNotBlank(request.getPromptCode());
        if (hasKeyword) {
            String keyword = request.getPromptCode();
            wrapper.and(w -> w
                    .like(SysPrompt::getPromptCode, keyword)
                    .or()
                    .like(SysPrompt::getPromptName, keyword));
        }
        // 提示词类型精确查询
        if (StrUtil.isNotBlank(request.getPromptType())) {
            try {
                wrapper.eq(SysPrompt::getPromptType, PromptType.valueOf(request.getPromptType()));
            } catch (IllegalArgumentException e) {
                log.warn("无效的提示词类型: {}", request.getPromptType());
            }
        }
        // 分类查询
        wrapper.eq(request.getCategoryId() != null, SysPrompt::getCategoryId, request.getCategoryId());
        // 激活状态查询
        wrapper.eq(request.getIsActive() != null, SysPrompt::getIsActive, request.getIsActive());
        // 租户查询
        wrapper.eq(request.getTenantId() != null, SysPrompt::getTenantId, request.getTenantId());
        // 作用域查询
        if (StrUtil.isNotBlank(request.getScope())) {
            wrapper.eq(SysPrompt::getScope, Scope.valueOf(request.getScope()));
        }

        wrapper.eq(SysPrompt::getDelFlag, 0);
        wrapper.orderByDesc(SysPrompt::getPriority).orderByDesc(SysPrompt::getCreateTime);

        IPage<SysPrompt> page = this.page(request.build(), wrapper);
        // 手动构建 PageResult，需要将 SysPrompt 转换为 PromptVO
        PageResult<PromptVO> result = new PageResult<>();
        result.setRows(buildVOList(page.getRecords()));
        result.setTotal(page.getTotal());
        result.setPageNum(page.getCurrent());
        result.setPageSize(page.getSize());
        return result;
    }

    @Override
    public List<SysPrompt> listByType(PromptType promptType) {
        LambdaQueryWrapper<SysPrompt> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SysPrompt::getPromptType, promptType)
                .eq(SysPrompt::getIsActive, true)
                .eq(SysPrompt::getDelFlag, 0)
                .orderByDesc(SysPrompt::getPriority)
                .orderByDesc(SysPrompt::getCreateTime);
        return sysPromptMapper.selectList(wrapper);
    }

    @Override
    public SysPrompt getByCode(String promptCode) {
        LambdaQueryWrapper<SysPrompt> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SysPrompt::getPromptCode, promptCode)
                .eq(SysPrompt::getDelFlag, 0);
        return sysPromptMapper.selectOne(wrapper);
    }

    @Override
    public PromptVO getDetail(Long id) {
        SysPrompt prompt = this.getById(id);
        if (prompt == null || prompt.getDelFlag() == 1) {
            return null;
        }

        PromptVO vo = buildVO(prompt);

        // 填充分类信息
        if (prompt.getCategoryId() != null) {
            SysPromptCategory category = categoryMapper.selectById(prompt.getCategoryId());
            if (category != null) {
                vo.setCategoryName(category.getCategoryName());
            }
        }

        // 填充标签信息
        List<SysPromptTag> tags = tagService.listByPromptId(id);
        List<PromptVO.TagVO> tagVOList = tags.stream()
                .map(tag -> PromptVO.TagVO.builder()
                        .id(tag.getId())
                        .tagName(tag.getTagName())
                        .tagColor(tag.getTagColor())
                        .build())
                .collect(Collectors.toList());
        vo.setTags(tagVOList);

        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Long create(PromptCreateRequest request) {
        // 检查代码是否重复
        SysPrompt existing = getByCode(request.getPromptCode());
        if (existing != null) {
            throw new IllegalArgumentException("提示词代码已存在: " + request.getPromptCode());
        }

        SysPrompt prompt = new SysPrompt();
        BeanUtils.copyProperties(request, prompt);

        // 设置枚举类型
        prompt.setPromptType(PromptType.valueOf(request.getPromptType()));
        prompt.setTemplateType(TemplateType.valueOf(request.getTemplateType()));
        prompt.setScope(Scope.valueOf(request.getScope()));
        prompt.setIsActive(true);
        prompt.setIsLocked(false);

        boolean saved = this.save(prompt);
        if (!saved) {
            throw new RuntimeException("创建提示词失败");
        }

        // 绑定标签
        if (request.getTagIds() != null && !request.getTagIds().isEmpty()) {
            bindTags(prompt.getId(), request.getTagIds());
        }

        // 创建初始版本快照
        createVersionSnapshot(prompt.getId(), "初始化版本");

        return prompt.getId();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @CacheEvict(value = "prompt", allEntries = true)
    public Boolean update(PromptUpdateRequest request) {
        SysPrompt prompt = this.getById(request.getId());
        if (prompt == null || prompt.getDelFlag() == 1) {
            throw new IllegalArgumentException("提示词不存在");
        }

        // 检查是否锁定
        if (prompt.getIsLocked()) {
            throw new IllegalArgumentException("提示词已锁定，无法修改");
        }

        // 创建版本快照（修改前）
        if (request.getContent() != null || StrUtil.isNotBlank(request.getChangeReason())) {
            createVersionSnapshot(request.getId(), request.getChangeReason());
        }

        // 更新基本信息
        if (StrUtil.isNotBlank(request.getPromptName())) {
            prompt.setPromptName(request.getPromptName());
        }
        if (request.getContent() != null) {
            prompt.setContent(request.getContent());
        }
        if (request.getCategoryId() != null) {
            prompt.setCategoryId(request.getCategoryId());
        }
        if (request.getVersion() != null) {
            prompt.setVersion(request.getVersion());
        }
        if (request.getIsActive() != null) {
            prompt.setIsActive(request.getIsActive());
        }
        if (request.getPriority() != null) {
            prompt.setPriority(request.getPriority());
        }
        if (StrUtil.isNotBlank(request.getRemark())) {
            prompt.setRemark(request.getRemark());
        }

        boolean updated = this.updateById(prompt);

        // 更新标签关联
        if (request.getTagIds() != null) {
            // 先删除旧关联
            LambdaQueryWrapper<SysPromptTagRelation> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(SysPromptTagRelation::getPromptId, request.getId());
            tagRelationMapper.delete(wrapper);

            // 添加新关联
            if (!request.getTagIds().isEmpty()) {
                bindTags(request.getId(), request.getTagIds());
            }
        }

        return updated;
    }

    @Override
    @CacheEvict(value = "prompt", allEntries = true)
    public Boolean delete(Long id) {
        SysPrompt prompt = this.getById(id);
        if (prompt == null) {
            return false;
        }

        // 检查是否锁定
        if (prompt.getIsLocked()) {
            throw new IllegalArgumentException("提示词已锁定，无法删除");
        }

        // 删除标签关联
        LambdaQueryWrapper<SysPromptTagRelation> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SysPromptTagRelation::getPromptId, id);
        tagRelationMapper.delete(wrapper);

        return this.removeById(id);
    }

    @Override
    @CacheEvict(value = "prompt", allEntries = true)
    public Boolean toggleActive(Long id, Boolean isActive) {
        SysPrompt prompt = new SysPrompt();
        prompt.setId(id);
        prompt.setIsActive(isActive);
        return sysPromptMapper.updateById(prompt) > 0;
    }

    @Override
    @CacheEvict(value = "prompt", allEntries = true)
    public Boolean toggleLocked(Long id, Boolean isLocked) {
        SysPrompt prompt = new SysPrompt();
        prompt.setId(id);
        prompt.setIsLocked(isLocked);
        return sysPromptMapper.updateById(prompt) > 0;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Boolean bindTags(Long promptId, List<Long> tagIds) {
        if (tagIds == null || tagIds.isEmpty()) {
            return true;
        }

        // 查询已存在的关联
        LambdaQueryWrapper<SysPromptTagRelation> existsWrapper = new LambdaQueryWrapper<>();
        existsWrapper.eq(SysPromptTagRelation::getPromptId, promptId)
                .in(SysPromptTagRelation::getTagId, tagIds);
        List<SysPromptTagRelation> existsList = tagRelationMapper.selectList(existsWrapper);

        // 已存在的标签ID
        List<Long> existsTagIds = existsList.stream()
                .map(SysPromptTagRelation::getTagId)
                .collect(Collectors.toList());

        // 过滤出需要新增的标签
        List<Long> newTagIds = tagIds.stream()
                .filter(tagId -> !existsTagIds.contains(tagId))
                .collect(Collectors.toList());

        if (!newTagIds.isEmpty()) {
            List<SysPromptTagRelation> relations = new ArrayList<>();
            for (Long tagId : newTagIds) {
                SysPromptTagRelation relation = SysPromptTagRelation.builder()
                        .promptId(promptId)
                        .tagId(tagId)
                        .build();
                relations.add(relation);
            }

            // 批量插入
            for (SysPromptTagRelation relation : relations) {
                tagRelationMapper.insert(relation);
            }
        }

        return true;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public Boolean unbindTags(Long promptId, List<Long> tagIds) {
        if (tagIds == null || tagIds.isEmpty()) {
            return true;
        }

        LambdaQueryWrapper<SysPromptTagRelation> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SysPromptTagRelation::getPromptId, promptId)
                .in(SysPromptTagRelation::getTagId, tagIds);
        return tagRelationMapper.delete(wrapper) >= 0;
    }

    @Override
    public Long createVersionSnapshot(Long promptId, String changeReason) {
        SysPrompt prompt = this.getById(promptId);
        if (prompt == null) {
            throw new IllegalArgumentException("提示词不存在");
        }

        // 查询最新版本号
        LambdaQueryWrapper<SysPromptVersion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SysPromptVersion::getPromptId, promptId)
                .eq(SysPromptVersion::getDelFlag, 0)
                .orderByDesc(SysPromptVersion::getCreateTime)
                .last("LIMIT 1");
        SysPromptVersion lastVersion = versionService.getByVersion(promptId, prompt.getVersion());

        SysPromptVersion version = new SysPromptVersion();
        version.setPromptId(promptId);
        version.setPromptCode(prompt.getPromptCode());
        version.setVersion(prompt.getVersion());
        version.setContent(prompt.getContent());
        version.setChangeType(ChangeType.UPDATE);
        version.setChangeReason(changeReason);
        version.setChangeFromVersion(lastVersion != null ? lastVersion.getVersion() : null);

        return versionService.create(version);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @CacheEvict(value = "prompt", allEntries = true)
    public Boolean rollbackToVersion(Long promptId, Long versionId) {
        SysPrompt prompt = this.getById(promptId);
        if (prompt == null) {
            throw new IllegalArgumentException("提示词不存在");
        }

        SysPromptVersion targetVersion = versionMapper.selectById(versionId);
        if (targetVersion == null || !targetVersion.getPromptId().equals(promptId)) {
            throw new IllegalArgumentException("版本不存在");
        }

        // 检查是否锁定
        if (prompt.getIsLocked()) {
            throw new IllegalArgumentException("提示词已锁定，无法回滚");
        }

        // 创建回滚前快照
        createVersionSnapshot(promptId, "回滚前快照");

        // 恢复内容
        prompt.setContent(targetVersion.getContent());
        prompt.setVersion(targetVersion.getVersion());
        boolean updated = this.updateById(prompt);

        // 记录回滚版本
        SysPromptVersion rollbackVersion = new SysPromptVersion();
        rollbackVersion.setPromptId(promptId);
        rollbackVersion.setPromptCode(prompt.getPromptCode());
        rollbackVersion.setVersion(targetVersion.getVersion() + "-rollback");
        rollbackVersion.setContent(targetVersion.getContent());
        rollbackVersion.setChangeType(ChangeType.ROLLBACK);
        rollbackVersion.setChangeReason("回滚到版本: " + targetVersion.getVersion());
        rollbackVersion.setChangeFromVersion(prompt.getVersion());
        versionService.create(rollbackVersion);

        return updated;
    }

    /**
     * 构建单个 VO
     */
    private PromptVO buildVO(SysPrompt prompt) {
        if (prompt == null) {
            return null;
        }

        return PromptVO.builder()
                .id(prompt.getId())
                .promptCode(prompt.getPromptCode())
                .promptName(prompt.getPromptName())
                .promptType(prompt.getPromptType() != null ? prompt.getPromptType().name() : null)
                .template(extractTemplate(prompt.getContent()))
                .content(prompt.getContent())
                .templateType(prompt.getTemplateType() != null ? prompt.getTemplateType().name() : null)
                .version(prompt.getVersion())
                .isActive(prompt.getIsActive())
                .isLocked(prompt.getIsLocked())
                .scope(prompt.getScope() != null ? prompt.getScope().name() : null)
                .priority(prompt.getPriority())
                .createTime(prompt.getCreateTime())
                .updateTime(prompt.getUpdateTime())
                .remark(prompt.getRemark())
                .build();
    }

    /**
     * 构建 VO 列表
     */
    private List<PromptVO> buildVOList(List<SysPrompt> list) {
        return list.stream()
                .map(this::buildVO)
                .collect(Collectors.toList());
    }

    /**
     * 从 content 中提取 template
     */
    private String extractTemplate(JsonNode content) {
        if (content == null || !content.has("template")) {
            return null;
        }
        return content.get("template").asText();
    }
}
