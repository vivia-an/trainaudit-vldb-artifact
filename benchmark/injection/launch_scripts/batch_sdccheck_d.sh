#!/bin/bash
# 批量执行 SDCCheck 约束检查脚本
# 使用方法: ./batch_sdccheck.sh [可选: 指定db目录]

# 注意: 不使用 set -e，让脚本在单个检查失败时继续执行
# set -e

# 配置：Megatron-LM 根目录
MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"

# 调试模式：打印更多日志
DEBUG=true

debug_log() {
    if [ "$DEBUG" = true ]; then
        echo "[DEBUG $(date '+%H:%M:%S')] $1" | tee -a "${LOG_FILE:-/dev/null}"
    fi
}

# 注入脚本就在 MEGATRON_ROOT 根目录下
SCRIPT_DIR="${MEGATRON_ROOT}"

# db 目录与注入脚本的映射
declare -A DB_SCRIPT_MAP=(
    # DP 相关检查
    ["optimizer_state_test_db"]="pretrain_inject_optimizer_state.sh"
    ["main_grad_test_db"]="pretrain_inject_main_grad.sh"
    ["requires_grad_test_db"]="pretrain_inject_requires_grad.sh"
    ["dtype_test_db"]="pretrain_inject_dtype.sh"
    ["optim_backward_test_db"]="pretrain_inject_optim_backward.sh"
    ["main_grad_after_backward_test_db"]="pretrain_inject_main_grad_after_backward.sh"
    ["requires_grad_before_backward_test_db"]="pretrain_inject_requires_grad_before_backward.sh"
    ["optim_state_before_backward_test_db"]="pretrain_inject_optim_state_before_backward.sh"
    ["quantile_test_db"]="pretrain_inject_quantile.sh"
    ["sharded_same_test_db"]="pretrain_inject_sharded_same.sh"
    ["cksum_before_backward_bitwise_level_test_db"]="pretrain_inject_cksum_before_backward_bitwise_level.sh"
    ["requires_grad_after_backward_test_db"]="pretrain_inject_requires_grad_after_backward.sh"
    ["higher_order_stats_test_db"]="pretrain_inject_higher_order_stats.sh"
    ["optim_state_before_optim_step_test_db"]="pretrain_inject_optim_state_before_optim_step.sh"
    ["lr_test_db"]="pretrain_inject_lr.sh"
    ["shape_test_db"]="pretrain_inject_shape.sh"
    ["distribution_shape_test_db"]="pretrain_inject_distribution_shape.sh"
    ["param_bitwise_before_optim_test_db"]="pretrain_inject_param_bitwise_before_optim.sh"
    ["param_bitwise_after_backward_test_db"]="pretrain_inject_param_bitwise_after_backward.sh"
    ["grad_existence_test_db"]="pretrain_inject_grad_existence.sh"
    ["optim_state_after_step_test_db"]="pretrain_inject_optim_state_after_step.sh"
    ["grad_test_db"]="pretrain_inject_grad.sh"
    
    # TP 相关检查
    ["tp_grad_same_test_db"]="pretrain_inject_tp_grad_same.sh"
    ["tp_attn_proj_test_db"]="pretrain_inject_tp_attn_proj.sh"
    ["tp_optim_state_test_db"]="pretrain_inject_tp_optim_state.sh"
    ["tp_router_test_db"]="pretrain_inject_tp_router.sh"
    ["tp_embedding_test_db"]="pretrain_inject_tp_embedding.sh"
    ["tp_layernorm_test_db"]="pretrain_inject_tp_layernorm.sh"
    ["tp_shared_experts_same_test_db"]="pretrain_inject_tp_shared_experts_same.sh"
    ["tp_grad_boundary_jump_test_db"]="pretrain_inject_tp_grad_boundary_jump.sh"
    ["tp_layernorm_bias_test_db"]="pretrain_inject_tp_layernorm_bias.sh"
    ["tp_qkv_boundary_jump_test_db"]="pretrain_inject_tp_qkv_boundary_jump.sh"
    ["tp_grad_distribution_test_db"]="pretrain_inject_tp_grad_distribution.sh"
    ["tp_main_grad_test_db"]="pretrain_inject_tp_main_grad.sh"
    ["tp_qkv_distribution_test_db"]="pretrain_inject_tp_qkv_distribution.sh"
    ["tp_grad_nan_test_db"]="pretrain_inject_tp_grad_nan.sh"
    ["tp_mlp_fc1_test_db"]="pretrain_inject_tp_mlp_fc1.sh"
    ["tp_requires_grad_test_db"]="pretrain_inject_tp_requires_grad.sh"
)

# 日志目录
LOG_DIR="${MEGATRON_ROOT}/sdccheck_logs"
mkdir -p "${LOG_DIR}"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 结果汇总文件
SUMMARY_FILE="${LOG_DIR}/batch_summary_${TIMESTAMP}.txt"

# 全局日志文件（用于调试）
LOG_FILE="${LOG_DIR}/batch_debug_${TIMESTAMP}.log"

debug_log "脚本启动"
debug_log "MEGATRON_ROOT=${MEGATRON_ROOT}"
debug_log "LOG_DIR=${LOG_DIR}"

echo "========================================" | tee "${SUMMARY_FILE}"
echo "SDCCheck 批量约束检查" | tee -a "${SUMMARY_FILE}"
echo "开始时间: $(date)" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"

# 统计变量
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# 处理函数
run_check() {
    local db_dir=$1
    local db_name=$(basename "$db_dir")
    
    debug_log "========== 开始处理: ${db_name} =========="
    
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "----------------------------------------" | tee -a "${SUMMARY_FILE}"
    echo "处理: ${db_name}" | tee -a "${SUMMARY_FILE}"
    
    # 查找对应的脚本
    local script_name="${DB_SCRIPT_MAP[$db_name]}"
    debug_log "查找脚本映射: db_name=${db_name}, script_name=${script_name}"
    
    if [ -z "$script_name" ]; then
        echo "⚠️  跳过: 未找到 ${db_name} 对应的注入脚本映射" | tee -a "${SUMMARY_FILE}"
        debug_log "跳过: 无映射"
        ((SKIPPED++)) || true
        debug_log "处理完成: ${db_name} (跳过-无映射)"
        return 0
    fi
    
    local script_path="${SCRIPT_DIR}/${script_name}"
    debug_log "脚本路径: ${script_path}"
    
    if [ ! -f "$script_path" ]; then
        echo "⚠️  跳过: 脚本不存在 ${script_path}" | tee -a "${SUMMARY_FILE}"
        debug_log "跳过: 脚本不存在"
        ((SKIPPED++)) || true
        debug_log "处理完成: ${db_name} (跳过-脚本不存在)"
        return 0
    fi
    
    # 检查 Collector 目录是否存在
    local collector_dir="${db_dir}/Collector"
    debug_log "检查 Collector 目录: ${collector_dir}"
    
    if [ ! -d "$collector_dir" ]; then
        echo "⚠️  跳过: Collector 目录不存在 ${collector_dir}" | tee -a "${SUMMARY_FILE}"
        debug_log "跳过: Collector 目录不存在"
        ((SKIPPED++)) || true
        debug_log "处理完成: ${db_name} (跳过-无Collector)"
        return 0
    fi
    
    # 进入 Collector 目录执行检查
    echo "📂 目录: ${collector_dir}" | tee -a "${SUMMARY_FILE}"
    echo "📜 脚本: ${script_path}" | tee -a "${SUMMARY_FILE}"
    
    local log_file="${LOG_DIR}/${db_name}_${TIMESTAMP}.log"
    debug_log "日志文件: ${log_file}"
    
    echo "🚀 开始执行 sdccheck-merge..." | tee -a "${SUMMARY_FILE}"
    debug_log "执行命令: cd ${collector_dir} && sdccheck-merge --auto-evaluate ${script_path} --hierarchical"
    
    # 使用 || true 确保即使子命令失败也不会终止脚本
    (
        cd "${collector_dir}" || exit 1
        echo "执行: sdccheck-merge --auto-evaluate ${script_path} --hierarchical"
        sdccheck-merge --auto-evaluate "${script_path}" --hierarchical
    ) > "${log_file}" 2>&1 || true
    
    local exit_code=$?
    debug_log "sdccheck-merge 执行完成, 退出码: ${exit_code}"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 成功: ${db_name}" | tee -a "${SUMMARY_FILE}"
        ((SUCCESS++)) || true
    else
        echo "❌ 失败: ${db_name} (退出码: ${exit_code})" | tee -a "${SUMMARY_FILE}"
        echo "   日志: ${log_file}" | tee -a "${SUMMARY_FILE}"
        # 显示日志最后几行作为快速参考
        echo "   最后10行日志:" | tee -a "${SUMMARY_FILE}"
        tail -10 "${log_file}" 2>/dev/null | sed 's/^/   > /' | tee -a "${SUMMARY_FILE}"
        ((FAILED++)) || true
    fi
    
    ((TOTAL++)) || true
    debug_log "========== 完成处理: ${db_name} (TOTAL=${TOTAL}, SUCCESS=${SUCCESS}, FAILED=${FAILED}, SKIPPED=${SKIPPED}) =========="
    
    return 0
}

# 主逻辑
debug_log "进入主逻辑, 参数数量: $#"

if [ $# -gt 0 ]; then
    # 处理指定的目录
    debug_log "处理指定的目录列表"
    for db_dir in "$@"; do
        debug_log "检查目录: $db_dir"
        if [ -d "$db_dir" ]; then
            run_check "$db_dir"
        else
            echo "⚠️  目录不存在: $db_dir" | tee -a "${SUMMARY_FILE}"
        fi
    done
else
    # 自动查找所有 *_test_db 目录
    echo "自动查找 *_test_db 目录..." | tee -a "${SUMMARY_FILE}"
    debug_log "开始扫描: ${MEGATRON_ROOT}/*_test_db"
    
    # 先列出所有目录
    db_dirs=(${MEGATRON_ROOT}/*_test_db)
    db_count=${#db_dirs[@]}
    debug_log "找到 ${db_count} 个 *_test_db 目录"
    echo "📊 找到 ${db_count} 个待检查目录" | tee -a "${SUMMARY_FILE}"
    
    current_idx=0
    for db_dir in "${db_dirs[@]}"; do
        ((current_idx++)) || true
        debug_log "循环 [${current_idx}/${db_count}]: ${db_dir}"
        echo "" | tee -a "${SUMMARY_FILE}"
        echo "📌 [${current_idx}/${db_count}] 处理中..." | tee -a "${SUMMARY_FILE}"
        
        if [ -d "$db_dir" ]; then
            run_check "$db_dir"
        else
            debug_log "目录不存在或不是目录: $db_dir"
        fi
        
        debug_log "循环 [${current_idx}/${db_count}] 完成"
    done
    
    debug_log "所有目录处理完成"
fi

debug_log "主逻辑执行完成"

# 汇总报告
debug_log "生成汇总报告"

echo "" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"
echo "批量检查完成" | tee -a "${SUMMARY_FILE}"
echo "结束时间: $(date)" | tee -a "${SUMMARY_FILE}"
echo "----------------------------------------" | tee -a "${SUMMARY_FILE}"
echo "📊 总计: ${TOTAL}" | tee -a "${SUMMARY_FILE}"
echo "✅ 成功: ${SUCCESS}" | tee -a "${SUMMARY_FILE}"
echo "❌ 失败: ${FAILED}" | tee -a "${SUMMARY_FILE}"
echo "⚠️  跳过: ${SKIPPED}" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"
echo "📁 详细日志目录: ${LOG_DIR}" | tee -a "${SUMMARY_FILE}"
echo "📄 汇总文件: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
echo "🔍 调试日志: ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"

debug_log "脚本执行完成 - TOTAL=${TOTAL}, SUCCESS=${SUCCESS}, FAILED=${FAILED}, SKIPPED=${SKIPPED}"
echo ""
echo "🎉 批量检查脚本执行完成!"