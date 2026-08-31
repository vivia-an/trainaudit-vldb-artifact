#!/bin/bash
# 批量执行 SDCCheck 约束检查脚本
# 使用方法: ./batch_sdccheck.sh [可选: 指定db目录]

set -e

# 配置：Megatron-LM 根目录（相对于 db 目录的位置）
MEGATRON_ROOT="/volume/qscai/lsk/Megatron-LM"
SCRIPT_DIR="${MEGATRON_ROOT}/megatron/core/pipeline_parallel"

# db 目录与注入脚本的映射
declare -A DB_SCRIPT_MAP=(
    # DP 相关检查
    ["optimizer_state_test_db"]="pretrain_inject_state.sh"
    ["main_grad_test_db"]="pretrain_inject_main_grad_after_backward.sh"
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
    
    # TP 相关检查 - 如果有对应脚本，添加映射；否则使用通用脚本
    ["tp_grad_same_test_db"]="pretrain_inject_grad.sh"
    ["tp_attn_proj_test_db"]="pretrain_inject_state.sh"
    ["tp_optim_state_test_db"]="pretrain_inject_state.sh"
    ["tp_router_test_db"]="pretrain_inject_state.sh"
    ["tp_embedding_test_db"]="pretrain_inject_state.sh"
    ["tp_layernorm_test_db"]="pretrain_inject_state.sh"
    ["tp_shared_experts_same_test_db"]="pretrain_inject_sharded_same.sh"
    ["tp_grad_boundary_jump_test_db"]="pretrain_inject_grad.sh"
    ["tp_layernorm_bias_test_db"]="pretrain_inject_state.sh"
    ["tp_qkv_boundary_jump_test_db"]="pretrain_inject_state.sh"
    ["tp_grad_distribution_test_db"]="pretrain_inject_distribution_shape.sh"
    ["tp_main_grad_test_db"]="pretrain_inject_main_grad_after_backward.sh"
    ["tp_qkv_distribution_test_db"]="pretrain_inject_distribution_shape.sh"
    ["tp_grad_nan_test_db"]="pretrain_inject_grad.sh"
    ["tp_mlp_fc1_test_db"]="pretrain_inject_state.sh"
    ["tp_requires_grad_test_db"]="pretrain_inject_requires_grad.sh"
)

# 日志目录
LOG_DIR="${MEGATRON_ROOT}/sdccheck_logs"
mkdir -p "${LOG_DIR}"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 结果汇总文件
SUMMARY_FILE="${LOG_DIR}/batch_summary_${TIMESTAMP}.txt"

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
    
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "----------------------------------------" | tee -a "${SUMMARY_FILE}"
    echo "处理: ${db_name}" | tee -a "${SUMMARY_FILE}"
    
    # 查找对应的脚本
    local script_name="${DB_SCRIPT_MAP[$db_name]}"
    
    if [ -z "$script_name" ]; then
        echo "⚠️  跳过: 未找到 ${db_name} 对应的注入脚本映射" | tee -a "${SUMMARY_FILE}"
        ((SKIPPED++))
        return
    fi
    
    local script_path="${SCRIPT_DIR}/${script_name}"
    
    if [ ! -f "$script_path" ]; then
        echo "⚠️  跳过: 脚本不存在 ${script_path}" | tee -a "${SUMMARY_FILE}"
        ((SKIPPED++))
        return
    fi
    
    # 检查 Collector 目录是否存在
    local collector_dir="${db_dir}/Collector"
    if [ ! -d "$collector_dir" ]; then
        echo "⚠️  跳过: Collector 目录不存在 ${collector_dir}" | tee -a "${SUMMARY_FILE}"
        ((SKIPPED++))
        return
    fi
    
    # 进入 Collector 目录执行检查
    echo "������ 目录: ${collector_dir}" | tee -a "${SUMMARY_FILE}"
    echo "������ 脚本: ${script_path}" | tee -a "${SUMMARY_FILE}"
    
    local log_file="${LOG_DIR}/${db_name}_${TIMESTAMP}.log"
    
    (
        cd "${collector_dir}"
        echo "执行: sdccheck-merge --auto-evaluate ${script_path} --hierarchical"
        sdccheck-merge --auto-evaluate "${script_path}" --hierarchical
    ) > "${log_file}" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 成功: ${db_name}" | tee -a "${SUMMARY_FILE}"
        ((SUCCESS++))
    else
        echo "❌ 失败: ${db_name} (退出码: ${exit_code})" | tee -a "${SUMMARY_FILE}"
        echo "   日志: ${log_file}" | tee -a "${SUMMARY_FILE}"
        ((FAILED++))
    fi
    
    ((TOTAL++))
}

# 主逻辑
if [ $# -gt 0 ]; then
    # 处理指定的目录
    for db_dir in "$@"; do
        if [ -d "$db_dir" ]; then
            run_check "$db_dir"
        else
            echo "⚠️  目录不存在: $db_dir" | tee -a "${SUMMARY_FILE}"
        fi
    done
else
    # 自动查找所有 *_test_db 目录
    echo "自动查找 *_test_db 目录..." | tee -a "${SUMMARY_FILE}"
    
    for db_dir in ${MEGATRON_ROOT}/*_test_db; do
        if [ -d "$db_dir" ]; then
            run_check "$db_dir"
        fi
    done
fi

# 汇总报告
echo "" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"
echo "批量检查完成" | tee -a "${SUMMARY_FILE}"
echo "结束时间: $(date)" | tee -a "${SUMMARY_FILE}"
echo "----------------------------------------" | tee -a "${SUMMARY_FILE}"
echo "总计: ${TOTAL}" | tee -a "${SUMMARY_FILE}"
echo "成功: ${SUCCESS}" | tee -a "${SUMMARY_FILE}"
echo "失败: ${FAILED}" | tee -a "${SUMMARY_FILE}"
echo "跳过: ${SKIPPED}" | tee -a "${SUMMARY_FILE}"
echo "========================================" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"
echo "详细日志目录: ${LOG_DIR}" | tee -a "${SUMMARY_FILE}"
echo "汇总文件: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
