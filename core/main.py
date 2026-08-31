#!/usr/bin/env python3
"""
SDCCheck主程序
实现分布式训练检查点一致性验证的完整流程
"""

import argparse
import os
import sys
import re
import logging
from pathlib import Path

from sdccheck.models import TrainingConfig
from sdccheck.llm_orchestrator import LLMOrchestrator
from sdccheck.schema_analyzer import SchemaAnalyzer
from sdccheck.generators import (
    PredefinedConstraintGenerator,
    LLMSQLGenerator,
    LLMResultAnalyzer
)
from sdccheck.database_executor import DatabaseExecutor
from sdccheck.llm_providers import LLMProviderFactory
from sdccheck.config import LLMConfig
from sdccheck.llm_logger import create_new_logger_session, setup_unified_logging
from sdccheck.script_config_injector import ScriptConfigInjector


def get_default_config_path(filename: str) -> str:
    """
    获取默认配置文件路径
    
    根据运行环境自动选择配置文件路径：
    - 如果是安装包环境，使用 site-packages/config/ 路径
    - 如果是开发环境，使用相对路径 config/
    
    参数:
        filename: 配置文件名，如 "predefined_constraints.json"
        
    返回:
        配置文件的绝对或相对路径
    """
    # 方法1: 尝试从安装包路径获取
    try:
        import config as config_pkg
        config_dir = Path(config_pkg.__file__).parent
        config_path = config_dir / filename
        if config_path.exists():
            return str(config_path)
    except (ImportError, AttributeError):
        pass
    
    # 方法2: 尝试相对于 main.py 的路径（开发模式）
    main_dir = Path(__file__).parent
    config_path = main_dir / "config" / filename
    if config_path.exists():
        return str(config_path)
    
    # 方法3: 回退到相对路径（兼容性）
    return f"config/{filename}"


def parse_megatron_script(script_path: str) -> TrainingConfig:
    """
    解析Megatron启动脚本，提取训练配置参数
    
    参数:
        script_path: Megatron启动脚本路径
        
    返回:
        TrainingConfig: 解析得到的训练配置
    """
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Megatron脚本文件不存在: {script_path}")
    
    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 默认配置值
    config_values = {
        'tp': 1,
        'pp': 1,
        'ep': 1,
        'sp': 1,
        'zero_stage': 0
    }
    
    # 解析变量定义
    patterns = {
        'tp': [r'TP_SIZE=\$\{TP_SIZE:-([0-9]+)\}', r'TP_SIZE=([0-9]+)', r'--tensor-model-parallel-size\s+\$\{TP_SIZE\}', r'--tensor-model-parallel-size\s+([0-9]+)'],
        'pp': [r'PP_SIZE=\$\{PP_SIZE:-([0-9]+)\}', r'PP_SIZE=([0-9]+)', r'--pipeline-model-parallel-size\s+\$\{PP_SIZE\}', r'--pipeline-model-parallel-size\s+([0-9]+)'],
        'ep': [r'EP_SIZE=\$\{EP_SIZE:-([0-9]+)\}', r'EP_SIZE=([0-9]+)', r'--expert-model-parallel-size\s+\$\{EP_SIZE\}', r'--expert-model-parallel-size\s+([0-9]+)'],
        'etp': [r'ETP_SIZE=\$\{ETP_SIZE:-([0-9]+)\}', r'ETP_SIZE=([0-9]+)', r'--expert-tensor-parallel-size\s+\$\{ETP_SIZE\}', r'--expert-tensor-parallel-size\s+([0-9]+)']
    }
    
    # 提取各个参数
    for param, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, content)
            if match:
                value = int(match.group(1))
                if param == 'etp':
                    # ETP_SIZE 对应 expert tensor parallel，暂时映射到 sp
                    config_values['sp'] = value
                else:
                    config_values[param] = value
                break
    
    # 检查是否使用了ZeRO优化器
    if '--use-distributed-optimizer' in content:
        # 如果使用了分布式优化器，通常对应ZeRO stage 1或2
        config_values['zero_stage'] = 1
    
    # 检查是否启用了sequence parallel
    if '--sequence-parallel' in content:
        # 如果没有从ETP_SIZE获取到sp值，且启用了sequence parallel，设为1
        if config_values['sp'] == 1 and 'ETP_SIZE' not in content:
            config_values['sp'] = 1
    
    # 计算DP大小
    world_size = None
    
    # 方法1: 尝试从脚本中的显式DP_SIZE变量获取
    dp_pattern = r'DP_SIZE=\$\(\(WORLD_SIZE / PP_SIZE / TP_SIZE\)\)'
    if re.search(dp_pattern, content):
        world_size_match = re.search(r'WORLD_SIZE=\$\(\(([0-9]+) \* \$PET_NNODES\)\)', content)
        if world_size_match:
            base_world_size = int(world_size_match.group(1))
            world_size = base_world_size  # 假设单节点情况，PET_NNODES=1
    
    # 方法2: 从torchrun参数中提取WORLD_SIZE
    if world_size is None:
        # 提取 --nproc_per_node 和 --nnodes
        nproc_match = re.search(r'--nproc[_-]per[_-]node[=\s]+([0-9]+)', content)
        nnodes_match = re.search(r'--nnodes[=\s]+([0-9]+)', content)
        
        if nproc_match:
            nproc_per_node = int(nproc_match.group(1))
            nnodes = int(nnodes_match.group(1)) if nnodes_match else 1
            world_size = nproc_per_node * nnodes
    
    # 计算DP
    if world_size:
        dp_size = world_size // (config_values['pp'] * config_values['tp'] * config_values['ep'])
        config_values['dp'] = max(1, dp_size)
    else:
        # 如果无法获取world_size，使用默认值
        config_values['dp'] = 1
    
    return TrainingConfig(
        dp=config_values['dp'],
        tp=config_values['tp'],
        pp=config_values['pp'],
        ep=config_values['ep'],
        sp=config_values['sp'],
        zero_stage=config_values['zero_stage']
    )


def main():
    """
    主函数：实现完整的SDCCheck流程
    
    流程步骤：
    1. 启动: Orchestrator接收用户提供的数据库路径和训练配置作为输入
    2. 了解数据: Orchestrator调用SchemaAnalyzer来获取数据库的SchemaInfo
    3. 生成约束: Orchestrator将SchemaInfo和TrainingConfig传递给ConstraintGenerator，获得Constraint对象列表
    4. 循环检查: 遍历每一个Constraint对象，执行以下子流程：
       a. 生成SQL: 将当前的Constraint对象交给SQLGenerator，得到对应的SQL查询字符串
       b. 执行查询: 调用DatabaseExecutor执行该SQL，获得结果DataFrame
       c. 分析结果: 将Constraint对象和结果DataFrame交给ResultAnalyzer，获得详细的AnalysisReport
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="SDCCheck - 分布式训练检查点一致性验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py /path/to/database.db /path/to/megatron-script.sh
  python main.py /path/to/database.db /path/to/megatron-script.sh --verbose
        """
    )
    
    parser.add_argument(
        "db_path",
        type=str,
        help="数据库文件路径"
    )
    
    parser.add_argument(
        "script_path",
        type=str,
        help="Megatron启动脚本路径"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细日志信息"
    )
    
    parser.add_argument(
        "--manual",
        action="store_true",
        help="手动执行流程（逐步执行各个模块）"
    )
    
    parser.add_argument(
        "--inject-script-config",
        action="store_true",
        help="从脚本注入参数到约束配置文件"
    )
    
    parser.add_argument(
        "--base-constraints",
        type=str,
        default=get_default_config_path("predefined_constraints.json"),
        help="基础约束配置文件路径（默认: config/predefined_constraints.json）"
    )
    
    parser.add_argument(
        "--output-constraints",
        type=str,
        default=get_default_config_path("dynamic_constraints.json"),
        help="输出的动态约束配置文件路径（默认: config/dynamic_constraints.json）"
    )
    
    args = parser.parse_args()
    
    # 验证数据库文件是否存在
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"错误: 数据库文件不存在: {db_path}")
        sys.exit(1)
    
    # 验证Megatron脚本文件是否存在
    script_path = Path(args.script_path)
    if not script_path.exists():
        print(f"错误: Megatron脚本文件不存在: {script_path}")
        sys.exit(1)
    
    # 从Megatron脚本解析训练配置
    try:
        config = parse_megatron_script(str(script_path))
        print(f" 成功解析Megatron脚本: {script_path}")
    except Exception as e:
        print(f"错误: 解析Megatron脚本失败: {e}")
        sys.exit(1)
    
    # 如果启用脚本配置注入，生成动态约束配置
    dynamic_constraints_path = None
    if args.inject_script_config:
        print(f"\n🔧 启用脚本配置注入模式")
        try:
            injector = ScriptConfigInjector()
            script_config = injector.inject_script_config_to_constraints(
                str(script_path),
                args.base_constraints, 
                args.output_constraints
            )
            dynamic_constraints_path = args.output_constraints
            print(f" 动态约束配置已生成: {dynamic_constraints_path}")
            print(f"   详细配置: TP={script_config.tp}, PP={script_config.pp}, DP={script_config.dp}")
            print(f"   分布式优化器: {script_config.use_distributed_optimizer}")
            print(f"   序列并行: {script_config.sequence_parallel}")
            print(f"   解绑嵌入层: {script_config.untie_embeddings}")
        except Exception as e:
            print(f"错误: 生成动态约束配置失败: {e}")
            sys.exit(1)
    
    print("=" * 80)
    print("SDCCheck - 分布式训练检查点一致性验证工具")
    print("=" * 80)
    print(f"数据库路径: {db_path}")
    print(f"脚本路径: {script_path}")
    print(f"训练配置: DP={config.dp}, TP={config.tp}, PP={config.pp}, EP={config.ep}, SP={config.sp}, ZeRO={config.zero_stage}")
    print("=" * 80)
    
    try:
        if args.manual:
            # 手动执行流程，逐步展示各个模块的工作
            run_manual_flow(config, str(db_path), args.verbose, dynamic_constraints_path)
        else:
            # 使用Orchestrator执行完整流程
            run_orchestrated_flow(config, str(db_path), dynamic_constraints_path)
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(130)
    except Exception as e:
        print(f"执行过程中出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_orchestrated_flow(config: TrainingConfig, db_path: str, 
                         dynamic_constraints_path: str = None):
    """
    使用预定义约束执行完整的检查流程
    
    参数:
        config: 训练配置
        db_path: 数据库路径
    """
    print("\n 启动预定义约束检查流程...")
    
    # 创建LLM提供者 - 仅用于SQL生成和结果分析
    llm_config = LLMConfig(
        model_name="gpt-4o",  # 使用OpenAI的GPT-3.5模型
        temperature=0.1,
        max_tokens=2000,
        base_url="https://api.openai.com/v1",  # OpenAI的API地址
        api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量读取OpenAI API Key
    )
    
    # 步骤1: 分析数据库模式
    print("\n🔍 步骤1: 分析数据库模式")
    schema_analyzer = SchemaAnalyzer()
    schema_info = schema_analyzer.analyze(db_path)
    print(f"    发现 {len(schema_info.tables)} 个表")
    
    # 步骤2: 使用预定义约束生成器选择约束
    print("\n⚙️ 步骤2: 根据训练配置选择预定义约束")
    
    # 如果有动态配置文件，使用动态配置初始化约束生成器
    if dynamic_constraints_path and Path(dynamic_constraints_path).exists():
        print(f"   🔧 使用动态约束配置: {dynamic_constraints_path}")
        constraint_generator = PredefinedConstraintGenerator(config_path=dynamic_constraints_path)
    else:
        print("    使用默认约束配置")
        constraint_generator = PredefinedConstraintGenerator()
    
    constraints = constraint_generator.generate(schema_info, config)
    print(f"    选择了 {len(constraints)} 个约束")
    
    if not constraints:
        print("    未找到适用的约束，请检查数据库结构或训练配置")
        return
    
    # 显示选择的约束
    for i, constraint in enumerate(constraints, 1):
        print(f"      {i}. {constraint.name} ({constraint.type})")
    
    # 步骤3: 执行约束检查
    print("\n 步骤3: 执行约束检查")
    sql_generator = LLMSQLGenerator(llm_config, "openai")
    db_executor = DatabaseExecutor()
    result_analyzer = LLMResultAnalyzer(llm_config, "openai")
    
    # 连接数据库
    db_executor.connect(db_path)
    reports = []
    
    for i, constraint in enumerate(constraints, 1):
        print(f"\n    检查约束 {i}/{len(constraints)}: {constraint.name}")
        
        try:
            # 生成SQL
            sql = sql_generator.generate(constraint)
            print("       SQL生成完成")
            
            # 打印SQL语句（用于调试）
            print(f"       SQL: {sql}")
            
            # 执行查询
            result = db_executor.execute(sql)
            print(f"       查询完成，返回 {len(result)} 行结果")
            
            # 分析结果
            report = result_analyzer.analyze(constraint, result)
            reports.append(report)
            
            # 显示结果
            status_emoji = {
                "pass": "",
                "fail": "", 
                "error": "",
                "warning": ""
            }.get(report.status, "❓")
            
            print(f"      {status_emoji} 结果: {report.status.upper()} - {report.summary}")
            
        except Exception as e:
            print(f"       检查失败: {e}")
            # 创建错误状态的report并添加到列表
            from sdccheck.models import AnalysisReport
            error_report = AnalysisReport(
                constraint=constraint,
                status="error",
                summary=f"执行失败: {str(e)[:100]}...",
                violations=[],
                suggestion="请检查SQL语法、数据类型或数据库连接"
            )
            reports.append(error_report)
    
    # 关闭数据库连
    db_executor.close()
    
    # 统计结果
    pass_count = sum(1 for r in reports if r.status == "pass")
    fail_count = sum(1 for r in reports if r.status == "fail")
    error_count = sum(1 for r in reports if r.status == "error")
    
    print(f"\n 检查结果统计:")
    print(f"    通过: {pass_count}")
    print(f"    失败: {fail_count}")
    print(f"    错误: {error_count}")
    print(f"    总计: {len(reports)}")
    
    # 如果有任何失败或错误的约束，返回非零退出码
    if fail_count > 0 or error_count > 0:
        print("\n 检查发现问题，请查看上述报告")
        sys.exit(3)
    else:
        print("\n 所有检查均通过")


def run_manual_flow(config: TrainingConfig, db_path: str, verbose: bool = False, 
                   dynamic_constraints_path: str = None):
    """
    手动执行流程，逐步展示各个模块的工作
    
    参数:
        config: 训练配置
        db_path: 数据库路径
        verbose: 是否输出详细信息
    """
    print("\n🔧 手动执行流程，逐步展示各个模块的工作...")
    
    # 设置统一的日志系统（包括LLM日志和应用日志）
    log_file_path = setup_unified_logging()
    print(f" 创建统一日志会话: {log_file_path}")
    
    # 获取logger实例
    logger = logging.getLogger("SDCCheck.ManualFlow")
    logger.info("开始手动执行流程")
    
    # 步骤1: 启动 - 创建各个模块实例
    print("\n 步骤1: 启动 - 初始化各个模块")
    
    # 创建LLM提供者 - 使用DeepSeek V3
    # llm_config = LLMConfig(
    #     model_name="deepseek-chat",
    #     temperature=0.1,
    #     max_tokens=2000,
    #     base_url="https://api.deepseek.com",
    #     api_key=os.getenv("DEEPSEEK_API_KEY")
    # )
    llm_config = LLMConfig(
        model_name="gpt-4o",  # 使用OpenAI的GPT-4o模型
        temperature=0.1,
        max_tokens=2000,
        base_url="https://api.openai.com/v1",  # OpenAI的API地址
        api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量读取OpenAI API Key
    )
    llm_provider = LLMProviderFactory.create_provider("openai", llm_config)
    
    schema_analyzer = SchemaAnalyzer()
    
    # 如果有动态配置文件，使用动态配置初始化约束生成器
    if dynamic_constraints_path and Path(dynamic_constraints_path).exists():
        print(f"   🔧 使用动态约束配置: {dynamic_constraints_path}")
        constraint_generator = PredefinedConstraintGenerator(config_path=dynamic_constraints_path)
    else:
        print("    使用默认约束配置")
        constraint_generator = PredefinedConstraintGenerator()
    
    sql_generator = LLMSQLGenerator(llm_config, "openai")
    db_executor = DatabaseExecutor()
    result_analyzer = LLMResultAnalyzer(llm_config, "openai")
    print("    所有LLM模块初始化完成（使用OpenAI）")
    
    # 步骤2: 了解数据 - 分析数据库模式
    print("\n🔍 步骤2: 了解数据 - 分析数据库模式")
    print(f"   正在分析数据库: {db_path}")
    logger.info(f"开始分析数据库: {db_path}")
    schema_info = schema_analyzer.analyze(db_path)
    print(f"    发现 {len(schema_info.tables)} 个表")
    logger.info(f"数据库分析完成，发现 {len(schema_info.tables)} 个表")
    
    if verbose:
        for table_name, table_info in schema_info.tables.items():
            columns = table_info.get('columns', [])
            json_keys = table_info.get('json_keys', {})
            print(f"       表 '{table_name}': {len(columns)} 列, {len(json_keys)} JSON字段")
    
    # 步骤3: 选择预定义约束
    print("\n⚙️ 步骤3: 选择预定义约束 - 根据数据库模式和训练配置选择适用的约束")
    constraints = constraint_generator.generate(schema_info, config)
    print(f"    选择了 {len(constraints)} 个约束")
    
    if not constraints:
        print("    未找到适用的约束，请检查数据库结构或训练配置")
        return
    
    if verbose:
        for i, constraint in enumerate(constraints, 1):
            print(f"      {i}. {constraint.name} ({constraint.type})")
    
    # 步骤4: 循环检查 - 连接数据库
    print("\n🔗 连接数据库")
    db_executor.connect(db_path)
    print("    数据库连接成功")
    
    # 步骤4: 循环检查每个约束
    print("\n 步骤4: 循环检查 - 对每个约束执行检查")
    reports = []
    
    for i, constraint in enumerate(constraints, 1):
        print(f"\n    检查约束 {i}/{len(constraints)}: {constraint.name}")
        
        # 步骤4a: 生成SQL
        print("      🔧 4a. 生成SQL查询")
        sql = sql_generator.generate(constraint)
        if verbose:
            print(f"          SQL: {sql}")
        print("          SQL生成完成")
        
        # 步骤4b: 执行查询
        print("       4b. 执行SQL查询")
        result = db_executor.execute(sql)
        print(f"          查询完成，返回 {len(result)} 行结果")
        
        # 步骤4c: 分析结果
        print("       4c. 分析查询结果")
        report = result_analyzer.analyze(constraint, result)
        reports.append(report)
        
        # 显示结果
        status_emoji = {
            "pass": "",
            "fail": "", 
            "error": "",
            "warning": ""
        }.get(report.status, "❓")
        
        print(f"         {status_emoji} 结果: {report.status.upper()} - {report.summary}")
        
        if report.violations and verbose:
            print(f"         违规数量: {len(report.violations)}")
    
    # 关闭数据库连接
    print("\n🔌 关闭数据库连接")
    db_executor.close()
    print("    数据库连接已关闭")
    
    # 显示最终报告
    print("\n" + "=" * 80)
    print(" 最终检查报告")
    print("=" * 80)
    
    for i, report in enumerate(reports, 1):
        status_emoji = {
            "pass": "",
            "fail": "",
            "error": "", 
            "warning": ""
        }.get(report.status, "❓")
        
        print(f"{i:2d}. {status_emoji} {report.constraint.name}")
        print(f"     状态: {report.status.upper()}")
        print(f"     摘要: {report.summary}")
        
        if report.violations:
            print(f"     违规: {len(report.violations)} 项")
            if verbose and report.violations:
                for j, violation in enumerate(report.violations[:2], 1):
                    print(f"       {j}. {violation}")
        
        if report.suggestion:
            print(f"     建议: {report.suggestion}")
        print()
    
    # 统计结果
    pass_count = sum(1 for r in reports if r.status == "pass")
    fail_count = sum(1 for r in reports if r.status == "fail")
    error_count = sum(1 for r in reports if r.status == "error")
    
    print(f" 检查结果统计:")
    print(f"    通过: {pass_count}")
    print(f"    失败: {fail_count}")
    print(f"    错误: {error_count}")
    print(f"    总计: {len(reports)}")
    
    # 设置退出码
    if fail_count > 0 or error_count > 0:
        print("\n 检查发现问题")
        sys.exit(3)
    else:
        print("\n 所有检查均通过")


if __name__ == "__main__":
    main()
