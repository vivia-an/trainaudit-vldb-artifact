#!/usr/bin/env python3
"""
验证 Data Parallel 组内 Optimizer State 一致性

约束描述:
在 model-after-backward 阶段，所有 data parallel 组内参数的 optimizer_state
（如 momentum/variance）必须一致。若发现不一致，表明存在同步异常或外部注入错误。

验证方法:
1. 从 coredump 表中读取 model-after-backward 阶段的数据
2. 对于每个参数，计算各 DP rank 的 optimizer_state 的 checksum
3. 比较同一参数在不同 DP rank 上的 checksum 是否一致
4. 报告不一致的参数

注意:
- 需要 VTimeline 收集 optimizer_state 数据（通过 dump_optimizer_state）
- 如果 coredump 中没有 optimizer_state 数据，本脚本会报告无法验证
"""

import duckdb
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 数据库路径
DB_PATH = Path(__file__).parent / "data" / "coredump.db"

def connect_db():
    """连接到 DuckDB 数据库"""
    if not DB_PATH.exists():
        print(f"❌ 错误: 数据库文件不存在: {DB_PATH}")
        print(f"   请先运行训练脚本收集数据")
        sys.exit(1)
    
    return duckdb.connect(str(DB_PATH), read_only=True)

def check_optimizer_state_data(conn):
    """检查是否有 optimizer_state 数据"""
    query = """
    SELECT COUNT(*) as count
    FROM coredump
    WHERE stage LIKE '%optimizer-state%'
    """
    result = conn.execute(query).fetchone()
    count = result[0] if result else 0
    
    if count == 0:
        print("⚠️  警告: 未找到 optimizer_state 数据")
        print("   可能的原因:")
        print("   1. VTimeline 未配置收集 optimizer_state")
        print("   2. 训练步数不足，optimizer state 尚未创建")
        print("   3. 使用的 optimizer 不产生 state（如 SGD without momentum）")
        print("   4. VTimeline 版本过旧，不支持 dump_optimizer_state 方法")
        return False
    
    print(f"✓ 找到 {count} 条 optimizer_state 记录")
    return True

def get_dp_config(conn):
    """获取 DP 配置"""
    query = """
    SELECT DISTINCT dp
    FROM coredump
    WHERE stage = 'model-after-backward'
    ORDER BY dp
    """
    dp_ranks = [row[0] for row in conn.execute(query).fetchall()]
    
    if not dp_ranks:
        print("❌ 错误: 未找到 DP rank 信息")
        return None
    
    if len(dp_ranks) <= 1:
        print(f"⚠️  警告: 只有 {len(dp_ranks)} 个 DP rank，无法验证一致性")
        print(f"   DP ranks: {dp_ranks}")
        return None
    
    print(f"✓ DP 配置: {len(dp_ranks)} 个 ranks: {dp_ranks}")
    return dp_ranks

def verify_optimizer_state_consistency(conn, stage="optimizer-state-after-injection"):
    """
    验证 optimizer state 一致性
    
    返回:
        (total_params, inconsistent_params, details)
    """
    print(f"\n{'='*60}")
    print(f"验证阶段: {stage}")
    print(f"{'='*60}\n")
    
    # 查询所有参数的 optimizer state checksum
    # 注意：数据存储在 JSON 字段中
    query = f"""
    SELECT 
        step,
        stage,
        data
    FROM coredump
    WHERE stage = '{stage}'
    ORDER BY step
    """
    
    results = conn.execute(query).fetchall()
    
    if not results:
        print(f"❌ 未找到 {stage} 阶段的 optimizer_state 数据")
        return 0, 0, []
    
    print(f"✓ 找到 {len(results)} 条记录")
    
    # 解析 JSON 数据并按参数分组
    param_groups = defaultdict(list)
    for row in results:
        step, stage_name, data_json = row
        data = json.loads(data_json)
        
        # 提取关键信息
        param_name = data.get('name', 'unknown')
        state_key = data.get('state_key', 'unknown')  # exp_avg, exp_avg_sq, etc.
        cksum = data.get('cksum')
        dp = data.get('dp')
        tp = data.get('tp', 0)
        pp = data.get('pp', 0)
        
        # 组合参数名和 state_key 作为唯一标识
        full_name = f"{param_name}[{state_key}]"
        key = (full_name, tp, pp, step)
        
        param_groups[key].append({
            'dp': dp,
            'cksum': cksum,
            'state_key': state_key,
            'param_name': param_name
        })
    
    print(f"✓ 共 {len(param_groups)} 个参数状态需要验证")
    
    # 检查一致性
    inconsistent_params = []
    total_params = len(param_groups)
    
    for (full_name, tp, pp, step), dp_records in param_groups.items():
        # 获取所有 DP rank 的 checksum
        checksums = {rec['dp']: rec['cksum'] for rec in dp_records}
        
        # 检查是否所有 checksum 都相同
        unique_checksums = set(checksums.values())
        
        if len(unique_checksums) > 1:
            # 提取参数名和 state_key
            param_name = dp_records[0]['param_name']
            state_key = dp_records[0]['state_key']
            
            inconsistent_params.append({
                'param_name': param_name,
                'state_key': state_key,
                'full_name': full_name,
                'tp': tp,
                'pp': pp,
                'step': step,
                'checksums': checksums,
                'unique_checksums': list(unique_checksums)
            })
    
    return total_params, len(inconsistent_params), inconsistent_params

def print_results(total_params, inconsistent_count, inconsistent_params):
    """打印验证结果"""
    print(f"\n{'='*60}")
    print("验证结果")
    print(f"{'='*60}\n")
    
    print(f"总参数数: {total_params}")
    print(f"不一致参数数: {inconsistent_count}")
    print(f"一致性比例: {(total_params - inconsistent_count) / total_params * 100:.2f}%")
    
    if inconsistent_count == 0:
        print("\n✅ 所有参数的 optimizer_state 在 DP 组内一致")
        return True
    else:
        print(f"\n❌ 发现 {inconsistent_count} 个参数状态不一致:")
        print()
        
        for i, param in enumerate(inconsistent_params[:10], 1):  # 只显示前10个
            print(f"{i}. {param['param_name']} [{param['state_key']}]")
            print(f"   位置: TP={param['tp']}, PP={param['pp']}, Step={param['step']}")
            print(f"   不同的 checksum 值: {len(param['unique_checksums'])} 种")
            
            for dp, cksum in sorted(param['checksums'].items()):
                print(f"     DP rank {dp}: {cksum[:16]}...")
            print()
        
        if inconsistent_count > 10:
            print(f"   ... 还有 {inconsistent_count - 10} 个不一致参数状态未显示")
        
        return False

def save_report(total_params, inconsistent_count, inconsistent_params, stage="optimizer-state-after-injection", output_file="optimizer_state_consistency_report.json"):
    """保存验证报告为 JSON"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'constraint': 'DP组内optimizer_state一致性',
        'stage': stage,
        'summary': {
            'total_params': total_params,
            'inconsistent_params': inconsistent_count,
            'consistency_rate': f"{(total_params - inconsistent_count) / total_params * 100:.2f}%"
        },
        'inconsistent_details': inconsistent_params
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ 详细报告已保存到: {output_path}")

def main():
    print("="*60)
    print("Data Parallel 组内 Optimizer State 一致性验证")
    print("="*60)
    print()
    
    # 连接数据库
    print("1. 连接数据库...")
    conn = connect_db()
    print(f"   ✓ 已连接: {DB_PATH}")
    
    # 检查 optimizer_state 数据
    print("\n2. 检查 optimizer_state 数据...")
    if not check_optimizer_state_data(conn):
        print("\n⚠️  无法验证: 缺少 optimizer_state 数据")
        print("\n建议:")
        print("1. 确保 VTimeline 配置了 dump_optimizer_state")
        print("2. 使用带 momentum 的 optimizer（如 Adam）")
        print("3. 训练至少 2 步以上")
        conn.close()
        return
    
    # 获取 DP 配置
    print("\n3. 获取 DP 配置...")
    dp_ranks = get_dp_config(conn)
    if dp_ranks is None:
        conn.close()
        return
    
    # 验证一致性
    print("\n4. 验证 optimizer_state 一致性...")
    stage = "optimizer-state-after-injection"
    total_params, inconsistent_count, inconsistent_params = verify_optimizer_state_consistency(conn, stage)
    
    # 打印结果
    success = print_results(total_params, inconsistent_count, inconsistent_params)
    
    # 保存报告
    if total_params > 0:
        save_report(total_params, inconsistent_count, inconsistent_params, stage)
    
    # 关闭连接
    conn.close()
    
    # 返回状态码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

