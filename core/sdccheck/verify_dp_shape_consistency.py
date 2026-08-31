#!/usr/bin/env python3
"""
验证 DP 参数 Shape 一致性约束

用途：检查 model-after-backward 阶段不同 DP rank 的参数 shape 是否一致
约束：backward后所有data parallel组内参数shape一致性检查

Author: SDCCheck Team
Date: 2025-01-16
"""

import duckdb
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


def verify_dp_shape_consistency(
    db_dir: str, 
    step: int = 0, 
    stage: str = "model-after-backward"
) -> Tuple[bool, Dict]:
    """
    验证指定step和stage下，所有DP rank的参数shape是否一致
    
    Args:
        db_dir: VTimeline数据库目录
        step: 训练步数
        stage: 训练阶段
    
    Returns:
        Tuple[bool, Dict]: (是否一致, 详细报告)
    """
    
    # 查找所有coredump数据库文件
    db_files = list(Path(db_dir).glob("**/coredump_*.db"))
    
    if not db_files:
        print(f"❌ 错误：在 {db_dir} 中未找到数据库文件")
        return False, {'error': 'No database files found'}
    
    print(f"✓ 找到 {len(db_files)} 个数据库文件\n")
    
    # 收集所有rank的参数信息
    all_ranks_data = {}
    
    for db_file in sorted(db_files):
        # 从文件名提取rank信息
        # 文件名格式：coredump_tp_pp_cp_dp.db
        parts = db_file.stem.split('_')
        if len(parts) >= 5:
            tp_rank = int(parts[1])
            pp_rank = int(parts[2])
            cp_rank = int(parts[3])
            dp_rank = int(parts[4])
        else:
            print(f"⚠ 警告：无法解析文件名 {db_file.name}")
            continue
        
        try:
            conn = duckdb.connect(str(db_file), read_only=True)
            
            # 查询该rank在指定step和stage的所有参数
            query = f"""
                SELECT 
                    step,
                    stage,
                    module_name,
                    param_name,
                    shape,
                    dtype,
                    requires_grad,
                    checksum
                FROM coredump
                WHERE step = {step}
                  AND stage = '{stage}'
                  AND requires_grad = true
                ORDER BY module_name, param_name
            """
            
            result = conn.execute(query).fetchall()
            
            if not result:
                print(f"⚠ 警告：DP rank {dp_rank} 在 step={step}, stage={stage} 没有数据")
                conn.close()
                continue
            
            # 存储该rank的参数信息
            rank_params = {}
            for row in result:
                step_val, stage_val, module_name, param_name, shape, dtype, req_grad, checksum = row
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                rank_params[full_name] = {
                    'shape': shape,
                    'dtype': dtype,
                    'requires_grad': req_grad,
                    'checksum': checksum
                }
            
            all_ranks_data[dp_rank] = {
                'params': rank_params,
                'file': db_file.name,
                'tp': tp_rank,
                'pp': pp_rank,
                'cp': cp_rank
            }
            print(f"  DP rank {dp_rank}: {len(rank_params)} 个参数 ({db_file.name})")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ 读取 {db_file} 失败: {e}")
            continue
    
    if len(all_ranks_data) < 2:
        error_msg = f"需要至少2个DP rank的数据进行对比（当前只有 {len(all_ranks_data)} 个）"
        print(f"\n❌ 错误：{error_msg}")
        return False, {'error': error_msg}
    
    # 验证shape一致性
    print(f"\n{'='*80}")
    print(f"验证约束：backward后所有data parallel组内参数shape一致性检查")
    print(f"Step: {step}, Stage: {stage}, DP ranks: {sorted(all_ranks_data.keys())}")
    print(f"{'='*80}\n")
    
    # 获取所有参数名称（取第一个rank作为基准）
    base_rank = min(all_ranks_data.keys())
    base_params = all_ranks_data[base_rank]['params']
    
    shape_mismatch_details = []
    value_mismatch_details = []
    consistent_count = 0
    
    for param_name in sorted(base_params.keys()):
        base_shape = base_params[param_name]['shape']
        base_checksum = base_params[param_name]['checksum']
        
        # 检查其他rank是否有相同shape
        shapes_by_rank = {base_rank: base_shape}
        checksums_by_rank = {base_rank: base_checksum}
        
        shape_mismatch = False
        value_mismatch = False
        
        for rank in sorted(all_ranks_data.keys()):
            if rank == base_rank:
                continue
            
            if param_name not in all_ranks_data[rank]['params']:
                shape_mismatch = True
                shapes_by_rank[rank] = '<缺失>'
                break
            
            rank_shape = all_ranks_data[rank]['params'][param_name]['shape']
            rank_checksum = all_ranks_data[rank]['params'][param_name]['checksum']
            shapes_by_rank[rank] = rank_shape
            checksums_by_rank[rank] = rank_checksum
            
            if rank_shape != base_shape:
                shape_mismatch = True
            
            if rank_checksum != base_checksum and rank_shape == base_shape:
                value_mismatch = True
        
        if shape_mismatch:
            shape_mismatch_details.append({
                'param': param_name,
                'shapes': shapes_by_rank,
                'checksums': checksums_by_rank
            })
        elif value_mismatch:
            value_mismatch_details.append({
                'param': param_name,
                'shapes': shapes_by_rank,
                'checksums': checksums_by_rank
            })
        else:
            consistent_count += 1
    
    # 输出结果
    total_params = len(base_params)
    
    # 构建报告
    report = {
        'constraint': 'backward后所有data parallel组内参数shape一致性检查',
        'step': step,
        'stage': stage,
        'dp_ranks': sorted(all_ranks_data.keys()),
        'total_params': total_params,
        'consistent_params': consistent_count,
        'shape_inconsistent_params': len(shape_mismatch_details),
        'value_inconsistent_params': len(value_mismatch_details),
        'shape_mismatch_details': shape_mismatch_details,
        'value_mismatch_details': value_mismatch_details
    }
    
    # 显示shape不一致的参数
    if shape_mismatch_details:
        print(f"❌ 发现 {len(shape_mismatch_details)} 个参数的shape不一致：\n")
        for idx, item in enumerate(shape_mismatch_details[:10], 1):  # 只显示前10个
            print(f"[{idx}] 📦 参数: {item['param']}")
            print(f"    Shape对比:")
            for rank in sorted(item['shapes'].keys()):
                shape = item['shapes'][rank]
                checksum = item['checksums'].get(rank, 'N/A')
                marker = "🔴" if rank == base_rank else "🟢"
                if isinstance(checksum, str) and checksum != 'N/A':
                    checksum_str = f"{checksum[:16]}..."
                else:
                    checksum_str = str(checksum)
                print(f"      {marker} DP rank {rank}: {shape} (checksum: {checksum_str})")
            print()
        
        if len(shape_mismatch_details) > 10:
            print(f"    ... 还有 {len(shape_mismatch_details) - 10} 个不一致的参数\n")
        
        print(f"{'='*80}")
        print(f"🔴 约束验证结果: FAILED")
        print(f"   一致参数: {consistent_count}/{total_params} ({consistent_count/total_params*100:.1f}%)")
        print(f"   Shape不一致参数: {len(shape_mismatch_details)}/{total_params} ({len(shape_mismatch_details)/total_params*100:.1f}%)")
        if value_mismatch_details:
            print(f"   值不一致参数: {len(value_mismatch_details)}/{total_params} ({len(value_mismatch_details)/total_params*100:.1f}%)")
        print(f"   违反约束: backward后所有data parallel组内参数shape一致性检查")
        print(f"{'='*80}\n")
        
        # 生成详细报告
        report_path = Path(db_dir) / "shape_inconsistency_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 详细报告已保存到: {report_path}\n")
        
        return False, report
    
    # 显示值不一致的参数（shape一致但值不同）
    elif value_mismatch_details:
        print(f"⚠️  发现 {len(value_mismatch_details)} 个参数的值不一致（shape一致）：\n")
        for idx, item in enumerate(value_mismatch_details[:5], 1):
            print(f"[{idx}] 📦 参数: {item['param']}")
            print(f"    Shape: {item['shapes'][base_rank]} (所有rank相同)")
            print(f"    Checksum对比:")
            for rank in sorted(item['checksums'].keys()):
                checksum = item['checksums'][rank]
                marker = "🔴" if checksum != item['checksums'][base_rank] else "🟢"
                print(f"      {marker} DP rank {rank}: {checksum[:16]}...")
            print()
        
        print(f"{'='*80}")
        print(f"✅ Shape一致性约束验证结果: PASSED")
        print(f"⚠️  但发现参数值不一致（这可能是其他类型的错误注入）")
        print(f"   Shape一致参数: {total_params}/{total_params}")
        print(f"   值不一致参数: {len(value_mismatch_details)}/{total_params}")
        print(f"{'='*80}\n")
        
        return True, report
    
    else:
        print(f"✅ 所有 {total_params} 个参数的shape在所有DP rank间一致\n")
        
        # 检查值是否也一致
        all_values_consistent = all(
            all(
                all_ranks_data[rank]['params'].get(param_name, {}).get('checksum') == 
                all_ranks_data[base_rank]['params'][param_name]['checksum']
                for rank in all_ranks_data.keys()
            )
            for param_name in base_params.keys()
        )
        
        print(f"{'='*80}")
        print(f"✅ 约束验证结果: PASSED")
        print(f"   一致参数: {consistent_count}/{total_params} (100%)")
        print(f"   满足约束: backward后所有data parallel组内参数shape一致性检查")
        if all_values_consistent:
            print(f"   额外信息: 参数值也完全一致（无任何错误注入）")
        print(f"{'='*80}\n")
        
        return True, report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="验证DP参数shape一致性约束",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 验证shape一致性
  python verify_dp_shape_consistency.py --db-dir /path/to/db --step 0
  
  # 指定不同的stage
  python verify_dp_shape_consistency.py --db-dir /path/to/db --step 2 --stage model-after-optimizer-step
  
  # 同时输出JSON报告
  python verify_dp_shape_consistency.py --db-dir /path/to/db --step 0 --output report.json
        """
    )
    parser.add_argument("--db-dir", required=True, help="VTimeline数据库目录")
    parser.add_argument("--step", type=int, default=0, help="训练步数 (默认: 0)")
    parser.add_argument("--stage", default="model-after-backward", help="训练阶段 (默认: model-after-backward)")
    parser.add_argument("--output", help="输出JSON报告路径（可选）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 验证数据库目录存在
    if not Path(args.db_dir).exists():
        print(f"❌ 错误：数据库目录不存在: {args.db_dir}")
        sys.exit(1)
    
    # 运行验证
    result, report = verify_dp_shape_consistency(args.db_dir, args.step, args.stage)
    
    # 保存报告到指定路径
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 报告已保存到: {output_path}")
    
    # 退出码：0表示通过，1表示失败
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()













