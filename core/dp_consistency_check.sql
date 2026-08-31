-- 检查同一name下，只有dp不同的行之间cksum和grad_cksum的不一致情况
WITH extracted_data AS (
  SELECT 
    json_extract(data, '$.name') AS name,
    json_extract(data, '$.cksum') AS cksum,
    json_extract(data, '$.grad_cksum') AS grad_cksum,
    json_extract(data, '$.shape') AS shape,
    json_extract(data, '$.type') AS type,
    json_extract(data, '$.dp') AS dp,
    json_extract(data, '$.tp') AS tp,
    stage,
    step
  FROM coredump
  WHERE stage = 'model-after-optimizer-step' AND step = 1
),

-- 为每个name+shape+type+tp+stage+step组合计算统计信息
name_groups AS (
  SELECT 
    name,
    shape,
    type,
    tp,
    stage,
    step,
    COUNT(DISTINCT dp) as dp_count,
    COUNT(DISTINCT cksum) as cksum_count,
    COUNT(DISTINCT grad_cksum) as grad_cksum_count,
    COUNT(*) as total_rows,
    STRING_AGG(DISTINCT dp::TEXT ORDER BY dp::TEXT) as dp_values,
    STRING_AGG(DISTINCT cksum ORDER BY cksum) as cksum_values,
    STRING_AGG(DISTINCT grad_cksum ORDER BY grad_cksum) as grad_cksum_values
  FROM extracted_data
  GROUP BY name, shape, type, tp, stage, step
),

-- 找出只有dp不同但cksum或grad_cksum不一致的情况
inconsistent_groups AS (
  SELECT 
    name,
    shape,
    type,
    tp,
    dp_count,
    cksum_count,
    grad_cksum_count,
    total_rows,
    dp_values,
    cksum_values,
    grad_cksum_values,
    CASE 
      WHEN dp_count > 1 AND cksum_count > 1 THEN 'cksum_inconsistent'
      WHEN dp_count > 1 AND grad_cksum_count > 1 THEN 'grad_cksum_inconsistent'
      WHEN dp_count > 1 AND cksum_count > 1 AND grad_cksum_count > 1 THEN 'both_inconsistent'
      ELSE 'consistent'
    END as inconsistency_type
  FROM name_groups
  WHERE dp_count > 1  -- 只关注有多个dp值的组
)

-- 输出不一致的情况
SELECT 
  name,
  shape,
  type,
  tp,
  inconsistency_type,
  dp_count,
  cksum_count,
  grad_cksum_count,
  total_rows,
  dp_values,
  cksum_values,
  grad_cksum_values
FROM inconsistent_groups
WHERE inconsistency_type != 'consistent'
ORDER BY name, shape, type, tp;

-- 同时输出统计摘要
-- SELECT 
--   inconsistency_type,
--   COUNT(*) as group_count
-- FROM inconsistent_groups
-- GROUP BY inconsistency_type
-- ORDER BY inconsistency_type;