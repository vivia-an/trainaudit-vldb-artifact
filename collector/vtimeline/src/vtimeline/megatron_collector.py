#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib
import threading
import queue as _queue_mod
import atexit


class _LazyVal:
    """异步模式下延迟 GPU→CPU 的占位:训练线程只持 GPU 标量,worker 线程才 .item()。
    hexfmt=True 表示 cksum(需 hex 格式化);否则是浮点统计量。"""
    __slots__ = ("t", "hexfmt")
    def __init__(self, t, hexfmt=False):
        self.t = t
        self.hexfmt = hexfmt


def _get_cksum(data: torch.Tensor):
    import os as _os
    if _os.getenv("VTIMELINE_FAST") == "1":
        try:
            _u = data.detach().contiguous().view(torch.uint8).to(torch.int64).flatten()
            _W = 257; _n = _u.numel(); _pad = (-_n) % _W
            if _pad: _u = torch.cat([_u, _u.new_zeros(_pad)])
            _wt = torch.arange(1, _W + 1, device=_u.device, dtype=torch.int64)
            _h = ((_u.view(-1, _W) * _wt).sum(0) * _wt).sum()
            if _os.getenv("VTIMELINE_ASYNC") == "1" or _os.getenv("VTIMELINE_BATCH") == "1":
                return _LazyVal(_h.detach(), hexfmt=True)
            return format(int(_h.item()) & 0xFFFFFFFFFFFFFFFF, "x")
        except Exception:
            pass
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)

    return hasher.hexdigest()


def _fast_param_stats(flat_param):
    out={}; n=flat_param.numel()
    s1=flat_param.sum(); s2=(flat_param*flat_param).sum(); s3=(flat_param**3).sum(); s4=(flat_param**4).sum()
    mn=flat_param.min(); mx=flat_param.max()
    q=torch.quantile(flat_param, torch.tensor([0.25,0.5,0.75],device=flat_param.device))
    if os.getenv("VTIMELINE_ASYNC") == "1" or os.getenv("VTIMELINE_BATCH") == "1":
        # 异步/批处理:全留 GPU,延迟 .item();跳过 hist/entropy(辅助分布形态项)
        out["min"]=_LazyVal(mn); out["max"]=_LazyVal(mx)
        out["quantile_25"]=_LazyVal(q[0]); out["quantile_50"]=_LazyVal(q[1]); out["quantile_75"]=_LazyVal(q[2])
        if n>2:
            m=s1/n; var=(s2/n-m*m).clamp_min(1e-20); sd=var**0.5
            out["skewness"]=_LazyVal((s3/n-3*m*s2/n+2*m**3)/sd**3)
            out["kurtosis"]=_LazyVal((s4/n-4*m*s3/n+6*m*m*s2/n-3*m**4)/sd**4-3.0)
        return out
    vals=torch.stack([mn,mx,q[0],q[1],q[2],s1,s2,s3,s4]).double().cpu().tolist()
    mn_,mx_,q25,q50,q75,S1,S2,S3,S4=vals
    out["min"]=mn_; out["max"]=mx_; out["quantile_25"]=q25; out["quantile_50"]=q50; out["quantile_75"]=q75
    if n>2:
        m=S1/n; var=S2/n-m*m
        if var>1e-20:
            sd=var**0.5
            out["skewness"]=(S3/n-3*m*S2/n+2*m**3)/sd**3
            out["kurtosis"]=(S4/n-4*m*S3/n+6*m*m*S2/n-3*m**4)/sd**4-3.0
    if n>10 and mx_>mn_:
        hist=torch.histc(flat_param,bins=64,min=float(mn_),max=float(mx_)); hn=hist/hist.sum()
        out["histogram_cksum"]=hash(hn.cpu().numpy().tobytes())&0xFFFFFFFF
        nz=hn[hn>1e-10]
        if len(nz)>0: out["entropy"]=float((-(nz*nz.log()).sum()).item())
    return out



class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))
    # 每个进程仅 dump 一个 batch 的一次性开关
    dumped_training_batch_once_: bool = False

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_"), "the rank information must be set"

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_dp{}_tp{}_pp{}_cp{}.db".format(
                cls.ranks_info_["dp"], cls.ranks_info_["tp"],cls.ranks_info_["pp"],cls.ranks_info_["cp"]
            ),
        )
        cls.db_ = duckdb.connect(db_path)

        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def set_process_group_info(cls, ranks_info):
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls.initialize()

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    # ---------------- 异步/批处理写库(env-guard VTIMELINE_ASYNC / VTIMELINE_BATCH) ----------------
    _async_q = None
    _async_worker = None
    _batch_buf = []

    @classmethod
    def _ensure_worker(cls):
        if cls._async_q is not None:
            return
        cls._async_q = _queue_mod.Queue(maxsize=30000)  # 背压上限,防 GPU 标量堆积

        def _run():
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(torch.cuda.current_device())
            except Exception:
                pass
            while True:
                item = cls._async_q.get()
                if item is None:
                    cls._async_q.task_done()
                    break
                step, stage, info = item
                try:
                    for k, v in list(info.items()):
                        if isinstance(v, _LazyVal):
                            x = v.t.item()
                            info[k] = format(int(x) & 0xFFFFFFFFFFFFFFFF, "x") if v.hexfmt else float(x)
                        elif isinstance(v, torch.Tensor):
                            info[k] = v.item() if v.ndim == 0 else v.tolist()
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (step, stage, json.dumps(info)),
                    )
                except Exception as e:
                    print(f"[vtimeline async] insert error: {e}")
                finally:
                    cls._async_q.task_done()

        cls._async_worker = threading.Thread(target=_run, name="vtimeline-writer", daemon=True)
        cls._async_worker.start()
        atexit.register(cls._async_flush)

    @classmethod
    def _async_flush(cls):
        if cls._async_q is not None:
            try:
                cls._async_q.join()
            except Exception:
                pass

    @classmethod
    def _flush_batch(cls):
        """批处理:把本步缓冲的所有 dict 的 _LazyVal 用 2 次批量 .cpu()(int64/float 各一)解析,
        再 executemany 一次性插入。把每步 13106 次 GPU 同步压成 ~2 次。"""
        buf = cls._batch_buf
        if not buf:
            return
        cls._batch_buf = []
        hex_t = []; hex_slot = []; f_t = []; f_slot = []
        for _, _, info in buf:
            for k, v in info.items():
                if isinstance(v, _LazyVal):
                    (hex_t if v.hexfmt else f_t).append(v.t.reshape(()))
                    (hex_slot if v.hexfmt else f_slot).append((info, k))
        try:
            if hex_t:
                for (d, k), x in zip(hex_slot, torch.stack([t.long() for t in hex_t]).cpu().tolist()):
                    d[k] = format(int(x) & 0xFFFFFFFFFFFFFFFF, "x")
            if f_t:
                for (d, k), x in zip(f_slot, torch.stack([t.float() for t in f_t]).cpu().tolist()):
                    d[k] = float(x)
        except Exception as e:
            print(f"[vtimeline batch] resolve error: {e}")
        try:
            cls.db_.executemany(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                [(s, st, json.dumps(d)) for s, st, d in buf],
            )
        except Exception as e:
            print(f"[vtimeline batch] insert error: {e}")

    @classmethod
    def _emit(cls, stage_name, info):
        """统一出口:BATCH 缓冲到步末批量解析;ASYNC 入队 worker;否则原同步。"""
        if os.getenv("VTIMELINE_BATCH") == "1":
            cls._batch_buf.append((cls.step_, stage_name, info))
            return
        if os.getenv("VTIMELINE_ASYNC") == "1":
            cls._ensure_worker()
            cls._async_q.put((cls.step_, stage_name, info))
        else:
            try:
                cls.db_.execute(
                    "INSERT INTO coredump VALUES (?, ?, ?);",
                    (cls.step_, stage_name, json.dumps(info)),
                )
            except Exception as e:
                print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "main-grad-in-bwd"
    ):
        if not cls.should_dump():
            return

        param_info = {
            "name": param_name,
            "cksum": _get_cksum(param.main_grad),
            "shape": list(param.main_grad.shape),
            "type": str(param.main_grad.type()),
        }
        param_info.update(cls.ranks_info_)
        cls._emit(stage_name, param_info)

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = (
                    hasattr(param, "main_param") and param.main_param is not None
                )

                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param.main_param) if main_param_exist else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "type": str(param.main_param.type())
                    if hasattr(param, "main_param") and param.main_param is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                cls._emit(stage_name, param_info)

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                # 基础信息
                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param),
                    "shape": list(param.shape),
                    "type": str(param.type()),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": _get_cksum(param.grad)
                    if param.grad is not None
                    else None,
                    "grad_shape": list(param.grad.shape)
                    if param.grad is not None
                    else None,
                    "grad_type": str(param.grad.type())
                    if param.grad is not None
                    else None,
                }
                
                # 添加分片参数标识 (用于DP分片参数互异性检查)
                # Megatron-LM 中 param.allreduce=False 表示专家并行参数（分片参数）
                # param.allreduce=True 或不存在表示全复制参数
                is_expert_parallel = not getattr(param, 'allreduce', True)
                param_info["param_sharded"] = is_expert_parallel
                param_info["param_full_replica"] = not is_expert_parallel
                
                # 添加分布统计量：min, max, quantile (用于极值/分位数一致性检查)
                try:
                    import os as _os2
                    if _os2.getenv("VTIMELINE_FAST")=="1" and param.requires_grad and param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        param_info.update(_fast_param_stats(param.detach().float().flatten()))
                    elif param.requires_grad and param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        flat_param = param.detach().float().flatten()
                        param_info["min"] = float(flat_param.min().item())
                        param_info["max"] = float(flat_param.max().item())
                        # 计算分位数 (25%, 50%, 75%)
                        param_info["quantile_25"] = float(torch.quantile(flat_param, 0.25).item())
                        param_info["quantile_50"] = float(torch.quantile(flat_param, 0.50).item())
                        param_info["quantile_75"] = float(torch.quantile(flat_param, 0.75).item())
                        
                        # 添加高阶统计量：skewness (偏度) 和 kurtosis (峰度)
                        # 用于 model-before-optimizer-step阶段DP参数分布高阶统计量一致性检查
                        n = flat_param.numel()
                        if n > 2:  # 需要至少3个元素才能计算偏度和峰度
                            mean = flat_param.mean()
                            std = flat_param.std()
                            if std > 1e-10:  # 避免除零
                                # 偏度 (skewness): E[(X-μ)³] / σ³
                                skewness = ((flat_param - mean) ** 3).mean() / (std ** 3)
                                param_info["skewness"] = float(skewness.item())
                                
                                # 峰度 (kurtosis): E[(X-μ)⁴] / σ⁴ - 3 (Fisher's definition)
                                kurtosis = ((flat_param - mean) ** 4).mean() / (std ** 4) - 3.0
                                param_info["kurtosis"] = float(kurtosis.item())
                        
                        # 添加分布形态统计量：histogram 和 entropy
                        # 用于 DP参数分布形态一致性检查
                        if n > 10:  # 需要足够的元素才能计算有意义的直方图
                            # 计算直方图 (使用 64 个 bin，归一化后计算 checksum)
                            hist_bins = 64
                            hist_min = flat_param.min()
                            hist_max = flat_param.max()
                            if hist_max > hist_min:
                                hist = torch.histc(flat_param, bins=hist_bins, min=float(hist_min), max=float(hist_max))
                                hist_normalized = hist / hist.sum()  # 归一化为概率分布
                                
                                # histogram_cksum: 直方图的 checksum（用于检测分布形态差异）
                                hist_bytes = hist_normalized.cpu().numpy().tobytes()
                                param_info["histogram_cksum"] = hash(hist_bytes) & 0xFFFFFFFF  # 32-bit hash
                                
                                # 计算熵 (entropy): -sum(p * log(p))
                                # 过滤掉零概率以避免 log(0)
                                hist_nonzero = hist_normalized[hist_normalized > 1e-10]
                                if len(hist_nonzero) > 0:
                                    entropy = -torch.sum(hist_nonzero * torch.log(hist_nonzero))
                                    param_info["entropy"] = float(entropy.item())
                except Exception:
                    pass  # 统计量计算失败不影响主流程
                
                param_info.update(cls.ranks_info_)
                cls._emit(stage_name, param_info)


    @classmethod
    def dump_optimizer_state(cls, stage_name: str):
        """Dump optimizer state (momentum, variance, etc.) to database.
        
        This method dumps the internal state of the optimizer (e.g., exp_avg, exp_avg_sq
        for Adam/AdamW, momentum_buffer for SGD) for each parameter.
        
        Args:
            stage_name: Stage identifier (e.g., "optimizer-state-after-step")
        """
        if not cls.should_dump():
            return
        
        if not hasattr(cls, 'optimizer_') or cls.optimizer_ is None:
            return
        
        # Access the underlying PyTorch optimizer
        # MegatronOptimizer wraps a PyTorch optimizer in self.optimizer
        pytorch_optimizer = cls.optimizer_.optimizer if hasattr(cls.optimizer_, 'optimizer') else cls.optimizer_
        
        # Build a mapping from param to name for faster lookup
        param_to_name = {}
        if hasattr(cls, 'model_'):
            for model in cls.model_:
                for name, param in model.named_parameters():
                    # Handle both regular params and main_params
                    param_to_name[id(param)] = name
                    if hasattr(param, 'main_param') and param.main_param is not None:
                        param_to_name[id(param.main_param)] = name
        
        # Iterate through optimizer state
        for group_idx, group in enumerate(pytorch_optimizer.param_groups):
            # 获取该 param_group 的 lr（用于 optimizer_state_dict.lr 一致性检查）
            group_lr = group.get('lr', None)
            
            for param in group['params']:
                if param not in pytorch_optimizer.state:
                    continue
                
                state = pytorch_optimizer.state[param]
                
                # Get parameter name
                param_name = param_to_name.get(id(param), "unknown")
                
                # Dump each state field (exp_avg, exp_avg_sq, momentum_buffer, etc.)
                for state_key, state_value in state.items():
                    if isinstance(state_value, torch.Tensor):
                        state_info = {
                            "name": param_name,
                            "state_key": state_key,  # e.g., "exp_avg", "exp_avg_sq"
                            "cksum": _get_cksum(state_value),
                            "shape": list(state_value.shape),
                            "type": str(state_value.type()),
                            "lr": group_lr,  # 添加 lr 字段用于一致性检查
                            "param_group_idx": group_idx,  # 添加 group 索引
                        }
                        state_info.update(cls.ranks_info_)
                        cls._emit(stage_name, state_info)

    @classmethod
    def dump_training_batch(cls, tokens, labels, loss_mask, attention_mask, position_ids, stage_name: str = "after-get-batch"):
        """Dump key training batch info after get_batch.

        Make the JSON data format consistent with other dumps: one row per tensor with fields
        name, cksum, shape, type, plus rank info.
        """
        # 若已在本进程 dump 过一次训练 batch，则不再重复 dump
        if cls.dumped_training_batch_once_:
            return
        if not cls.should_dump():
            return

        items = [
            ("tokens", tokens),
            ("labels", labels),
            ("loss_mask", loss_mask),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ]

        for name, t in items:
            try:
                info = {
                    "name": name,
                    "cksum": _get_cksum(t) if t is not None else None,
                    "shape": list(t.shape) if t is not None else None,
                    "type": str(t.type()) if t is not None else None,
                }
                info.update(cls.ranks_info_)
                cls._emit(stage_name, info)
            except Exception as e:
                print(f"Error inserting batch data into coredump: {e}")
        # 标记本进程已完成一次训练 batch dump
        cls.dumped_training_batch_once_ = True

    @classmethod
    def dump_parallel_metadata(cls, stage_name: str, extra_meta: dict):
        """B-light only: topology/process-group rows. Does not alter existing dump_* paths."""
        if not hasattr(cls, "db_"):
            return
        if not cls.should_dump():
            return
        meta_info = {"type": "parallel_metadata", **(extra_meta or {})}
        if hasattr(cls, "ranks_info_"):
            meta_info.update(cls.ranks_info_)
        cls._emit(stage_name, meta_info)

    @classmethod
    def dump_rng_state(cls, stage_name: str):
        """B-light only: RNG fingerprint at checkpoint-save/load. Additive API."""
        if not hasattr(cls, "db_"):
            return
        if not cls.should_dump():
            return
        rng_info = {
            "type": "rng_state",
            "cpu_rng_state_cksum": _get_cksum(torch.get_rng_state()),
            "cuda_rng_state_cksum": (
                _get_cksum(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None
            ),
        }
        if hasattr(cls, "ranks_info_"):
            rng_info.update(cls.ranks_info_)
        cls._emit(stage_name, rng_info)

    @classmethod
    def step(cls):
        cls._flush_batch()
        cls.step_ += 1
