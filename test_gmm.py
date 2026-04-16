import time
import traceback
from typing import List, Tuple

import torch

try:
    import torch_npu
except Exception as e:
    raise RuntimeError(f"torch_npu import failed: {e}")


# ============================================================
# 1) 直接复用/等价实现你第二个文件里的 GmmFunction
# ============================================================
class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_list, *all_params):
        """
        x: [M, K]
        group_list: [E]，每个 expert 处理的 token 数
        all_params: E 个权重，每个 [K, N]
        """
        E = len(all_params)
        ctx.E = E
        ctx.save_for_backward(x, *all_params)
        ctx.group_list = group_list

        outputs = torch_npu.npu_grouped_matmul(
            [x],
            list(all_params),
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_tensor = saved[0]
        weights = saved[1:]
        group_list = ctx.group_list
        E = ctx.E

        # grad_input: [M, N] x grouped [N, K] -> [M, K]
        weights_t = [w.transpose(0, 1).contiguous() for w in weights]
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            weights_t,
            bias=None,
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )[0]

        # grad_weight: grouped over expert blocks
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T.contiguous()],
            [grad_output],
            bias=None,
            group_list=group_list,
            group_type=2,
            split_item=3,
            group_list_type=1,
        )[0]

        experts = torch.chunk(grad_weight, E, dim=0)
        grad_weight_list = [expert.squeeze(0).contiguous() for expert in experts]

        return (grad_input, None, *grad_weight_list)


def gmmfunction_wrapper(x: torch.Tensor, weight_ekn: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    给 GmmFunction 的包装函数。
    对齐 transformers 的 _grouped_mm 语义：

      input:  [M, K]
      weight: [E, K, N]
      offsets:[E] cumulative ends

    转成：
      group_list: [E]
      weights: E 个 [K, N]
    """
    assert x.dim() == 2, f"x must be [M,K], got {x.shape}"
    assert weight_ekn.dim() == 3, f"weight must be [E,K,N], got {weight_ekn.shape}"
    assert offsets.dim() == 1, f"offsets must be [E], got {offsets.shape}"

    counts = torch.empty_like(offsets)
    counts[0] = offsets[0]
    if offsets.numel() > 1:
        counts[1:] = offsets[1:] - offsets[:-1]
    counts = counts.to(torch.int64)

    weights = [weight_ekn[i].contiguous() for i in range(weight_ekn.size(0))]
    return GmmFunction.apply(x, counts, *weights)


# ============================================================
# 2) transformers _grouped_mm 的“参考语义实现”
#    不依赖 torch.grouped_mm，直接用 for-loop + matmul，方便核对数值
# ============================================================
def ref_grouped_mm(x: torch.Tensor, weight_ekn: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    语义对齐 _grouped_mm(input, weight, offs)
    x:         [M, K]
    weight:    [E, K, N]
    offsets:   [E]
    out:       [M, N]
    """
    M, K = x.shape
    E, K2, N = weight_ekn.shape
    assert K == K2
    assert offsets.numel() == E

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    start = 0
    for i, end in enumerate(offsets.tolist()):
        if end > start:
            out[start:end] = x[start:end] @ weight_ekn[i]
        start = end
    return out


# ============================================================
# 3) 构造固定随机输入
# ============================================================
def set_seed(seed: int = 20260416):
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(seed)


def sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def make_counts(total_tokens: int, num_experts: int, device) -> torch.Tensor:
    """
    随机分配每个 expert 的 token 数，允许有 0。
    """
    if num_experts == 1:
        return torch.tensor([total_tokens], device=device, dtype=torch.int64)

    cuts = torch.randint(
        low=0,
        high=total_tokens + 1,
        size=(num_experts - 1,),
        device=device,
        dtype=torch.int64,
    )
    cuts, _ = torch.sort(cuts)
    cuts = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int64),
        cuts,
        torch.tensor([total_tokens], device=device, dtype=torch.int64),
    ])
    counts = cuts[1:] - cuts[:-1]
    return counts


def clone_weight_list_grads(weight_ekn: torch.Tensor) -> List[torch.Tensor]:
    return [weight_ekn[i].grad.detach().clone() for i in range(weight_ekn.size(0))]


def merge_grad_list_to_tensor(grad_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(grad_list, dim=0)


def error_stats(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    a64 = a.detach().to(torch.float64)
    b64 = b.detach().to(torch.float64)
    abs_diff = (a64 - b64).abs()
    rel_diff = abs_diff / torch.maximum(b64.abs(), torch.full_like(b64, eps))
    return {
        "max_abs": abs_diff.max().item() if abs_diff.numel() else 0.0,
        "mean_abs": abs_diff.mean().item() if abs_diff.numel() else 0.0,
        "max_rel": rel_diff.max().item() if rel_diff.numel() else 0.0,
        "mean_rel": rel_diff.mean().item() if rel_diff.numel() else 0.0,
    }


def print_stats(title: str, stats: dict):
    print(f"\n[{title}]")
    for k, v in stats.items():
        print(f"  {k:>10}: {v:.10e}")


# ============================================================
# 4) 性能计时
# ============================================================
def benchmark_forward(fn, warmup=10, iters=50):
    for _ in range(warmup):
        y = fn()
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def benchmark_fwd_bwd(fn, warmup=10, iters=30):
    for _ in range(warmup):
        loss = fn()
        loss.backward()
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        loss = fn()
        loss.backward()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ============================================================
# 5) 主测试
# ============================================================
def main():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("当前环境没有可用的 NPU，请在 Ascend + torch_npu 环境运行")

    device = torch.device("npu")
    set_seed(20260416)

    # -----------------------------
    # 你可以改这几个规模
    # -----------------------------
    M = 4096          # 总 token 数（已经是按 expert 排好序后的总数）
    K = 1024          # 输入维度
    N = 2048          # 输出维度
    E = 8             # expert 数
    DTYPE = torch.bfloat16  # 可改 torch.float16 / torch.float32 / torch.bfloat16

    print("==== Config ====")
    print(f"device={device}")
    print(f"M={M}, K={K}, N={N}, E={E}, dtype={DTYPE}")

    # counts / offsets
    counts = make_counts(M, E, device)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int64)

    print(f"counts={counts.tolist()}")
    print(f"offsets={offsets.tolist()}")

    # 构造输入
    x_ref = torch.randn(M, K, device=device, dtype=DTYPE, requires_grad=True)
    w_ref = torch.randn(E, K, N, device=device, dtype=DTYPE, requires_grad=True)

    x_gmm = x_ref.detach().clone().requires_grad_(True)
    w_gmm = w_ref.detach().clone().requires_grad_(True)

    # -----------------------------
    # Forward 精度
    # -----------------------------
    y_ref = ref_grouped_mm(x_ref, w_ref, offsets)
    y_gmm = gmmfunction_wrapper(x_gmm, w_gmm, offsets)
    sync()

    fwd_stats = error_stats(y_gmm, y_ref)
    print_stats("forward output diff", fwd_stats)

    # -----------------------------
    # Backward 精度
    # -----------------------------
    grad_out = torch.randn_like(y_ref)

    y_ref.backward(grad_out)
    y_gmm.backward(grad_out)
    sync()

    xg_stats = error_stats(x_gmm.grad, x_ref.grad)
    wg_stats = error_stats(w_gmm.grad, w_ref.grad)

    print_stats("grad x diff", xg_stats)
    print_stats("grad weight diff", wg_stats)

    # -----------------------------
    # 性能测试：forward
    # -----------------------------
    x_ref_b = torch.randn(M, K, device=device, dtype=DTYPE)
    w_ref_b = torch.randn(E, K, N, device=device, dtype=DTYPE)

    def fn_ref_fwd():
        return ref_grouped_mm(x_ref_b, w_ref_b, offsets)

    def fn_gmm_fwd():
        return gmmfunction_wrapper(x_ref_b, w_ref_b, offsets)

    t_ref_fwd = benchmark_forward(fn_ref_fwd, warmup=10, iters=50)
    t_gmm_fwd = benchmark_forward(fn_gmm_fwd, warmup=10, iters=50)

    print("\n[forward performance]")
    print(f"  ref_grouped_mm     : {t_ref_fwd * 1000:.3f} ms/iter")
    print(f"  gmmfunction_wrapper: {t_gmm_fwd * 1000:.3f} ms/iter")
    print(f"  speedup            : {t_ref_fwd / t_gmm_fwd:.3f} x")

    # -----------------------------
    # 性能测试：forward + backward
    # -----------------------------
    x_ref_fb = torch.randn(M, K, device=device, dtype=DTYPE, requires_grad=True)
    w_ref_fb = torch.randn(E, K, N, device=device, dtype=DTYPE, requires_grad=True)
    grad_fb = torch.randn(M, N, device=device, dtype=DTYPE)

    def fn_ref_fwbw():
        if x_ref_fb.grad is not None:
            x_ref_fb.grad.zero_()
        if w_ref_fb.grad is not None:
            w_ref_fb.grad.zero_()
        y = ref_grouped_mm(x_ref_fb, w_ref_fb, offsets)
        return (y * grad_fb).sum()

    x_gmm_fb = x_ref_fb.detach().clone().requires_grad_(True)
    w_gmm_fb = w_ref_fb.detach().clone().requires_grad_(True)

    def fn_gmm_fwbw():
        if x_gmm_fb.grad is not None:
            x_gmm_fb.grad.zero_()
        if w_gmm_fb.grad is not None:
            w_gmm_fb.grad.zero_()
        y = gmmfunction_wrapper(x_gmm_fb, w_gmm_fb, offsets)
        return (y * grad_fb).sum()

    t_ref_fwbw = benchmark_fwd_bwd(fn_ref_fwbw, warmup=5, iters=20)
    t_gmm_fwbw = benchmark_fwd_bwd(fn_gmm_fwbw, warmup=5, iters=20)

    print("\n[fwd+bwd performance]")
    print(f"  ref_grouped_mm     : {t_ref_fwbw * 1000:.3f} ms/iter")
    print(f"  gmmfunction_wrapper: {t_gmm_fwbw * 1000:.3f} ms/iter")
    print(f"  speedup            : {t_ref_fwbw / t_gmm_fwbw:.3f} x")

    # -----------------------------
    # 最终建议提示
    # -----------------------------
    print("\n==== Quick Heuristic ====")
    print("如果满足下面两个条件，通常可以认为 `_grouped_mm -> GmmFunction` 具备替换可行性：")
    print("1) forward / grad x / grad weight 的 max_abs 和 mean_abs 都足够小")
    print("2) gmmfunction_wrapper 的性能优于 ref_grouped_mm")
    print("\n把你跑出来的结果贴给我，我来帮你判断“能否替换”。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
