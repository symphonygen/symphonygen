"""Standalone speed & VRAM benchmark for the architectures in Table 1.

Compares the four factorizations of a (Bar, Track, Event) score grid:

    3D  [B,T,E] : bar stack over B,  track stack over T,  event stack over E
    2D  [BT,E]  : patch stack over B*T (NotaGen-style),   event stack over E
    2D  [B,TE]  : bar stack over B,   event stack over T*E per bar
    1D  [BTE]   : one flat stack over B*T*E
    3D packed   : 3D with pack_tracks/unpack_tracks mapping between the 2D
                  flattened (active tracks only) and 3D logical (padded grid)
                  layouts: position-wise ops run on packed tokens, attention
                  runs on the logical grouping

All variants use identical pre-LN causal transformer blocks (SDPA attention,
head_dim=64) and the SAME total number of layers / parameters; they differ
only in how sequences are grouped. Lower->upper features come from mean
pooling; upper->lower conditioning is broadcast-add (the cascade's cheap
glue, mirroring the paper's encoder/decoder coupling without extra stacks).

Inputs are random embeddings; no vocab/embedding layers, so the numbers
isolate architectural cost. Each run reports ms/iter and peak CUDA memory
for a training step (fwd+bwd).

Fairness note: non-3D representations need no per-bar TRACK padding in real
data — inactive tracks simply contribute no patches/tokens. Event padding
within a track is kept everywhere (same E for 3D and [BT,E]). Default
--active-tracks 32 compares raw capacity (all grids padded); pass e.g.
--active-tracks 12 to emulate the observed average density, where
[BT,E] holds B*12 patches, [B,TE] holds 12*E ~= 384 tokens per bar, and 1D
holds B*12*E tokens, while 3D still pays the padded 32x32 track-event grid.

--components instead breaks one event-level training step into its attention
path (ln+qkv+SDPA+proj), MLP path (ln+mlp), and the SDPA kernel alone, at
each layout's characteristic (n_seq, seq_len) — quantifying how much of the
compute is position-wise (and thus indifferent to packing vs. padding).

Usage:
    python benchmark_architectures.py                     # all archs, defaults
    python benchmark_architectures.py --arch 3d 3d_packed # subset
    python benchmark_architectures.py --active-tracks 12
    python benchmark_architectures.py --bf16 --compile
    python benchmark_architectures.py --components --active-tracks 12
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Pre-LN causal transformer block with SDPA attention."""

    def __init__(self, dim, head_dim=64):
        super().__init__()
        assert dim % head_dim == 0
        self.n_head = dim // head_dim
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # x: (n_seq, seq_len, dim)
        n, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (
            t.view(n, s, self.n_head, -1).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(n, s, d)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class Stack(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(n_layers))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def make_track_mask(b, B, T, T_eff, device):
    """Exactly T_eff active tracks per bar, at random positions."""
    idx = torch.rand(b, B, T, device=device).topk(T_eff, dim=-1).indices
    mask = torch.zeros(b, B, T, dtype=torch.bool, device=device)
    mask.scatter_(-1, idx, True)
    return mask


def pack_tracks(x, flat_idx):
    """(b,B,T,E,D) logical grid -> (n_active, E, D) flattened active tracks."""
    b, B, T, E, D = x.shape
    return x.reshape(b * B * T, E, D).index_select(0, flat_idx)


def unpack_tracks(packed, flat_idx, n_flat):
    """(n_active, E, D) flattened -> (n_flat, E, D) logical grid (zeros where
    inactive). Together with pack_tracks this is the 2D<->3D mapping."""
    full = torch.zeros(
        n_flat, packed.shape[1], packed.shape[2],
        device=packed.device, dtype=packed.dtype,
    )
    return full.index_copy(0, flat_idx, packed)


class Arch(nn.Module):
    """One factorization: a list of (name, stack) applied top-down with
    mean-pool bottom-up and broadcast-add top-down, per `plan`."""

    def __init__(self, name, dim, plan, flat_idx=None):
        # plan: list of dicts, ordered top (coarsest) -> bottom (finest);
        # each has 'layers' and a 'shape' fn mapping the full grid
        # (b, B, T, E, D) tensor to (n_seq, seq_len, D). A level marked
        # 'pack': True runs its stack on active tracks only, mapping
        # 2D-flattened <-> 3D-logical via pack_tracks/unpack_tracks.
        super().__init__()
        self.name = name
        self.plan = plan
        self.stacks = nn.ModuleList(Stack(dim, p["layers"]) for p in plan)
        if flat_idx is not None:
            self.register_buffer("flat_idx", flat_idx, persistent=False)

    def forward(self, x):
        # x: (b, B, T, E, D) padded grid
        ctx = None
        out = None
        for i, (p, stack) in enumerate(zip(self.plan, self.stacks)):
            seq = p["shape"](x)  # (n_seq, seq_len, D), coarse -> fine
            n_logical = seq.shape[0]
            if p.get("pack"):
                seq = seq.index_select(0, self.flat_idx)
                if ctx is not None:
                    ctx = ctx.index_select(0, self.flat_idx)
            if ctx is not None:
                # each upper-level position conditions one finer sequence:
                # bar->track, track->event, patch->event, bar->(T*E) events
                seq = seq + ctx.unsqueeze(1)
            out = stack(seq)
            if p.get("pack"):
                # back to the logical grid for downstream (e.g. bar encoding)
                out = unpack_tracks(out, self.flat_idx, n_logical)
            last = i == len(self.plan) - 1
            # per-position outputs become per-sequence context one level down
            ctx = None if last else out.reshape(-1, out.shape[-1])
        return out


def build_arch(name, dim, B, T, E, T_eff, layers_event, layers_upper,
               flat_idx=None):
    """Returns the architecture module for the given name.

    T_eff: active tracks kept by non-3D variants (they drop track padding);
    the plain 3D grid always uses the full padded T, while 3d_packed runs
    its event stack on active tracks only (via flat_idx) and keeps the
    logical grid at the bar/track levels.
    """
    if name == "1d":
        plan = [
            {"layers": layers_event + layers_upper,
             "shape": lambda x: x[:, :, :T_eff].reshape(x.shape[0], -1, x.shape[-1])}
        ]
    elif name == "2d_b_te":
        plan = [
            {"layers": layers_upper,
             "shape": lambda x: x[:, :, :T_eff].mean(dim=(2, 3))},    # (b, B, D)
            {"layers": layers_event,
             "shape": lambda x: x[:, :, :T_eff].reshape(
                 x.shape[0] * x.shape[1], -1, x.shape[-1])},          # (b*B, T_eff*E, D)
        ]
    elif name == "2d_bt_e":
        plan = [
            {"layers": layers_upper,
             "shape": lambda x: x[:, :, :T_eff].mean(dim=3).reshape(
                 x.shape[0], -1, x.shape[-1])},                       # (b, B*T_eff, D)
            {"layers": layers_event,
             "shape": lambda x: x[:, :, :T_eff].reshape(
                 -1, x.shape[3], x.shape[-1])},                       # (b*B*T_eff, E, D)
        ]
    elif name in ("3d", "3d_packed"):
        lu1, lu2 = layers_upper - layers_upper // 2, layers_upper // 2
        plan = [
            {"layers": lu1,
             "shape": lambda x: x.mean(dim=(2, 3))},                  # bar:   (b, B, D)
            {"layers": lu2,
             "shape": lambda x: x.mean(dim=3).reshape(
                 -1, x.shape[2], x.shape[-1])},                       # track: (b*B, T, D)
            {"layers": layers_event, "pack": name == "3d_packed",
             "shape": lambda x: x.reshape(-1, x.shape[3], x.shape[-1])},  # event: (b*B*T, E, D)
        ]
    else:
        raise ValueError(name)
    return Arch(name, dim, plan, flat_idx=flat_idx)


LABELS = {
    "3d": "3D [B,T,E]",
    "3d_packed": "3D packed",
    "2d_bt_e": "2D [BT,E]",
    "2d_b_te": "2D [B,TE]",
    "1d": "1D [BTE]",
}


def decode_state_mib(arch_name, args, T_eff):
    """Cached state during autoregressive decoding (Table 1's VRAM column):
    K+V per cached token per layer, using each level's maximal live context.
    Conditioning caches (shared by all variants) are omitted; the 3D figure
    includes the previous-bar hidden states kept for track-aligned retrieval.
    """
    bytes_el = 2 if args.bf16 else 4
    kv = 2 * args.dim * bytes_el  # K+V bytes per token per layer
    Le, Lu = args.event_layers, args.upper_layers
    B, T, E = args.bars, args.tracks, args.events
    extra = 0
    if arch_name == "1d":
        tok_layers = (Le + Lu) * B * T_eff * E
    elif arch_name == "2d_b_te":
        tok_layers = Lu * B + Le * T_eff * E          # bar cache + current bar
    elif arch_name == "2d_bt_e":
        tok_layers = Lu * B * T_eff + Le * E          # patch cache + current patch
    else:  # 3d
        lu1, lu2 = Lu - Lu // 2, Lu // 2
        tok_layers = lu1 * B + lu2 * T + Le * E
        extra = 2 * Le * T * E * args.dim * bytes_el
    return (tok_layers * kv + extra) / 2**20


def bench(arch_name, args, device):
    torch.cuda.empty_cache()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    B, T, E = args.bars, args.tracks, args.events
    # non-3D variants drop per-bar track padding; 3D keeps the full padded grid
    T_eff = T if arch_name == "3d" else args.active_tracks

    flat_idx = None
    if arch_name == "3d_packed":
        mask = make_track_mask(args.batch, B, T, args.active_tracks, device)
        flat_idx = torch.where(mask.reshape(-1))[0]

    model = build_arch(
        arch_name, args.dim, B, T, E, T_eff, args.event_layers,
        args.upper_layers, flat_idx=flat_idx,
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    def make_input():
        return torch.randn(args.batch, B, T, E, args.dim, device=device, dtype=dtype)

    fwd = torch.compile(model, fullgraph=True) if args.compile else model

    def train_step():
        y = fwd(make_input())
        y.float().sum().backward()
        model.zero_grad(set_to_none=True)

    label = LABELS[arch_name]
    try:
        # ---- training: fwd + bwd ----
        model.train()
        for _ in range(args.warmup):
            train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            train_step()
        torch.cuda.synchronize()
        train_ms = (time.perf_counter() - t0) / args.iters * 1000
        train_mem = torch.cuda.max_memory_allocated() / 2**20

        print(
            f"{label:<12} params={n_params / 1e6:6.1f}M  "
            f"train: {train_ms:9.2f} ms  {train_mem:9.0f} MiB   "
            f"decode-state~{decode_state_mib(arch_name, args, T_eff):7.1f} MiB/sample"
        )
    except torch.cuda.OutOfMemoryError:
        print(f"{label:<12} params={n_params / 1e6:6.1f}M  OOM")
    del model, fwd
    torch.cuda.empty_cache()


def bench_components(args, device):
    """Time the attention path vs. the MLP path of one event-level training
    step at each layout's characteristic (n_seq, seq_len). Total tokens are
    identical across rows; only the grouping changes."""
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    B, T, E = args.bars, args.tracks, args.events
    n_tok_pad = args.batch * B * T * E
    shapes = {
        "3D [B,T,E]": (args.batch * B * T, E),
        "2D [BT,E]": (args.batch * B * args.active_tracks, E),
        "2D [B,TE]": (args.batch * B, args.active_tracks * E),
        "1D [BTE]": (args.batch, B * args.active_tracks * E),
    }
    block = Block(args.dim).to(device=device, dtype=dtype)

    def attn_path(x):
        n, s, d = x.shape
        q, k, v = block.qkv(block.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(n, s, block.n_head, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return block.proj(y.transpose(1, 2).reshape(n, s, d))

    def sdpa_kernel(x):
        # pure attention kernel: no LN/projections (those are position-wise)
        n, s, d = x.shape
        q = k = v = x.view(n, s, block.n_head, -1).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def mlp_path(x):
        return block.mlp(block.ln2(x))

    parts = [("attn path", attn_path), ("sdpa kernel", sdpa_kernel),
             ("mlp path", mlp_path)]
    print(f"\nPer-layer component timings, fwd+bwd, ms "
          f"(padded tokens={n_tok_pad}, packed={shapes['1D [BTE]'][0] * shapes['1D [BTE]'][1]})")
    print(f"{'layout':<12}" + "".join(f"{n:>14}" for n, _ in parts))
    for label, (n_seq, s) in shapes.items():
        cols = []
        for _, fn in parts:
            x = torch.randn(n_seq, s, args.dim, device=device, dtype=dtype,
                            requires_grad=True)
            for _ in range(args.warmup):
                fn(x).float().sum().backward()
                x.grad = None
                block.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn(x).float().sum().backward()
                x.grad = None
                block.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            cols.append((time.perf_counter() - t0) / args.iters * 1000)
        print(f"{label:<12}" + "".join(f"{c:14.3f}" for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", nargs="+", default=["3d", "3d_packed", "2d_bt_e", "2d_b_te", "1d"],
                   choices=list(LABELS))
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--tracks", type=int, default=32)
    p.add_argument("--events", type=int, default=32)
    p.add_argument("--active-tracks", type=int, default=32,
                   help="active tracks per bar for non-3D variants, which "
                        "drop track padding (set ~12 to emulate real average "
                        "density; 3D always pays the padded track grid)")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--event-layers", type=int, default=8)
    p.add_argument("--upper-layers", type=int, default=4,
                   help="layers for coarser levels (3D splits them bar/track); "
                        "total layers = event+upper for every architecture")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--components", action="store_true",
                   help="instead of full architectures, time the attention "
                        "vs. MLP paths of one block at each layout's "
                        "(n_seq, seq_len)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}  "
          f"B={args.bars} T={args.tracks} E={args.events} "
          f"(active tracks for non-3D={args.active_tracks})  D={args.dim}  "
          f"layers={args.event_layers}+{args.upper_layers}  batch={args.batch}  "
          f"dtype={'bf16' if args.bf16 else 'fp32'}  compile={args.compile}")
    if args.components:
        bench_components(args, device)
        return
    for name in args.arch:
        bench(name, args, device)


if __name__ == "__main__":
    main()
