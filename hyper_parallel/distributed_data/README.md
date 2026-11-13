# Distributed sample balancing

This PyTorch data pipeline reassigns complete samples across data-parallel
replicas before packing. It provides an independent build API without requiring
a particular trainer, dataset package, FSDP implementation or model.

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
```

The planner uses capacity-constrained greedy placement and bounded refinement.
It accepts a candidate only when its predicted objective improves by more than
`min_balance_gain`. Every selected sample occurrence is retained exactly once;
packing never mixes independent samples' attention or labels on its own. The
application's `pack_fn` is responsible for rebuilding valid model inputs.

## Mesh and communication

Both `torch.distributed.device_mesh.DeviceMesh` and HP `DeviceMesh` are accepted.
Pass the named **root mesh covering WORLD**, with `dp_dim_names` describing
all dimensions that distinguish independent data replicas. For example:

```python
# HSDP8 x CP4: all eight DP coordinates get different data; CP peers share it.
mesh = DeviceMesh("npu", ranks, mesh_dim_names=("dp", "cp"))
config = DistributedDatasetConfig(
    seq_len=max_seq_len,
    local_batch_size=microbatches_per_optimizer_step,
    dp_dim_names=("dp",),
    balance_group_size=8,
)
```

For a `(dp_replicate, dp_shard, cp)` root mesh, use
`dp_dim_names=("dp_replicate", "dp_shard")`. Do not identify DP replicas by the
parameter-sharding group alone. Mesh ranks may be noncontiguous or permuted.

One rank at model-parallel coordinate zero is the Data Constructor for each DP
coordinate. Constructors read samples, gather metadata, plan and exchange moved
sample payloads with A2A. Other CP/TP/PP peers do not advance their data source.
The constructor then broadcasts each complete packed batch to its model peers.
**There is no CP sequence slicing in the dataset.** Models retain control of CP.

External sources default to balancing among constructors on the same node
(`GROUP_RANK`, falling back to hostname). Set `balance_group_size` to the total
DP size to balance across nodes, or to a divisor for independent groups. The
size counts **DP constructors, not WORLD ranks**. Native sampler loading uses a
global constructor group. New data groups do not modify model/FSDP groups.

`communication_backend="hccl"` is the default and requires the rank-local NPU
`device`. `"gloo"` transports CPU data and defaults to CPU output; with an
explicit accelerator device, H2D and model tensor broadcasts use that device.
Select the current NPU before constructing the loader on every rank.

Pass `batch_to_device_fn(pinned_batch, device)` to keep application metadata on
CPU while prefetching model tensors. HP invokes it on its independent copy
stream after pinning; use nonblocking copies on that stream and return a
tensor/container tree. HP records completion and waits on the consumer stream.
Without this callback, all tensors move to the selected accelerator. CPU output
does not invoke it. Dynamic external broadcasts preserve each returned tensor's
CPU/device placement; a native `model_batch_spec` must match the returned tree.
Checkpoints retain the original Host batch and apply the callback again on
resume, so keep its semantics consistent across ranks and resumed runs.

## Existing loaders and HY

Use `external_step_source` to preserve the application's existing tokenizer,
processor, sampler, filtering and pack selection. Each yield must contain one
complete local optimizer step as `[[raw_sample, ...], ...]`, with exactly
`config.local_batch_size` nonempty bins. The source owns DP sharding, RNG,
workers and step limits. Every constructor must yield the same number of steps.

For HY, build the existing dynamic loader with an identity **final** collator,
so it returns raw bins after its original selection. Keep its internal read
collator unchanged. Pass HY's real `MyCollator` as the HP packing callback:

```python
from contextlib import closing
from hyper_parallel.distributed_data import WorkloadCost


def metadata_fn(sample):
    # Count the actual packed footprint, including diffusion/image positions.
    # TND Attention cost uses sum(length_i**2), not just the number of tokens.
    return SampleMetadata(
        pack_tokens=packed_token_count(sample),
        cost=WorkloadCost(llm=estimated_work(sample)),
        packing_costs=stage_costs(sample),
    )


def pack_fn(samples, seq_len):
    return hy_collator(samples)


with closing(build_distributed_dataloader(
    None, mesh, config,
    external_step_source=hy_raw_step_loader,
    metadata_fn=metadata_fn,
    cost_model=lambda metadata: metadata.cost,
    pack_fn=pack_fn,
    collate_fn=list,
    device=device,
)) as loader:
    for microbatches in loader:
        for index, batch in enumerate(microbatches):
            loss = forward(batch)
            if index == 0:
                loader.prefetch_plan()
            loss.backward()
            if index == 0:
                loader.prefetch()
        optimizer.step()
        optimizer.zero_grad()
```

This imports the builder directly from HP; HY supplies only its source and
model-specific callbacks. HP owns planning, sample exchange, H2D, model-peer
broadcast and loader checkpoints. Do not also broadcast the same inputs with
a separate HY CP broadcast hook. If HY must retain CPU inputs for preprocessing,
use `communication_backend="gloo", device="cpu"` and move them later.

`packing_budgets` sets optional additive hard limits for ViT/VAE or other stages.
Metadata must contain all configured costs. `seq_len` is a fixed packing cap;
a source's token-budget warmup is not automatically propagated to it. Disable
warmup or supply a compatible fixed cap for this integration. Avoid custom
collators that depend on uncaptured process-global random state.

By default `DefaultCostModel(model_config)` estimates backbone FLOPs from `P`,
`D` and conditional-image token runs in `SampleMetadata.features`. `P+D` must
equal `pack_tokens`. Pass the effective layer configuration (including an
accurate `mlp_layer_types` list). An explicit `cost_model`, as above, replaces
that estimate. `balancing_algorithm` can override assignment/objective while
HP enforces capacities, membership and the gain threshold. The external route
balances total workload per DP optimizer step; native windows additionally
model the synchronized forward/backward rounds.

## Prefetch and variable model inputs

External loading buffers one future step. With HCCL the main training thread
launches metadata and payload communication on an independent data stream.
Source loading, planning, decoding and collation run in a background worker;
H2D uses a copy stream. Call `prefetch_plan()` and `prefetch()` at matching
boundaries on all ranks, before blocking loss reads, to overlap that work with
queued model computation. Without these hooks, `next(loader)` completes the
same phases on demand. Gloo runs the pipeline in its background producer.

For external sources, dict/list/tuple trees of dense tensors and
`None/bool/int/float/str` values can change shape and structure on every step.
HP broadcasts tree/dtype/shape metadata separately, then tensor payloads in
their original dtypes. Empty tensors and variable image lists are supported.
An optional `model_batch_spec` avoids dynamic tree serialization. Native custom
MP batches require that fixed declaration; see `TensorFieldSpec` in
[`model_batch.py`](model_batch.py). Pure-DP loading does not need a declaration.

`last_host_batch` and `last_balance_stats` are available on constructors for
host-side metering; non-constructor model peers do not keep those views.
`config.log_balance=True` enables INFO reports on rank zero. Source and planner
errors are forwarded to model peers, but data-plane failures remain subject to
the distributed process-group timeout: all readers must have valid inputs and
matching step counts.

## Checkpoint and resume

Call `loader.state_dict()` on **every rank at the same complete optimizer-step
boundary**, after saving consistent model/optimizer/RNG progress. Store each
rank's loader state with its corresponding training checkpoint, for example:

```python
# At a coordinated completed optimizer step on every rank:
state = loader.state_dict()
torch.save(state, f"{checkpoint_dir}/data_rank_{dist.get_rank()}.pt")

# Rebuild the source and loader with the same mesh, config and callbacks:
state = torch.load(
    f"{checkpoint_dir}/data_rank_{dist.get_rank()}.pt", weights_only=False,
)
loader.load_state_dict(state)
loader.set_epoch(saved_epoch)  # Keeps a just-restored same-epoch cursor.
for microbatches in loader:
    train_step(microbatches)
```

For **external sources**, the source must provide both `state_dict` and
`load_state_dict`. Its state must include its RNG, pack buffers and worker
progress. HP drains speculative work, then saves the advanced source state
**together with the unconsumed prefetched host batch**. Restore delivers that
saved batch first, avoiding skipped or duplicated samples. No source snapshot
is copied on every training step. The checkpoint can be large because it
contains actual pending inputs and source buffers. Stateless iterables still
work for training, but checkpoint requests fail explicitly.

External checkpoints require the same root mesh, rank placement, balancing
groups and `DistributedDatasetConfig`; elastic resizing is not supported.
Call `set_epoch(new_epoch)` to start another epoch. Reentering `iter(loader)`
preserves an unfinished iterator and never discards its pending input.
Custom policies, dataset contents and source configuration must also remain
compatible; callback identities are not fingerprinted.

For **native sampling**, the loader commits only complete optimizer-step
cursors, independent of speculative sampler progress. A checkpoint taken
inside gradient accumulation replays that optimizer step in full. Dataset
reads and transforms must be deterministic by index and epoch; arbitrary
worker RNG state is not captured by native replay. See the sampler and loader
APIs for their cursor validation. Training checkpoints must align with that
complete-step cursor.

## Native mapping datasets

For datasets addressable by index, use the bundled HP batch sampler:

```python
from hyper_parallel.distributed_data import build_dataset_batch_sampler

sampler = build_dataset_batch_sampler(
    total_samples=len(dataset), micro_batch_size=1,
    global_batch_size=dp_size * accumulation_steps,
    dp_world_size=dp_size, dp_rank=dp_rank,
)
loader = build_distributed_dataloader(
    dataset, mesh,
    DistributedDatasetConfig(
        seq_len=seq_len, local_batch_size=1,
        dp_dim_names=dp_dims, balance_window_size=accumulation_steps,
        communication_backend="gloo",
    ),
    batch_sampler=sampler, metadata_fn=metadata_fn,
    pack_fn=pack_fn, collate_fn=collate_fn,
    model_batch_spec=model_batch_spec,
    cost_model=cost_model, device="cpu",
)
```

Every rank constructs the sampler with its DP coordinate; only constructors
advance it. Each loader yield is one forward/backward round. Windows are clipped
at optimizer and epoch boundaries; `balance_window_size` controls host planning
and `device_prefetch_depth` independently controls device lookahead. Dataset
outputs remain indivisible by default. `split_fn` plus `pack_fn`, or built-in
`repack_sequences=True`, permits independent sequence repacking.

With `metadata_mode=True`, provide `metadata[index]` instead of `metadata_fn`.
Constructors read their assigned indices directly from the shared dataset,
avoiding payload A2A. Metadata must describe deterministic, rank-independent
outputs, and lengths must agree across constructors. Native sampling requires
`drop_last=True` and `dataset_already_sharded=False`.

See [NODE_LOCAL_BALANCING.md](NODE_LOCAL_BALANCING.md) for the external pipeline
and policy details.
