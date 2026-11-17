# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Independent data communication and staging with inline CPU metadata."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import os
import pickle
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from datetime import timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cloudpickle
import numpy as np
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.device_prefetch import (
    DevicePrefetchedStep,
    DeviceStepPrefetcher,
)
from hyper_parallel.distributed_data.model_batch import ModelBatchLayout
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    DataPlaneTransport,
    _decode_received_payloads,
    _encode_payload_segment,
)

_REPLY_TIMEOUT = 300


class ProducerTransport:
    """Encode CPU data and execute collectives in the data process."""

    def __init__(self, ranks: tuple[int, ...], planner_rank: int, global_rank: int, device: Any) -> None:
        """Map logical sample owners onto the child process group."""
        self.ranks = ranks
        self.planner_rank = planner_rank
        distributed = len(ranks) > 1
        group = dist.group.WORLD if distributed else None
        groups = DataGroups(ranks, group, group, None, ranks.index(planner_rank), distributed)
        self._transport = DataPlaneTransport(groups, global_rank, communication_device=device)

    def _call(self, operation: str, value: Any) -> Any:
        return self._transport.exchange_cpu_request(operation, value)

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Encode and decode metadata exclusively in the CPU producer.

        Args:
            value: CPU request value or nested tensor batch.
        """
        if len(self.ranks) == 1:
            return (value,)
        encoded = torch.frombuffer(bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), dtype=torch.uint8)
        payload, sizes = self._call("gather", encoded)
        rows = payload.view(len(sizes), -1).numpy()
        return tuple(pickle.loads(row[:size].tobytes()) for row, size in zip(rows, sizes))

    def broadcast_from_planner(self, value: Any) -> Any:
        """Broadcast planner output in the data process.

        Args:
            value: CPU request value or nested tensor batch.
        """
        if len(self.ranks) == 1:
            return value
        encoded = torch.frombuffer(bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), dtype=torch.uint8)
        return pickle.loads(self._call("broadcast", encoded).numpy().tobytes())

    def exchange(self, outgoing: dict[int, Any]) -> dict[Any, Any]:
        """Serialize raw samples and execute byte A2A in the data process.

        Args:
            outgoing: Sample payloads grouped by destination global rank.
        """
        unexpected = outgoing.keys() - self.ranks
        if unexpected:
            raise ValueError(f"Payload routes target ranks outside the data plane: {sorted(unexpected)}.")
        segments = [_encode_payload_segment(outgoing.get(rank, ())) for rank in self.ranks]
        sizes = tuple(map(len, segments))
        storage = bytearray(b"".join(segments))
        encoded = torch.frombuffer(storage, dtype=torch.uint8) if storage else torch.empty(0, dtype=torch.uint8)
        if len(self.ranks) == 1:
            return _decode_received_payloads(bytes(storage), sizes)
        received, splits = self._call("exchange", (encoded, sizes))
        return _decode_received_payloads(received.numpy().tobytes(), splits)


def _cpu_rng_state() -> tuple[Any, Any, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _restore_cpu_rng(state: tuple[Any, Any, torch.Tensor]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])


def _cpu_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize CPU tensor contents without invoking multiprocessing reducers."""
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError("Inline metadata requires a dense CPU tensor.")
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _cpu_tensor_from_bytes(raw: bytes, field: Any) -> torch.Tensor:
    """Give an inline CPU tensor its own storage, including empty/scalar values."""
    if not raw:
        if 0 not in field.shape:
            raise ValueError("Missing contents for a nonempty CPU tensor.")
        return torch.empty(field.shape, dtype=field.dtype)
    return torch.frombuffer(bytearray(raw), dtype=field.dtype).reshape(field.shape)


def _encode_host_batch(batch: Any) -> tuple[Any, list[bytes]]:
    """Materialize a complete CPU batch only for an explicit debugging request."""
    schema = ModelBatchLayout.describe(batch)
    return schema, [_cpu_tensor_bytes(tensor) for tensor in ModelBatchLayout(schema).flatten(batch)]


def _decode_host_batch(message: tuple[Any, list[bytes]]) -> Any:
    """Rebuild a debugging snapshot without opening shared-memory files."""
    schema, values = message
    layout = ModelBatchLayout(schema)
    return layout.rebuild([_cpu_tensor_from_bytes(raw, field) for raw, field in zip(values, layout.fields)])


def _store_descriptor() -> dict[str, Any]:
    """Reuse the training rendezvous without assuming a particular launcher."""
    # PyTorch exposes no public accessor for the already initialized rendezvous.
    store = dist.distributed_c10d._get_default_store()  # pylint: disable=protected-access
    while isinstance(store, dist.PrefixStore):
        store = store.underlying_store
    if isinstance(store, dist.TCPStore):
        return {"kind": "tcp", "address": store.host, "port": store.port}
    if isinstance(store, dist.FileStore):
        return {"kind": "file", "path": store.path}
    raise ValueError("Independent data processes require a TCPStore or FileStore rendezvous.")


def _child_environment(backend: str) -> dict[str, str]:
    """Keep a child's HCCL listener separate from the training process."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if backend != "hccl":
        return env
    selected = env.get("HP_DATA_HCCL_SOCKET_PORT_RANGE", "50000-50031")
    match = re.fullmatch(r"(\d+)-(\d+)", selected)
    if match is None or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
        raise ValueError("HP_DATA_HCCL_SOCKET_PORT_RANGE must be an inclusive range within 1024-65535.")
    parent = env.get("HCCL_NPU_SOCKET_PORT_RANGE", "16666-16666")
    for start, end in re.findall(r"(\d+)-(\d+)", parent):
        if max(int(start), int(match[1])) <= min(int(end), int(match[2])):
            raise ValueError("HP_DATA_HCCL_SOCKET_PORT_RANGE overlaps the training HCCL listener range.")
    env["HCCL_NPU_SOCKET_PORT_RANGE"] = selected
    return env


def _initialize_child_group(bootstrap: dict[str, Any], ranks: tuple[int, ...],
                            stage: DeviceStepPrefetcher | None) -> bool:
    """Bind the output device before joining a namespaced data group."""
    if stage is not None:
        getattr(torch, stage.device.type).set_device(stage.device)
    if len(ranks) == 1:
        return False
    if bootstrap["kind"] == "tcp":
        store = dist.TCPStore(bootstrap["address"], bootstrap["port"], is_master=False,
                              timeout=timedelta(seconds=_REPLY_TIMEOUT))
    else:
        store = dist.FileStore(bootstrap["path"])
    store = dist.PrefixStore(bootstrap["prefix"], store)
    dist.init_process_group(bootstrap["backend"], store=store, rank=bootstrap["rank"],
                            world_size=len(ranks), timeout=timedelta(seconds=600))
    return True


def _prepare_batch_message(host_batch: Any, stage: DeviceStepPrefetcher | None, stats: Any) -> tuple:
    """Finish device writes and encode only retained CPU fields as bytes."""
    batch = host_batch
    if stage is not None:
        staged = stage(host_batch)
        staged.ready_event.synchronize()
        batch = staged.device_batch
    layout = ModelBatchLayout.describe(batch)
    leaves = ModelBatchLayout(layout).flatten(batch)
    device_indices = [i for i, tensor in enumerate(leaves) if tensor.device.type != "cpu"]
    cpu_indices = [i for i, tensor in enumerate(leaves) if tensor.device.type == "cpu"]
    cpu_values = [_cpu_tensor_bytes(leaves[i]) for i in cpu_indices]
    device_values = [leaves[i] for i in device_indices]
    return layout, cpu_indices, device_indices, device_values, cpu_values, stats


def _producer_main(capsule_path: str, descriptor: int) -> None:
    """Own source, collectives and staging in a fresh interpreter."""
    connection = Connection(descriptor)
    cleanup = ExitStack()
    cleanup.callback(connection.close)
    host_batches = {}
    initialized = False
    try:
        torch.set_num_threads(1)
        with open(capsule_path, "rb") as handle:
            module_paths, capsule = pickle.load(handle)
        sys.path[:] = module_paths
        factory, ranks, planner_rank, rng_state, stage, device, bootstrap = cloudpickle.loads(capsule)
        _restore_cpu_rng(rng_state)
        initialized = _initialize_child_group(bootstrap, ranks, stage)
        transport = ProducerTransport(ranks, planner_rank, ranks[bootstrap["rank"]], device)
        producer = factory(transport)
        cleanup.callback(producer.close)
        connection.send(("READY", os.getpid()))
        while True:
            command, value = connection.recv()
            if command == "CLOSE":
                if stage is not None:
                    stage.close()
                    getattr(torch, stage.device.type).synchronize(stage.device)
                connection.send(("CLOSED", None))
                break
            if command == "NEXT":
                try:
                    host_batch, stats = next(producer)
                except StopIteration:
                    connection.send(("EOF", None))
                else:
                    host_batches[value] = host_batch
                    if len(host_batches) > 2:
                        del host_batches[next(iter(host_batches))]
                    message = _prepare_batch_message(host_batch, stage, stats)
                    connection.send(("BATCH", message))
            elif command == "HOST":
                connection.send(("HOST", _encode_host_batch(host_batches[value])))
            elif command == "SAVE":
                consumed, path = value
                torch.save(producer.state_dict(consumed), path)
                connection.send(("SAVED", None))
            elif command == "LOAD":
                host_batches.clear()
                producer.load_state_dict(torch.load(value, map_location="cpu", weights_only=False))
                connection.send(("LOADED", None))
            elif command == "EPOCH":
                host_batches.clear()
                producer.set_epoch(value)
                connection.send(("EPOCH", None))
            else:
                raise ValueError(f"Unknown producer command: {command}.")
    except BaseException:
        try:
            connection.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        cleanup.close()
        if initialized:
            dist.destroy_process_group()


class ProcessDataProducer:
    """Run source, collectives and H2D in one child; consume IPC in the parent."""

    def __init__(self, factory: Callable[..., Any], transport: DataPlaneTransport,
                 stage: DeviceStepPrefetcher | None = None) -> None:
        """Capture a replayable source and its child-owned device stage."""
        self._resources = ExitStack()
        self._directory = self._resources.enter_context(TemporaryDirectory(prefix="hp-producer-"))
        self._stage = stage
        self._process = None
        self._connection = None
        self._pending = False
        self._result = None
        self._pending_error = None
        self._request_id = 0
        self._delivered_id = None
        self._host_snapshot = None
        self._ipc_pending = []
        self._closed = False
        self._failed = False
        self._capsule = Path(self._directory) / "source.pkl"
        try:
            self._env = _child_environment(transport.communication_backend)
            bootstrap = {"rank": 0, "backend": transport.communication_backend}
            if len(transport.ranks) > 1:
                tokens = transport.all_gather_object(uuid.uuid4().hex)
                bootstrap.update(_store_descriptor(), prefix="hp-device-producer-" + tokens[0],
                                 rank=transport.ranks.index(dist.get_rank()))
            with self._capsule.open("wb") as handle:
                capsule = cloudpickle.dumps((factory, transport.ranks, transport.planner_rank,
                                             _cpu_rng_state(), stage, transport.communication_device, bootstrap))
                pickle.dump((sys.path, capsule), handle)
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        if self._closed:
            raise RuntimeError("Data producer is closed.")
        if self._process is not None:
            return
        parent, child = socket.socketpair()
        try:
            self._process = self._resources.enter_context(subprocess.Popen(
                [sys.executable, "-c",
                 ("import sys\nfrom hyper_parallel.distributed_data.process_data import _producer_main\n"
                  "_producer_main(sys.argv[1], int(sys.argv[2]))\n"),
                 str(self._capsule), str(child.fileno())],
                pass_fds=(child.fileno(),), start_new_session=True, env=self._env))
            self._connection = Connection(parent.detach())
            status, _ = self._receive()
            if status != "READY":
                raise RuntimeError(status)
        finally:
            child.close()
            parent.close()

    def _receive(self) -> tuple[str, Any]:
        deadline = time.monotonic() + _REPLY_TIMEOUT
        while not self._connection.poll(0.1):
            if self._process.poll() is not None:
                raise RuntimeError(f"Data producer exited with status {self._process.returncode}.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Data producer did not reply within the data timeout.")
        with torch.profiler.record_function("ipc/receive_tensor_handles"):
            try:
                status, value = self._connection.recv()
            except (EOFError, OSError) as error:
                self._failed = True
                raise RuntimeError("Data producer disconnected before returning its result.") from error
        if status == "ERROR":
            self._failed = True
            raise RuntimeError(f"Data producer failed:\n{value}")
        if status == "BATCH":
            schema, cpu_indices, device_indices, device_values, cpu_values, stats = value
            accelerator = None
            if self._stage is not None:
                accelerator = getattr(torch, self._stage.device.type)
                accelerator.set_device(self._stage.device)
            layout = ModelBatchLayout(schema)
            leaves = [None] * len(layout.fields)
            for index, raw in zip(cpu_indices, cpu_values):
                leaves[index] = _cpu_tensor_from_bytes(raw, layout.fields[index])
            with torch.profiler.record_function("ipc/release_previous_handles"):
                self._ipc_pending = [(ready, tensors) for ready, tensors in self._ipc_pending if not ready.query()]
            with torch.profiler.record_function("ipc/copy_device_batch"):
                for index, tensor in zip(device_indices, device_values):
                    leaves[index] = tensor.clone()
            batch = layout.rebuild(leaves)
            if accelerator is not None:
                event = accelerator.Event()
                event.record()
                self._ipc_pending.append((event, device_values))
                batch = DevicePrefetchedStep(batch, event, self._stage.device)
            value = (batch, stats)
        return status, value

    def prefetch(self) -> None:
        """Request one speculative step without starting a communication thread."""
        if not self._pending:
            self._start()
            self._request_id += 1
            try:
                self._connection.send(("NEXT", self._request_id))
            except (BrokenPipeError, EOFError, OSError) as error:
                # A ready batch stays consumable even if its successor cannot start.
                self._pending_error = error
                self._failed = True
            self._pending = True

    def wait(self) -> tuple[str, Any] | None:
        """Receive pending work without advancing the consumed cursor."""
        if self._pending_error is not None:
            raise RuntimeError("Data producer disconnected before accepting prefetch.") from self._pending_error
        if self._pending and self._result is None:
            try:
                self._result = self._receive()
            except BaseException:
                self._failed = True
                raise
        return self._result

    def take(self) -> Any:
        """Consume one result and retain its identifier for explicit debugging."""
        self.prefetch()
        status, result = self.wait()
        self._pending = False
        self._result = None
        if status == "EOF":
            raise StopIteration
        if status != "BATCH":
            raise RuntimeError(status)
        self._delivered_id = self._request_id
        self._host_snapshot = None
        return result

    @property
    def last_host_batch(self) -> Any:
        """Fetch the delivered CPU batch on demand; this may wait for prefetch."""
        if self._closed or self._delivered_id is None:
            return None
        if self._host_snapshot is None:
            self.wait()
            status, message = self._command("HOST", self._delivered_id)
            if status != "HOST":
                raise RuntimeError(f"Expected a Host snapshot, received {status}.")
            self._host_snapshot = _decode_host_batch(message)
        return self._host_snapshot

    def _command(self, command: str, value: Any) -> tuple[str, Any]:
        self._connection.send((command, value))
        return self._receive()

    def state_dict(self, consumed: int) -> dict[str, Any]:
        """Snapshot the consumed cursor and the unconsumed prefetched step.

        Args:
            consumed: Number of steps already delivered to training.
        """
        self._start()
        self.wait()
        path = str(Path(self._directory) / "checkpoint.pt")
        self._command("SAVE", (consumed, path))
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Drain old work before restoring source and pending-batch state.

        Args:
            state: Previously saved source and pending-batch state.
        """
        self._start()
        self.wait()
        self._pending, self._result = False, None
        self._delivered_id = self._host_snapshot = None
        path = str(Path(self._directory) / "restore.pt")
        torch.save(state, path)
        self._command("LOAD", path)

    def set_epoch(self, epoch: int) -> None:
        """Discard speculative work and reset the source epoch.

        Args:
            epoch: Source epoch to select; resetting discards speculative work.
        """
        self._start()
        self.wait()
        self._pending, self._result = False, None
        self._delivered_id = self._host_snapshot = None
        self._command("EPOCH", epoch)

    def close(self) -> None:
        """Finish IPC reads before releasing the child and its device memory."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None and self._process.poll() is None and not self._failed:
                self.wait()
                if self._stage is not None:
                    getattr(torch, self._stage.device.type).synchronize(self._stage.device)
                self._ipc_pending.clear()
                self._command("CLOSE", None)
                self._process.wait(timeout=30)
        finally:
            if self._ipc_pending:
                for ready, _ in self._ipc_pending:
                    ready.synchronize()
                self._ipc_pending.clear()
            if self._process is not None:
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self._process.pid, signal.SIGKILL)
                    self._process.wait()
            self._pending, self._result = False, None
            if self._connection is not None:
                self._connection.close()
            self._delivered_id = self._host_snapshot = None
            self._resources.close()
