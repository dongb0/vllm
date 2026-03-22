# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

logger = init_logger(__name__)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    # ===== Layer-wise transfer fields =====
    total_layers: int = 0  # Total number of layers for layer-wise transfer
    layer_names: list[str] = field(default_factory=list)  # Layer names in order
    notify_port: int = 0  # Consumer's notification port for layer completion


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0
    # ===== Layer-wise transfer fields (Consumer side) =====
    layer_progress: dict[str, bool] = field(default_factory=dict)  # Per-layer receive status
    total_layers: int = 0
    completed_layers: int = 0
    layer_events: dict[str, asyncio.Event] = field(default_factory=dict)  # Per-layer completion events


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0
    # ===== Layer-wise transfer fields (Producer side) =====
    layer_progress: dict[str, bool] = field(default_factory=dict)  # Per-layer send status
    total_layers: int = 0
    completed_layers: int = 0
    remote_hostname: str = ""  # D side address (obtained from start_load_kv)
    remote_port: int = 0
    remote_kv_caches_base_addr: list[int] = field(default_factory=list)
    remote_notify_port: int = 0  # D side notification port for layer-wise mode


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        # Layer-wise transfer mode configuration
        self.layer_wise_mode = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "layer_wise", False # TODO: 我觉得应该默认打开；没必要用这个配置
        )
        if self.layer_wise_mode:
            logger.info("MooncakeConnector: Layer-wise transfer mode enabled")

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, self.layer_wise_mode
            )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Consumer side: Wait for specified layer's KV transfer to complete.
        Called within attention layer to ensure this layer's KV is ready.

        Push mode:
        - D passively receives data sent by P
        - Background thread updates layer_progress
        - This function checks and waits for this layer to complete
        """
        if not self.layer_wise_mode:
            # Old mode: no operation (bulk transfer done in start_load_kv)
            return

        assert self.connector_worker is not None

        # Only consumer waits for layer load
        if self.connector_worker.is_kv_producer:
            return

        # Synchronously wait for this layer to complete
        future = asyncio.run_coroutine_threadsafe(
            self.connector_worker.wait_for_layer_load_async(layer_name),
            self.connector_worker.receiver_loop
        )
        future.result()  # Block until this layer completes

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """
        Layer-wise KV transfer implementation (Push mode).
        Called after each layer's forward pass on Producer side.

        Args:
            layer_name: Name of the layer (e.g., "layer_0")
            kv_layer: The KV cache tensor for this layer
            attn_metadata: Attention metadata
        """
        if not self.layer_wise_mode:
            # Old mode: do nothing, wait for request_finished to send all at once
            return

        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)

        # Only producer saves and sends KV
        if self.connector_worker.is_kv_consumer:
            return

        # Submit to worker's sender_loop
        asyncio.run_coroutine_threadsafe(
            self.connector_worker.save_kv_layer_async(
                self._connector_metadata, layer_name, kv_layer, attn_metadata
            ),
            self.connector_worker.sender_loop
        )

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, layer_wise_mode: bool = False):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        self.layer_wise_mode = layer_wise_mode

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer(
                    vllm_config, "0.0.0.0", port
                )
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

            # For layer-wise mode, start a notification listener
            if self.layer_wise_mode:
                self._layer_notify_t = threading.Thread(
                    target=self._layer_notification_listener,
                    daemon=True,
                )
                self._layer_notify_t.start()
                logger.debug("Mooncake Decoder: start layer notification listener")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def __del__(self):
        self.shutdown()

    def _layer_notification_listener(self):
        """
        Background thread on Consumer side to listen for layer completion notifications.
        Producer sends a notification via ZMQ after each layer is transferred.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._layer_notification_loop())

    async def _layer_notification_loop(self):
        """
        Async loop for layer notification listener.
        """
        sock = self.async_zmq_ctx.socket(zmq.PULL)
        self.layer_notify_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.info(
            "Mooncake layer notification listener started on %s:%d",
            self.hostname, self.layer_notify_port
        )

        try:
            while True:
                msg = await sock.recv_json()
                transfer_id = msg.get("transfer_id")
                layer_name = msg.get("layer_name")

                if transfer_id and layer_name:
                    # Find the corresponding pull_meta and mark layer as received
                    found = False
                    for pull_metas in self.reqs_to_recv.values():
                        for req_id, pull_meta in pull_metas.items():
                            if pull_meta.transfer_id == transfer_id:
                                if layer_name not in pull_meta.layer_progress:
                                    pull_meta.layer_progress[layer_name] = True
                                    pull_meta.completed_layers += 1

                                    # Signal waiting coroutines
                                    if layer_name in pull_meta.layer_events:
                                        pull_meta.layer_events[layer_name].set()

                                    logger.debug(
                                        "Layer %s received for req %s (%d/%d)",
                                        layer_name, req_id,
                                        pull_meta.completed_layers,
                                        pull_meta.total_layers
                                    )

                                    # Check if all layers completed
                                    if pull_meta.completed_layers >= pull_meta.total_layers:
                                        self.finished_recving_reqs.add(pull_meta.d_req_id)
                                        logger.info(
                                            "Finished receiving all layers for req %s",
                                            req_id
                                        )
                                found = True
                                break
                        if found:
                            break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting layer notification listener.")
        except Exception as e:
            logger.error("Error in layer notification listener: %s", e)
        finally:
            sock.close()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer:
            if self.receiver_loop.is_running():
                self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
                self._mooncake_receiver_t.join()
            # Stop layer notification listener if running
            if self.layer_wise_mode and hasattr(self, '_layer_notify_t'):
                # The notification listener uses ZMQ, which will be terminated
                # by async_zmq_ctx.term() above, causing the listener to exit.
                self._layer_notify_t.join(timeout=1.0)

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if self.tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = f"This P tp_rank {self.tp_rank} not in remote D target ranks {remote_tp_ranks}"  # noqa: E501
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                # For layer-wise mode, also save D's address info
                if meta.total_layers > 0 and meta.layer_names:
                    # Layer-wise mode: initialize with layer tracking
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id="",
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                        remote_hostname=meta.remote_hostname,
                        remote_port=meta.remote_port,
                        remote_kv_caches_base_addr=meta.kv_caches_base_addr,
                        remote_notify_port=meta.notify_port,
                        total_layers=meta.total_layers,
                        layer_progress={ln: False for ln in meta.layer_names},
                    )
                    logger.debug(
                        "Layer-wise mode: Created send_meta for transfer_id %s "
                        "with %d layers", transfer_id, meta.total_layers
                    )
                else:
                    # Old mode: bulk transfer
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id="",
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
            else:
                # Request already exists (may have been created by record_send_reqs)
                # For layer-wise mode, update with layer tracking info from D
                send_meta = self.reqs_need_send[transfer_id]
                if meta.total_layers > 0 and meta.layer_names:
                    if send_meta.total_layers == 0:
                        # Update with layer tracking info
                        send_meta.remote_hostname = meta.remote_hostname
                        send_meta.remote_port = meta.remote_port
                        send_meta.remote_kv_caches_base_addr = meta.kv_caches_base_addr
                        send_meta.remote_notify_port = meta.notify_port
                        send_meta.total_layers = meta.total_layers
                        send_meta.layer_progress = {ln: False for ln in meta.layer_names}
                        logger.debug(
                            "Layer-wise mode: Updated send_meta for transfer_id %s "
                            "with %d layers", transfer_id, meta.total_layers
                        )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            src_ptrs, dst_ptrs, lengths, err_reqs = await self._build_transfer_params(
                ready_reqs, meta
            )

            if err_reqs:
                response = MooncakeXferResponse(
                    status=response_status,
                    err_reqs=err_reqs,
                    err_msg="P num blocks less than D",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    err_reqs = []
                    for d_req_id, send_meta in ready_reqs:
                        send_meta.sending -= 1
                        err_reqs.append(d_req_id)
                    # Do best effort to transfer the remaining reqs.
                    response = MooncakeXferResponse(
                        status=response_status,
                        err_reqs=err_reqs,
                        err_msg=f"Mooncake transfer engine returned {ret_value}",
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(response))
                    )
                    continue

            for d_req_id, send_meta in ready_reqs:
                # TODO: for heterogeneous TP (one P pairs to multiple D),
                # we need to check whether all headers are sent.
                # If not, we should set expire_time to normal and skip the below.
                send_meta.sending -= 1
                send_meta.sent += 1
                if send_meta.sent == send_meta.need_send:
                    del self.reqs_need_send[send_meta.transfer_id]
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ready_reqs],
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        if send_meta.need_send != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId]]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(d_req_id)
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs

    async def _send_layer_kv_push(
        self,
        send_meta: SendBlockMeta,
        layer_name: str,
        kv_layer: torch.Tensor,
        block_ids: list[int],
    ) -> bool:
        """
        Push mode: Send single layer KV to Consumer.
        D's address is already in send_meta (obtained from ZMQ message in start_load_kv).

        Args:
            send_meta: Send metadata (contains D's address and base_addr)
            layer_name: Layer name
            kv_layer: KV cache tensor for this layer
            block_ids: Block IDs to transfer

        Returns:
            bool: Whether transfer succeeded
        """
        # 1. Calculate layer index
        layer_names = list(self.device_kv_caches.keys())
        if layer_name not in layer_names:
            logger.error("Layer %s not found in device_kv_caches", layer_name)
            return False
        layer_idx = layer_names.index(layer_name)

        # 2. Build transfer params (only this layer)
        src_ptrs, dst_ptrs, lengths = self._build_layer_transfer_params_push(
            send_meta, layer_idx, block_ids
        )

        if not src_ptrs:
            return False

        # 3. Wait for CUDA computation to complete # TODO wdb: 这里是必要的吗？
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        event.record(stream)
        event.synchronize()  # Ensure layer computation is done

        # 4. Execute Mooncake transfer
        remote_session = f"{send_meta.remote_hostname}:{send_meta.remote_port}"

        ret_value = await self.sender_loop.run_in_executor(
            self._sender_executor,
            self._send_blocks,
            remote_session,
            src_ptrs,
            dst_ptrs,
            lengths,
        )

        if ret_value != 0:
            logger.error(
                "Failed to send layer %s to %s: ret=%d",
                layer_name, remote_session, ret_value
            )
            return False

        # 5. Send notification to Consumer that layer is ready
        if send_meta.remote_notify_port > 0:
            try:
                notify_addr = f"tcp://{send_meta.remote_hostname}:{send_meta.remote_notify_port}"
                notify_sock = self.async_zmq_ctx.socket(zmq.PUSH)
                notify_sock.connect(notify_addr)
                notify_sock.send_json({
                    "transfer_id": send_meta.transfer_id,
                    "layer_name": layer_name,
                })
                notify_sock.close()
                logger.debug(
                    "Sent layer completion notification for %s to %s",
                    layer_name, notify_addr
                )
            except Exception as e:
                logger.warning("Failed to send layer notification: %s", e)
                # Continue even if notification fails - data is already transferred

        return True

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        return ret_value

    def _build_layer_transfer_params_push(
        self,
        send_meta: SendBlockMeta,
        layer_idx: int,
        block_ids: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Build transfer parameters for a single layer.

        Args:
            send_meta: Send metadata containing remote address info
            layer_idx: Index of the layer to transfer
            block_ids: Local block IDs to transfer

        Returns:
            Tuple of (src_ptrs, dst_ptrs, lengths)
        """
        src_ptrs = []
        dst_ptrs = []
        lengths = []

        if not block_ids:
            return src_ptrs, dst_ptrs, lengths

        # Get base addresses for this layer
        if layer_idx >= len(self.kv_caches_base_addr):
            logger.error("Layer index %d out of range", layer_idx)
            return src_ptrs, dst_ptrs, lengths

        local_base_addr = self.kv_caches_base_addr[layer_idx:layer_idx+1]
        remote_base_addr = send_meta.remote_kv_caches_base_addr[layer_idx:layer_idx+1] if send_meta.remote_kv_caches_base_addr else []

        if not remote_base_addr:
            logger.error("No remote base address for layer %d", layer_idx)
            return src_ptrs, dst_ptrs, lengths

        block_len = self.block_len

        # Group contiguous blocks
        group_local_block_ids = []
        if len(block_ids) > 0:
            start = block_ids[0]
            curr_group = [start]
            for i in range(1, len(block_ids)):
                if block_ids[i] == block_ids[i-1] + 1:
                    curr_group.append(block_ids[i])
                else:
                    group_local_block_ids.append(curr_group)
                    curr_group = [block_ids[i]]
            group_local_block_ids.append(curr_group)

        # For layer-wise transfer, remote and local block IDs are the same
        for group in group_local_block_ids:
            for local_layer_addr, remote_layer_addr in zip(local_base_addr, remote_base_addr):
                src_ptrs.append(local_layer_addr + group[0] * block_len)
                dst_ptrs.append(remote_layer_addr + group[0] * block_len)
                lengths.append(block_len * len(group))

        return src_ptrs, dst_ptrs, lengths

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def wait_for_layer_load_async(self, layer_name: str) -> None:
        """
        Push mode: Consumer side waits for specified layer to complete.

        Flow:
        1. Check if this layer is already received (layer_progress[layer_name])
        2. If completed, return immediately
        3. If not completed, create asyncio.Event and wait
        4. Background receive thread will set event when this layer arrives
        """
        for pull_metas in self.reqs_to_recv.values():
            for req_id, pull_meta in pull_metas.items():
                # Check if already completed
                if pull_meta.layer_progress.get(layer_name, False):
                    continue

                # Create wait event for this layer (if not exists)
                if layer_name not in pull_meta.layer_events:
                    pull_meta.layer_events[layer_name] = asyncio.Event()

                # Wait for this layer to arrive
                try:
                    await asyncio.wait_for(
                        pull_meta.layer_events[layer_name].wait(),
                        timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout waiting for layer %s of req %s",
                        layer_name, req_id
                    )
                    raise

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)

        # Build metadata
        meta_kwargs = {
            "remote_hostname": self.hostname,
            "remote_port": self.rpc_port,
            "remote_tp_size": self.tp_size,
            "remote_tp_rank": self.tp_rank,
            "req_blocks": {
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            "kv_caches_base_addr": self.kv_caches_base_addr,
        }

        # For layer-wise mode, include layer information
        if self.layer_wise_mode and self.device_kv_caches:
            layer_names = list(self.device_kv_caches.keys())
            meta_kwargs["total_layers"] = len(layer_names)
            meta_kwargs["layer_names"] = layer_names

            # Wait for notification listener to be ready (if it's starting)
            # This avoids race condition where port is used before being bound
            notify_port = getattr(self, 'layer_notify_port', 0)
            if notify_port == 0 and hasattr(self, '_layer_notify_t'):
                # Wait a bit for the listener to bind the port
                import time
                for _ in range(100):  # Max 1 second wait
                    notify_port = getattr(self, 'layer_notify_port', 0)
                    if notify_port > 0:
                        break
                    time.sleep(0.01)

            meta_kwargs["notify_port"] = notify_port

            # Initialize layer tracking for each pull_meta
            for pull_meta in pull_metas.values():
                pull_meta.total_layers = len(layer_names)
                pull_meta.layer_progress = {ln: False for ln in layer_names}

            logger.debug(
                "Layer-wise mode: Sending metadata with %d layers, notify_port=%d",
                len(layer_names), notify_port
            )

        metadata = MooncakeXferMetadata(**meta_kwargs)

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request. # TODO wdb: when all the layer kv cache is calculated, send zmq msg to trigger kvcache transfer; we send it earlier for layer-wise transfer
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def process_layer_received(self, layer_name: str, pull_metas: dict[ReqId, PullReqMeta]):
        """
        Mark a layer as received for all requests in pull_metas.
        Called when a layer transfer completes (in Push mode).
        """
        for req_id, pull_meta in pull_metas.items():
            if layer_name not in pull_meta.layer_progress:
                pull_meta.layer_progress[layer_name] = True
                pull_meta.completed_layers += 1

                # Signal waiting coroutines
                if layer_name in pull_meta.layer_events:
                    pull_meta.layer_events[layer_name].set()

                logger.debug(
                    "Layer %s received for req %s (%d/%d)",
                    layer_name, req_id, pull_meta.completed_layers, pull_meta.total_layers
                )

                # Check if all layers completed
                if pull_meta.completed_layers >= pull_meta.total_layers:
                    self.finished_recving_reqs.add(pull_meta.d_req_id)
                    logger.info("Finished receiving all layers for req %s", req_id)

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        if count != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] # TODO wdb: call it after each layer forward
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def save_kv_layer_async(
        self,
        metadata: MooncakeConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ):
        """
        Producer side: Send the KV layer immediately after computation completes.
        Push mode: D's address was obtained during start_load_kv, send directly.

        Flow:
        1. Check if this layer's request is in reqs_need_send
        2. Wait for D side ready (ready event)
        3. Wait for layer computation to complete (CUDA synchronize)
        4. Call Mooncake to send this layer
        5. Update layer_progress, clean up if all layers completed
        """
        if not metadata.reqs_to_send:
            return

        for req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if transfer_id not in self.reqs_need_send:
                continue

            send_meta = self.reqs_need_send[transfer_id]

            # 1. Wait for D side ready (first time needs to wait for ZMQ handshake)
            if not send_meta.ready.is_set():
                try:
                    await asyncio.wait_for(
                        send_meta.ready.wait(),
                        timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for D side ready for req %s", req_id)
                    continue

            # 2. Check if this layer already sent (idempotency)
            if send_meta.layer_progress.get(layer_name, False):
                continue

            # 3. Execute layer transfer
            success = await self._send_layer_kv_push(
                send_meta, layer_name, kv_layer, block_ids
            )

            if success:
                # 4. Update progress
                send_meta.layer_progress[layer_name] = True
                send_meta.completed_layers += 1

                logger.debug(
                    "Sent layer %s for req %s (%d/%d)",
                    layer_name, req_id, send_meta.completed_layers, send_meta.total_layers
                )

                # 5. Check if all layers completed
                if send_meta.completed_layers >= send_meta.total_layers:
                    del self.reqs_need_send[transfer_id]
                    self.finished_sending_reqs.add(send_meta.p_req_id)
                    logger.info("Finished sending all layers for req %s", req_id)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    # In layer-wise mode, we need to wait for D's metadata
                    # which contains layer tracking info, so we create a
                    # placeholder here that will be filled by send_kv_to_decode
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
