"""Retry single-rank rendezvous without a probe/close/bind port race."""
import importlib
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def retry_single_rank_store_port():
    """Only retry EADDRINUSE for a locally hosted one-rank TCPStore.

    A one-rank store has no remote clients, so port 0 safely asks the OS to
    bind a free port atomically. Multi-rank and agent-hosted stores must keep
    their agreed endpoint. Used only during worker device initialization.
    """
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    original = rendezvous._create_c10d_store

    def create_store(hostname, port, rank, world_size, timeout, use_libuv=True):
        try:
            return original(hostname, port, rank, world_size, timeout, use_libuv)
        except RuntimeError as exc:
            occupied = "EADDRINUSE" in str(exc) or "address already in use" in str(exc).lower()
            if (
                not occupied
                or rank != 0
                or world_size != 1
                or rendezvous._torchelastic_use_agent_store()
            ):
                raise
            logger.warning(
                "Single-rank vLLM rendezvous port %s occupied; retrying with an "
                "OS-assigned port (port=0).", port
            )
            return original(hostname, 0, rank, world_size, timeout, use_libuv)

    rendezvous._create_c10d_store = create_store
    try:
        yield
    finally:
        rendezvous._create_c10d_store = original
