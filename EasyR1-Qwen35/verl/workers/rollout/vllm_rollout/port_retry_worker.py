"""CUDA worker with collision recovery for single-rank rollout rendezvous."""
from vllm.v1.worker.gpu_worker import Worker

from verl.workers.rollout.vllm_rollout.port_retry import retry_single_rank_store_port


class PortRetryWorker(Worker):
    def init_device(self):
        with retry_single_rank_store_port():
            return super().init_device()
