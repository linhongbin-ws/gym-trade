from queue import Empty as QEmpty
import multiprocessing as mp
from typing import Any, Generator
from copy import deepcopy as cp
from tqdm import tqdm
from abc import abstractmethod
class Data:
    def __init__(self, **kwargs):
        self.data = cp(kwargs)


class MPServer:
    """recieve a bt request, and return a result"""

    def __init__(
        self,
        n_workers: int = 2,
    ):
        self._n_workers = n_workers
        assert self._n_workers > 1, "n_workers must be greater than 1"

        
        ctx = mp.get_context("spawn")  # 跨平台更稳（Windows/macOS 必须 spawn）
        self._request_queue = ctx.Queue(maxsize=self.n_workers * 4)
        self._result_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        self.procs = [
            ctx.Process(
                target=bt_server_loop,
                args=(
                    policy_name,
                    policy_args,
                    env_args,
                    self._stop_event,
                    self._request_queue,
                    self._result_queue,
                ),
            )
            for _ in range(self.n_workers)
        ]
        for p in self.procs:
            # p.daemon = True
            p.start()

        self._closed = False

    @property
    def n_workers(self):
        return self._n_workers

    def shutdown(self, join_timeout: float = 5.0):

        if self._n_workers > 1:
            self._stop_event.set()

            # wait for graceful exit
            for p in self.procs:
                p.join(join_timeout)

            # force kill remaining
            for p in self.procs:
                if p.is_alive():
                    p.terminate()
                    p.join()

            # NOW it's safe to close queues
            self._request_queue.close()
            self._result_queue.close()
            self._request_queue.join_thread()
            self._result_queue.join_thread()

    # 让 with BTServer(...) 自动清理
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False
    
    def get_input_generator(self, inputs: dict[str, dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        def generator():
            for k, v in inputs.items():
                yield k,v, False
            while True:
                yield None, None, True
        return generator

    @abstractmethod
    @staticmethod
    def worker_func(self,request: Data) -> Data:
        return request
    
    @staticmethod
    def _worker_loop(
     self,stop_event, request_queue, result_queue
    ):
        while not stop_event.is_set():
            request = None
            try:
                request = request_queue.get(timeout=0.5)  # 定期醒来检查 stop_event
            except QEmpty:
                continue
            if request is not None:
                result = self.worker_func(request)
                result_queue.put(result)

    def run(self, inputs: dict[str, Data]) -> dict[str, Data]:
        
        in_gen = self.get_input_generator(inputs)
        pbar = tqdm(total=len(inputs), desc="Running")
        result_dict = {}
        stop_gen = False
        while(
                not self._stop_event.is_set() 
                and pbar.n < len(inputs)
            ):
            if not self._request_queue.full() and not stop_gen:
                key, value, stop_gen = next(in_gen)
                input_data = Data(key=key, value=value)
                self._request_queue.put(input_data)
                
            if not self._result_queue.empty():
                try:
                    result_data = self._result_queue.get(
                        timeout=0.01
                    )  # 定期醒来检查 stop_event
          
                    result_dict[result_data.data['key']] = result_data.data['value']
                    pbar.update(1)
                except QEmpty:
                    pass

    
if __name__ == "__main__":
    server = MPServer(n_workers=2)
    out = server.run(inputs={
        "input1": Data(value=1),
        "input2": Data(value=2),
    })