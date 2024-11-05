from queue import Empty
from typing import List
from dataclasses import dataclass
import threading
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing import Process
# @dataclass
# class BaseRequest(ABC):
#     """
#     Abstract class for implementing Request data type
#     """

class Parallel(ABC):
    """
    Abstract class to implement manager for managing multiple-thread process
    """

    def __init__(self, worker_type="thread", worker_num=10):
        if worker_type == "thread":
            from queue import Queue, Event
        else:
            from multiprocessing import Queue, Event
        self.request_queue = Queue()
        self.return_queue = Queue()
        self.stop_event = Event()
        self.workers: List = [] # a list of thread objects
        self.verbose_level = 2 # 0 for no printing
                               # 1 for some important printing
                               # 2 for no printing
        self.worker_type = worker_type
        assert worker_type in ["thread", "process"]
        self.worker_num = worker_num

    @abstractmethod
    def parallel_func(self, input)->None:
        """ the function that a thread need to work on given a request

        input:
            :param request: a request object
        """
        pass

    def _worker(self):
        """ a thread function running in the loop
        """

        while not self.stop_event.is_set():
            try:
                request = self.request_queue.get(block=False)
                return_msg = self.parallel_func(request)
                self.return_queue.put(return_msg)

            except Empty:
                continue

            except Exception as e:
                if self.verbose_level>1:
                    print(e)
            # try:
            #     is_break = self.close_queue.get(block=False)
            #     if is_break:
            #         break
            #     break
            # except Empty:
            #     continue

            # except Exception as e:
            #     if self.verbose_level>1:
            #         print(e)


    def run(self, input_list):
        """ main run function for manager, we will show a basic workflow example
        """
        print("Initialize {} workers..".format( self.worker_num))

        # intialize process

        for index in range( self.worker_num):
            if self.worker_type == "thread":
                thread = threading.Thread(target=self._worker)
                self.workers.append(thread)
                thread.start()
            elif self.worker_type == "process":
                p = Process(target=self._worker)
                self.workers.append(p)
                p.start()


        for input in input_list:
            self.request_queue.put(input)

        # wait until the workers process all the request
        self.wait_for_workers()
        return_list = []
        print("get return list")
        for _ in range(len(input_list)):
            return_list.append(self.return_queue.get())
        return return_list
    
    def wait_for_workers(self):
        """
        A function that waits until the workers process all the request. A progress bar will show
        """
        total = self.request_queue.qsize()
        pbar = tqdm(total=total)
        count = 0

        while not self.request_queue.empty():
            if count != total - self.request_queue.qsize():
                delta = total - self.request_queue.qsize() - count
                count = total - self.request_queue.qsize()
                pbar.update(delta) # display the progress bar

            if self.request_queue.empty():
                break
        pbar.close()

        print("thread finish")


    def close(self):
        """close the manager including thread process
        """


        # break the thread function loop
        self.stop_event.set()

        print("wait worker join...")
        # wait for workers exit
        for w in self.workers:
            w.join()

        # destroy all thread objects
        for w in self.workers:
            del(w)
        print("close data manager")