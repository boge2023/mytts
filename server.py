import asyncio
import multiprocessing
import os
import queue
import threading
import argparse
import time

from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import run
from typing import Type

from server_process_job import ProcessJobBase, ProcessJobVITSApi
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=36901, help="The port to listen on")
parser.add_argument("--host", default="0.0.0.0", help="The host address to bind to")
parser.add_argument("--max-workers", type=int, default=1, help="worker num")
parser.add_argument("--config", type=str, default="./checkpoint/config.json", help='JSON file for configuration')
parser.add_argument("--model", type=str, default="./checkpoint/G.pth", required=True, help='Model name')
parser.add_argument("--api-prefix", type=str, default="tts", required=True, help='Model name')
parser.add_argument("--thread-num-per-worker", type=int, default=3, help="thread num per worker")
args = parser.parse_args()


class ProcessPoolBase(ProcessPoolExecutor):
    def __init__(self,
                 pool_size: int = 2,
                 pool_start_method: str = "spawn",
                 initializer: Type[ProcessJobBase] = ProcessJobBase,
                 kwargs: dict = {},
                 ):
        
        print(f"mp_context: {pool_start_method}")
        mp_context = multiprocessing.get_context(pool_start_method)
        print(f"mp_context: {mp_context}")
        super().__init__(
            max_workers=pool_size,
            mp_context=mp_context,
            initializer=initializer,
            initargs=(kwargs,),
        )
        self.job_class: Type[ProcessJobBase] = initializer


class ResponseBase(BaseModel):
    code: int
    message: str = None


class ResponseWave(ResponseBase):
    sample_rate: int = None
    data: list = None


class ResponseFile(ResponseBase):
    data: bytes = None


class ResponseSpkList(ResponseBase):
    speakers: list = None


class RequestT2S(BaseModel):
    text: str
    speaker_id: str
    speed: float


def task_wrapper(func: callable, process_pool: ProcessPoolBase, *_args):
    return process_pool.submit(func, *_args).result()


def create_app_core(process_pool: ProcessPoolBase, args):
    fast_api = FastAPI()
    fast_api.debug = True
    thread_pool_for_fetch_data = ThreadPoolExecutor(max_workers=32, thread_name_prefix="fast_api_fetch_result")
    pipe_pair_queue = queue.Queue()
    for i in range(32):
        pipe_pair_queue.put(multiprocessing.Pipe())
    
    async def submit_task_wait_result(task_func, *_args):
        if pipe_pair_queue.qsize() == 0:
            pipe_pair = multiprocessing.Pipe()
        else:
            pipe_pair = pipe_pair_queue.get()
        process_pool.submit(ProcessJobVITSApi.run_task_by_thread_pool, task_func, pipe_pair[1], *_args)
        result = await asyncio.get_event_loop().run_in_executor(thread_pool_for_fetch_data, pipe_pair[0].recv)
        pipe_pair_queue.put(pipe_pair)
        return result
    @fast_api.get(f"{args.api_prefix}/speakers", response_model=ResponseSpkList)
    async def fast_api_get_speakers():
        return await asyncio.get_event_loop().run_in_executor(process_pool, ProcessJobVITSApi.api_get_speakers)
    
    @fast_api.post(f"{args.api_prefix}/t2s", response_model=ResponseWave)
    async def fast_api_t2s(req: RequestT2S):
        return await submit_task_wait_result(ProcessJobVITSApi.api_t2s, req.text, req.speaker_id, req.speed)
    
    @fast_api.post(f"{args.api_prefix}/t2s_bin", response_model=ResponseFile)
    async def fast_api_t2s_bin(req: RequestT2S):
        return await submit_task_wait_result(ProcessJobVITSApi.api_t2s_bin, req.text, req.speaker_id, req.speed)
    
    print(f"create app pid={os.getpid()}, thread_id={threading.get_ident()}, api_prefix={args.api_prefix}")

    return fast_api


def create_app():
    global args
    print(f"create app args={args}, pid={os.getpid()}, thread_id={threading.get_ident()}")
    process_pool_init_args = {
        "args": args
    }
    process_pool = ProcessPoolBase(
        pool_size=args.max_workers,
        initializer=ProcessJobVITSApi,
        kwargs=process_pool_init_args,
    )
    
    fast_api = create_app_core(process_pool=process_pool, args=args)
    
    for _ in process_pool.map(time.sleep, [0.5] * args.max_workers):
        pass
    return fast_api


if __name__ == "__main__":
    print(args)
    run("server:create_app", host=args.host, port=args.port, workers=1, factory=True)
