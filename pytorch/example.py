import os
import sys
import torch
import torch.distributed as dist
from random import randint
import traceback
import datetime
import socket

def run(rank, world_size):
    steps = 5
    for step in range(1, steps + 1):
        value = randint(0, 10)

        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)

        tensor = torch.tensor(value, dtype=torch.int)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        print(f"rank: {rank}, step: {step}, value: {value}, reduced sum: {tensor.item()}")

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "porsche"
    os.environ["MASTER_PORT"] = "17171"

    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method="tcp://porsche:23456", timeout=datetime.timedelta(weeks=120))

    torch.manual_seed(42)

if __name__ == "__main__":
    try:
        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname() + ": Setup Completed")
        run(int(sys.argv[1]), int(sys.argv[2]))
    except Exception as e:
        traceback.print_exc()