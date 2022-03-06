import torch
import torch.distributed as dist
from utils import init_dist_process_group

local_rank, local_size, world_rank, world_size = init_dist_process_group(backend='nccl')
assert local_size <= torch.cuda.device_count()
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()
is_distributed = world_size > 0
print(f'device: {device} local_rank: {local_rank}/{local_size} world_rank: {world_rank}/{world_size}')


tensor = torch.ones(3) * world_rank
tensor = tensor.to(device)
group = dist.new_group()
if world_rank % 2 == 0:
    print(world_rank, 'send', tensor)
    dist.send(tensor, dst=world_rank+1, tag=0)
    tensor += 5
    print(world_rank, 'send', tensor)
    dist.send(tensor, dst=world_rank+1, tag=1)
elif world_rank % 2 == 1:
    dist.recv(tensor, src=world_rank-1, tag=2)
    print(world_rank, 'recv', tensor)
    dist.recv(tensor, src=world_rank - 1, tag=3)
    print(world_rank, 'recv', tensor)
