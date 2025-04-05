from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 6 # 2 pipelines with 3 ranks each
os.environ["MASTER_PORT"] = "29501"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // 3
seq_l = 256
batch_size = 6 # batch size across all ranks
pipeline_size = 6 // 2 # batch size for each pipeline
microbatch_size = batch_size // world_size  # split batch into microbatches
device = "cuda" if torch.cuda.is_available() else "cpu"


# group ranks for each pipeline
pipeline1 = dist.new_group(ranks=[0, 1, 2])
pipeline2 = dist.new_group(ranks=[3, 4, 5])

# group first stage of each pipeline to enable data parallelism
# (synchronize gradients between the pipelines during training)
data_parallel_group = dist.new_group([0, 3])

# assign each rank to a pipeline
if rank in [0, 1, 2]:
    group = pipeline1
    pipeline_rank = rank
elif rank in [3, 4, 5]:
    group = pipeline2
    pipeline_rank = rank

# make the tokenizer
if rank in [0, 2, 3, 5]:  # only ranks with first/last stages need tokenizer
    tokenizer = SPTokenizer()

# make the model
if rank in [0,3]:
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    # apply skip to ensure each pipeline processes different data
    ds = TinyStories(tokenizer, batch_size=pipeline_size, seq_l=seq_l, skip=0 if rank == 0 else rank*5000)
    iter_ds = iter(ds)
elif rank in [1,4]:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank in [2,5]:
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    # apply skip to ensure each pipeline processes different data
    ds = TinyStories(tokenizer, batch_size=pipeline_size, seq_l=seq_l, skip=0 if rank == 2 else rank*5000)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

# pre-allocate tensors
inp_batch = torch.empty((microbatch_size, seq_l, dmodel), device=device)
inp_grad = torch.empty((microbatch_size, seq_l, dmodel), device=device)

# train model
for itr in range(5_000):
    optim.zero_grad()

    # FORWARD PASS
    if rank in [0,3]:
        out_full = next(iter_ds).to(device) # get next full batch from dataset
        out_full = net.embed(out_full) # embed full batch

        # split full batch into microbatches and send async to rank 1 or 4
        for i in range(0, out_full.size(0), microbatch_size):
            # get current microbatch
            out = out_full[i:i + microbatch_size]
            # send microbatch to rank 1 or 4, depending on the pipeline
            req = dist.isend(out.to("cpu"), 1 if rank == 0 else 4, tag=itr, group=group)
            # wait for send to complete
            req.wait()
        
    elif rank in [1,4]:
        # handle incoming microbatches and send output to rank 2 or 5 async
        for i in range(0, pipeline_size, microbatch_size):
            # receive embedded microbatch from rank 0 or 3, depending on the pipeline
            req = dist.irecv(inp_batch, 0 if rank == 1 else 3, tag=itr, group=group)
            # wait for received to complete
            req.wait()

            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_() # enable gradient tracking
            inp_batch.retain_grad() # get gradient

            out = net(inp_batch) # forward pass

            # send microbatch to rank 2 or 5, depending on the pipeline
            req = dist.isend(out.to("cpu"), 2 if rank == 1 else 5, tag=itr, group=group)
            # wait for send to complete
            req.wait()

    elif rank in [2,5]:
        target = next(iter_ds).to(device)
        # handle incoming microbatches and compute loss for each 
        for i in range(0, pipeline_size, microbatch_size):
            # receive microbatch from rank 1 or 4, depending on the pipeline
            req = dist.irecv(inp_batch, 1 if rank == 2 else 4, tag=itr, group=group)
            # wait for received to complete
            req.wait()

            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_() # enable gradient tracking
            inp_batch.retain_grad() # get gradient
            
            # get target data for each microbatch to calculate loss
            target_microbatch = target[i:i + microbatch_size]
            logits = net(inp_batch) # forward pass
            loss = causalLLMLoss(logits, target_microbatch, tokenizer.vocab_size)
            print(loss.item())
            # backpropagate loss to compute gradients
            loss.backward()

    # BACKWARD PASS:
    if rank in [2,5]:
        # send gradients async to rank 1 or 4, depending on the pipeline
        req = dist.isend(inp_batch.grad.to("cpu"), 1 if rank == 2 else 4, tag=itr, group=group)
        # wait for send to complete
        req.wait()

    elif rank in [1,4]:
        # receive gradients async from rank 2 or 5, depending on the pipeline
        req = dist.irecv(inp_grad, 2 if rank == 1 else 5, tag=itr, group=group)
        # wait for received to complete
        req.wait()

        out.backward(inp_grad.to(device)) # backpropagation

        # send gradients async to rank 0 or 3, depending on the pipeline
        req = dist.isend(inp_batch.grad.to("cpu"), 0 if rank == 1 else 3, tag=itr, group=group)
        # wait for send to complete
        req.wait()

    elif rank in [0,3]:
        # receive gradients async from rank 1 or 4, depending on the pipeline
        req = dist.irecv(inp_grad, 1 if rank == 0 else 4, tag=itr, group=group)
        # wait for received to complete
        req.wait()

        out.backward(inp_grad.to(device))

   # synchronize all ranks in both pipelines before optimization step
    dist.barrier()

    # average gradient across pipelines after backward pass is done
    if rank in [0,3]:
        for param in net.parameters():
            if param.grad is not None:
                # sum up all gradients across ranks 0 and 3 (first stages of both pipelines)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=data_parallel_group)
                param.grad /= 2  # average gradients across the two pipelines
        
    optim.step()
    torch.cuda.empty_cache() if device == "cuda" else None