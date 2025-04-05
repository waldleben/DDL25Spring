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
world_size = 3
os.environ["MASTER_PORT"] = "29501"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
microbatch_size = batch_size // world_size  # split batch into microbatches
device = "cuda" if torch.cuda.is_available() else "cpu"

# make the tokenizer
if rank == 0 or rank == 2:
    tokenizer = SPTokenizer()

# make the model
if rank == 0:
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 2:
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

# pre-allocate tensors
inp_batch = torch.empty((microbatch_size, seq_l, dmodel), device=device)
inp_grad = torch.empty((microbatch_size, seq_l, dmodel), device=device)

# train model using pipeline parallelism with microbatches
for itr in range(5_000):
    optim.zero_grad()

    # FORWARD PASS:
    if rank == 0:
        out_full = next(iter_ds).to(device) # get next full batch from dataset
        out_full = net.embed(out_full) # embed full batch

        # split full batch into microbatches and send async to rank 1
        for i in range(0, out_full.size(0), microbatch_size):
            # get current microbatch
            out = out_full[i:i + microbatch_size]
            # send microbatch to rank 1
            req = dist.isend(out.to("cpu"), 1, tag=itr)
            # wait for send to complete
            req.wait()
        
    elif rank == 1:
        # handle incoming microbatches and send output to rank 2 async
        for i in range(0, batch_size, microbatch_size):
            # receive embedded microbatch from rank 0
            req = dist.irecv(inp_batch, 0, tag=itr)
            # wait for received to complete
            req.wait()

            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_() # enable gradient tracking
            inp_batch.retain_grad() # get gradient

            out = net(inp_batch) # forward pass

            # send output to rank 2
            req = dist.isend(out.to("cpu"), 2, tag=itr)
            # wait for send to complete
            req.wait()

    elif rank == 2:
        target = next(iter_ds).to(device)
        # handle incoming microbatches and compute loss for each 
        for i in range(0, batch_size, microbatch_size):
            # receive microbatch from rank 1
            req = dist.irecv(inp_batch, 1, tag=itr)
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
    if rank == 2:
        # send gradients async to rank 1
        req = dist.isend(inp_batch.grad.to("cpu"), 1, tag=itr)
        # wait for send to complete
        req.wait()

    elif rank == 1:
        # receive gradients async from rank 2
        req = dist.irecv(inp_grad, 2, tag=itr)
        # wait for received to complete
        req.wait()

        out.backward(inp_grad.to(device)) # backpropagation

        # send gradients async to rank 0
        req = dist.isend(inp_batch.grad.to("cpu"), 0, tag=itr)
        # wait for send to complete
        req.wait()

    elif rank == 0:
        # receive gradients async from rank 1
        req = dist.irecv(inp_grad, 1, tag=itr)
        # wait for received to complete
        req.wait()

        out.backward(inp_grad.to(device)) # backpropagation

    # synchronize all ranks before optimization step
    dist.barrier()

    optim.step()
    torch.cuda.empty_cache() if device == "cuda" else None