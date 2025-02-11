from simplellm.llama import CausalLLama, LLama # get our models
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
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 1
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=rank*3000) # skip so we can have different things
# we can iterate the dataset with:
iter_ds = iter(ds)

optim = Adam(net.parameters(),lr=8e-4)

sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

for itr in range(5_000):
    optim.zero_grad()
    x = next(iter_ds)
    target = x.clone().detach()
    x = x.to(device)
    
    x = net(x)
    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
    # log the loss:
    print(itr,loss.item())
    loss.backward()
    
    dist.barrier() # wait for everyone
    
    tmp = []
    for param in net.parameters():
        if param.grad == None:
            tmp.append(torch.zeros_like(param,device="cpu").view(-1))                      
            continue
        tmp.append(param.grad.view(-1))
        param.grad = None
    prev_grad = torch.cat(tmp).to("cpu")
    dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
    tmp = torch.split(prev_grad, len_sizes)
    for i, param in enumerate(net.parameters()):
        param.grad = tmp[i].view(sizes[i]).to(device)/world_size # average
    optim.step()
    torch.cuda.empty_cache()



