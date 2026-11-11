"""Fixed-seed diffusion sampler batched over paths, same 20-frame state."""
import torch

@torch.no_grad()
def sample_batched(model,h,seeds):
    k=len(seeds);b=len(h)
    noises=[]
    for seed in seeds:
        gen=torch.Generator(device=h.device).manual_seed(seed)
        noises.append(torch.randn(h.shape,device=h.device,dtype=h.dtype,generator=gen))
    x=torch.stack(noises).reshape(k*b,*h.shape[1:])
    history=h.unsqueeze(0).expand(k,*h.shape).reshape(k*b,*h.shape[1:])
    for t in reversed(range(4)):
        alpha=model.abar[t]
        labels=torch.full((len(history),),t*33,device=h.device,dtype=torch.long)
        clean=model.net(x,labels,history)
        noise=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        previous=model.abar[t-1] if t else x.new_tensor(1.)
        x=previous.sqrt()*clean+(1-previous).sqrt()*noise
    return x.reshape(k,b,*h.shape[1:])
