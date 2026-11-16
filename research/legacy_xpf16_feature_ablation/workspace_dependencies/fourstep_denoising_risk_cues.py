"""Risk cues from the actual native four-call inference, never future labels."""

import torch

from learned_feedback_risk import FeedbackScaleAdapter,adapter_inputs


@torch.no_grad()
def forecast_last_change(model,history):
    outputs,labels=[],[]
    def capture(module,args,result):
        labels.append(int(args[1][0]))
        outputs.append(result.detach().clone())
    hook=model.net.register_forward_hook(capture)
    try:
        point=model.rollout(history,500000,sign=0)
    finally:
        hook.remove()
    assert labels==[99,66,33,0] and len(outputs)==4
    assert torch.equal(point,outputs[-1])
    # These are two clean-state estimates inside the SAME four-call rollout.
    # This is not an ensemble variance or a replacement three-call forecast.
    return point,(outputs[-1]-outputs[-2]).abs(),labels


def cue_inputs(history,point,lower,upper,change,cue,available):
    value=adapter_inputs(history,point,lower,upper,None)
    value[:,13:14]=float(available)
    width=(0.5*(upper-lower)).clamp_min(1e-6)
    if cue=="history_change":
        signal=(history[:,:,-1:]-history[:,:,-2:-1]).abs().expand_as(point)
    elif cue=="denoising_change":
        signal=change
    elif cue=="none":
        signal=None
    else:
        raise ValueError("Choose none,history_change,denoising_change.")
    if signal is not None:
        value[:,10:12]=torch.log1p(signal/width).clamp(0,5)
    return value


class DenoisingCueAdapter(FeedbackScaleAdapter):
    def __init__(self):
        # Reuse the two reserved input slots; they now hold computation cues,
        # not previous errors. Parameter count stays exactly13090.
        super().__init__(use_memory=True)
