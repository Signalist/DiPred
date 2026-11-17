from risk import train
from evaluate_risk import run
from common import *
def main():
    dump(OUT/'curve_protocol.json',dict(seed=42,arms=['F','P'],widths=[8,16,32],label_fractions=[.25,.5,1.],steps=2000,selection='nested seed42 permutation of train trajectory identities',normalization='same all-train unlabeled feature statistics; curve is labeled-trajectory efficiency'))
    for tag,width,fraction in [('width8',8,1.),('width32',32,1.),('labels25',16,.25),('labels50',16,.5)]:
        train(('F','P'),width,2000,tag,fraction);run(tag)
if __name__=='__main__':setup();main()
