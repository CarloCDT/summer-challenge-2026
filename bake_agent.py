#!/usr/bin/env python3
"""Bake a trained checkpoint into a single self-contained CodinGame submission file.

The generated .py imports only sys/numpy/base64/zlib - no torch - and carries the network
weights inline as a base85 blob. It reimplements RailroadUNet's forward pass in numpy and
reproduces the exact observation encoding, action masks and sub-turn decision order that
training/simulator.py uses, so the submitted bot plays the same policy it was trained as.

Three things shrink the payload, in the order they matter:
  1. The value head is dropped. PPO's critic is training-only; only the policy head plays.
  2. BatchNorm is folded into the preceding convolution. Exact, not an approximation - it
     removes every BN parameter and takes the runtime down to conv + relu.
  3. Weights are quantized per output channel. int4 is the default and the only mode that fits
     the cap at a useful size; it costs no score despite a low --verify agreement rate, which is
     a misleading metric here - see quantize() for the measurements.

    python3 bake_agent.py checkpoints/<run>-distill_iterN.pt -o submission.py
    python3 bake_agent.py checkpoints/... -o submission.py --verify 20
    python3 bake_agent.py checkpoints/... -o submission.py --quant int8   # research only, too big

CodinGame caps source at 100,000 characters; --verify replays N games against the real
environment and reports the agreement rate with the torch model.
"""
import argparse
import base64
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

CG_SOURCE_LIMIT = 100_000

# Convolutions in forward order. Each entry is (baked name, conv prefix, bn prefix or None).
LAYERS = [
    ("stem",  "stem.0",             "stem.1"),
    ("e1a",   "enc1.block.0",       "enc1.block.1"),
    ("e1b",   "enc1.block.3",       "enc1.block.4"),
    ("e2a",   "enc2.block.0",       "enc2.block.1"),
    ("e2b",   "enc2.block.3",       "enc2.block.4"),
    ("bna",   "bottleneck.block.0", "bottleneck.block.1"),
    ("bnb",   "bottleneck.block.3", "bottleneck.block.4"),
    ("r2",    "up2_reduce",         None),
    ("d2a",   "dec2.block.0",       "dec2.block.1"),
    ("d2b",   "dec2.block.3",       "dec2.block.4"),
    ("r1",    "up1_reduce",         None),
    ("d1a",   "dec1.block.0",       "dec1.block.1"),
    ("d1b",   "dec1.block.3",       "dec1.block.4"),
    ("head",  "head",               None),
]


def fold_batchnorm(sd):
    """Conv->BN collapsed into one biased conv. BN in eval mode is an affine map per channel,
    so it composes exactly with the convolution that precedes it - no accuracy is lost."""
    baked = []
    for name, conv, bn in LAYERS:
        w = sd[f"{conv}.weight"].double().numpy()
        b = sd[f"{conv}.bias"].double().numpy()
        if bn is not None:
            gamma = sd[f"{bn}.weight"].double().numpy()
            beta = sd[f"{bn}.bias"].double().numpy()
            mean = sd[f"{bn}.running_mean"].double().numpy()
            var = sd[f"{bn}.running_var"].double().numpy()
            scale = gamma / np.sqrt(var + 1e-5)
            w = w * scale[:, None, None, None]
            b = (b - mean) * scale + beta
        baked.append((name, w.astype(np.float32), b.astype(np.float32)))
    return baked


def quantize(baked, mode):
    """Per-output-channel symmetric quantization. Biases stay float32 - there are only a few
    thousand of them and they sit directly on the logits.

    int4 is what a submittable model ships as; it is the only mode that fits the 100k cap at a
    useful size. Do not be alarmed by --verify's agreement rate: int4 reproduces only ~62% of the
    torch model's argmax moves against int8's ~95%, which looks alarming and is not. Measured on
    the distilled student over 40 real games against level2:

        mode   agreement   margin   win     source
        int8      95.4%     +2098   1.00   172,941 chars (over the cap)
        int4      61.9%     +2212   1.00    83,671 chars

    Identical score. The policy picks one of ~1800 cells per decision and most disagreements are
    between near-tied cells, so agreement overstates the damage badly. Judge a quantization mode
    by test_submission.py's margin, never by the agreement rate alone.

    A per-64-weight fp16 scale was tried to buy back that agreement and REJECTED: it scored no
    better, reproduced no more moves (60.9%), and cost 9k extra chars. Group scales only cut the
    weight error to 0.75x, while int8 cuts it to ~1/18th - the loss is the 4-bit resolution
    itself, 15 levels, not outliers skewing a channel's scale. Widening the scale granularity
    cannot fix that; only more bits can, and more bits do not fit."""
    blob = bytearray()
    shapes = []
    for name, w, b in baked:
        co = w.shape[0]
        flat = w.reshape(co, -1)
        if mode == "fp16":
            blob += flat.astype(np.float16).tobytes()
        else:
            qmax = 127 if mode == "int8" else 7
            scale = np.abs(flat).max(axis=1) / qmax
            scale[scale == 0] = 1.0
            q = np.round(flat / scale[:, None]).clip(-qmax, qmax).astype(np.int8)
            blob += scale.astype(np.float32).tobytes()
            if mode == "int8":
                blob += q.tobytes()
            else:
                nib = (q.reshape(-1) & 0x0F).astype(np.uint8)
                if nib.size % 2:
                    nib = np.append(nib, 0)
                blob += (nib[0::2] | (nib[1::2] << 4)).tobytes()
        blob += b.astype(np.float32).tobytes()
        shapes.append((name, w.shape))
    return bytes(blob), shapes


RUNTIME = r'''import sys,base64,zlib,os
# Pin BLAS to one thread BEFORE numpy loads. These GEMMs are far too small to gain from
# threading, and the thread pool's scheduling jitter is what pushes the occasional turn past
# the referee's 50ms deadline. One thread is both faster here and, more importantly, has a
# much tighter worst case - and the deadline is enforced per turn, not on the average.
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np

_Q=%%QUANT%%
_SH=%%SHAPES%%
_W=b"%%BLOB%%"

def _load():
    raw=zlib.decompress(base64.b85decode(_W)); off=0; P={}
    for name,shape in _SH:
        co=shape[0]; n=1
        for d in shape: n*=d
        if _Q=="fp16":
            w=np.frombuffer(raw,np.float16,n,off).astype(np.float32); off+=n*2
        else:
            sc=np.frombuffer(raw,np.float32,co,off).copy(); off+=co*4
            if _Q=="int8":
                q=np.frombuffer(raw,np.int8,n,off).astype(np.float32); off+=n
            else:
                nb=(n+1)//2
                pk=np.frombuffer(raw,np.uint8,nb,off); off+=nb
                lo=(pk&0x0F).astype(np.int8); hi=(pk>>4).astype(np.int8)
                q=np.empty(nb*2,np.int8); q[0::2]=lo; q[1::2]=hi
                q=np.where(q>7,q-16,q).astype(np.float32)[:n]
            w=(q.reshape(co,-1)*sc[:,None]).reshape(-1)
        b=np.frombuffer(raw,np.float32,co,off).copy(); off+=co*4
        w=w.reshape(shape); kh=shape[2]; ci=shape[1]
        # Flatten each filter to a row so a convolution is ONE matmul against an im2col buffer.
        # For 3x3 the row order must be (kh,kw,ci) to match how _conv stacks its 9 shifted
        # copies; doing the transpose here means it costs nothing per turn.
        if kh==1:
            m=np.ascontiguousarray(w.reshape(co,ci))
        else:
            m=np.ascontiguousarray(w.transpose(0,2,3,1)).reshape(co,kh*kh*ci)
        P[name]=(m,b,kh,ci)
    return P
P=_load()

_PB={}
_CB={}
def _conv(x,wb,pad):
    """One matmul per convolution, against an im2col buffer built by 9 slice copies.

    The obvious alternative - a strided (ci,3,3,oh,ow) view reshaped to a matrix - is 90x
    SLOWER here, because numpy materialises that reshape through a scattered elementwise copy.
    Nine contiguous slice assignments into a preallocated buffer cost 0.16ms where that path
    costs 14ms. Measured 11.5x end to end against the previous nine-small-matmuls version
    (17.8ms of convolution per turn down to 1.5ms), which matters because the referee enforces
    a 50ms per-turn deadline.

    Both buffers are cached by shape and reused, so a turn allocates nothing here. The padded
    buffer's border is written once at creation and never touched again - only the interior is
    overwritten - so the zero frame stays valid across turns."""
    w,b,kh,ci=wb; h,ww=x.shape[1],x.shape[2]
    if kh==1:
        return (w@x.reshape(ci,-1)).reshape(-1,h,ww)+b[:,None,None]
    if pad:
        k=(ci,h+2,ww+2)
        q=_PB.get(k)
        if q is None: q=_PB[k]=np.zeros(k,np.float32)
        q[:,1:-1,1:-1]=x; x=q; h+=2; ww+=2
    oh,ow=h-kh+1,ww-kh+1
    ck=(ci,oh,ow)
    c=_CB.get(ck)
    if c is None: c=_CB[ck]=np.empty((kh*kh*ci,oh*ow),np.float32)
    t=0
    for i in range(kh):
        for j in range(kh):
            c[t:t+ci]=x[:,i:i+oh,j:j+ow].reshape(ci,-1); t+=ci
    return (w@c).reshape(-1,oh,ow)+b[:,None,None]

def _relu(x):
    np.maximum(x,0,out=x); return x

def _blk(x,a,b):
    return _relu(_conv(_relu(_conv(x,P[a],1)),P[b],1))

def _pool(x):
    c,h,w=x.shape; h2,w2=h//2,w//2
    v=x[:,:h2*2,:w2*2].reshape(c,h2,2,w2,2)
    return v.max(axis=4).max(axis=2)

def _wm(src,dst):
    """PyTorch bilinear, align_corners=False: src=(i+0.5)*scale-0.5, clamped at 0."""
    s=np.arange(dst,dtype=np.float64)
    s=np.clip((s+0.5)*(src/dst)-0.5,0,None)
    i0=np.floor(s).astype(int); i1=np.minimum(i0+1,src-1); f=(s-i0).astype(np.float32)
    m=np.zeros((dst,src),np.float32)
    r=np.arange(dst)
    np.add.at(m,(r,i0),1-f); np.add.at(m,(r,i1),f)
    return m

_UC={}
def _up(x,oh,ow):
    c,h,w=x.shape
    k=(h,w,oh,ow)
    if k not in _UC: _UC[k]=(_wm(h,oh),_wm(w,ow))
    mh,mw=_UC[k]
    y=(mh@x.reshape(c*h,w).reshape(c,h,w).transpose(1,0,2).reshape(h,-1)).reshape(oh,c,w).transpose(1,0,2)
    return (y.reshape(c*oh,w)@mw.T).reshape(c,oh,ow)

def _fwd(g):
    x=_relu(_conv(g,P["stem"],1))
    s1=_blk(x,"e1a","e1b")
    s2=_blk(_pool(s1),"e2a","e2b")
    bo=_blk(_pool(s2),"bna","bnb")
    u2=_conv(_up(bo,*s2.shape[1:]),P["r2"],0)
    d2=_blk(np.concatenate([u2,s2],0),"d2a","d2b")
    u1=_conv(_up(d2,*s1.shape[1:]),P["r1"],0)
    d1=_blk(np.concatenate([u1,s1],0),"d1a","d1b")
    return _conv(d1,P["head"],1)

# ---------------------------------------------------------------- game constants
BH,BW=20,30
NCH=28
THRESH=4.0
N4=((0,-1),(1,0),(0,1),(-1,0))   # N,E,S,W - the referee's tie-break order
COST=(1,2,3,3)                   # plains, river, mountain, poi

my_id=int(input()); W=int(input()); H=int(input())
zone=np.zeros((H,W),np.int32); typ=np.zeros((H,W),np.int32)
for y in range(H):
    for x in range(W):
        a,b=input().split(); zone[y,x]=int(a); typ[y,x]=int(b)
NZ=int(zone.max())+1
towns=[]
for _ in range(int(input())):
    p=input().split()
    tid=int(p[0]); tx=int(p[1]); ty=int(p[2])
    dc=[] if len(p)<4 or p[3]=="x" else [int(v) for v in p[3].split(",")]
    towns.append((tid,tx,ty,dc))
tpos={t[0]:(t[1],t[2]) for t in towns}

PT=(BH-H)//2; PL=(BW-W)//2
# Towns are NOT type 3 - the generator puts them on ordinary terrain and the referee marks them
# by townId, so type 3 never actually occurs. Take the town cells from the init town list.
IS_TOWN=np.zeros((H,W),bool)
for _t,_x,_y,_d in towns: IS_TOWN[_y,_x]=True
COSTMAP=np.take(np.array(COST,np.int32),typ)
ZSIZE=np.bincount(zone.ravel(),minlength=NZ).astype(np.float32)
TOTAL=float(H*W)
# A region holding a town can never be disrupted (Game.doActions).
ZHASTOWN=np.zeros(NZ,bool)
for _,tx,ty,_dc in towns: ZHASTOWN[zone[ty,tx]]=True

_rc_cache={}
def route_channels(inked_cell):
    """Channels 10-21: per-town shortest open-terrain route to each desired connection,
    endpoints 1.0 and interior 0.5. Blocked only by inked regions, so it is recomputed
    whenever the inked set changes - and cached because it usually does not."""
    key=inked_cell.tobytes()
    hit=_rc_cache.get(key)
    if hit is not None: return hit
    ch=np.zeros((12,H,W),np.float32)
    for tid,tx,ty,dc in towns:
        if tid>=12: continue
        for target in dc:
            if target not in tpos: continue
            gx,gy=tpos[target]
            if (tx,ty)==(gx,gy):
                ch[tid,ty,tx]=1.0; continue
            prev={(tx,ty):None}; q=[(tx,ty)]; qi=0; found=False
            while qi<len(q):
                cx,cy=q[qi]; qi+=1
                if (cx,cy)==(gx,gy): found=True; break
                for dx,dy in N4:
                    nx,ny=cx+dx,cy+dy
                    if nx<0 or ny<0 or nx>=W or ny>=H: continue
                    if (nx,ny) in prev: continue
                    if inked_cell[ny,nx]: continue
                    prev[(nx,ny)]=(cx,cy); q.append((nx,ny))
            if not found: continue
            path=[]; node=(gx,gy)
            while node is not None: path.append(node); node=prev[node]
            path.reverse()
            # Increasing precedence, matching GameState._compute_route_channels exactly: the
            # cells between, then the far town, then this channel's own town. A route can run
            # through a third town, so the later writes must win.
            for px,py in path[1:-1]: ch[tid,py,px]=1.0
            ch[tid,path[-1][1],path[-1][0]]=-0.5
            ch[tid,path[0][1],path[0][0]]=-1.0
    if len(_rc_cache)>64: _rc_cache.clear()
    _rc_cache[key]=ch
    return ch

def observe(track,inst,inked_z,active,score_diff,turn):
    """The 28 channels of GameState.get_observation, centered in the 20x30 canvas with the
    padding marked inked. Channel 3 is the enemy and 4 is us regardless of player index."""
    inked_cell=inked_z[zone]
    o=np.zeros((NCH,BH,BW),np.float32)
    b=np.zeros((NCH,H,W),np.float32)
    for t in range(3): b[t]=(typ==t)
    foe=1-my_id
    b[3]=(track==foe); b[4]=(track==my_id)
    is_foe=(track==foe)|(track==2)
    is_own=(track==my_id)|(track==2)
    fl=zone.ravel()
    fz=np.bincount(fl,weights=is_foe.ravel().astype(np.float64),minlength=NZ)
    oz=np.bincount(fl,weights=is_own.ravel().astype(np.float64),minlength=NZ)
    af=np.bincount(fl,weights=(is_foe&active).ravel().astype(np.float64),minlength=NZ)
    ao=np.bincount(fl,weights=(is_own&active).ravel().astype(np.float64),minlength=NZ)
    # Raw counts, not densities - matches GameState channels 5/6/8/9.
    b[5]=(fz/10.0)[zone]; b[6]=(oz/10.0)[zone]
    b[7]=ZSIZE[zone]/TOTAL
    b[8]=(af/10.0)[zone]; b[9]=(ao/10.0)[zone]
    b[10:22]=route_channels(inked_cell)
    b[22]=inst/THRESH
    b[23]=inked_cell
    b[24]=active
    for _t,_tx,_ty,_dc in towns: b[25,_ty,_tx]=1.0
    # Board-wide scalars, flat across every cell - a convolution is local and has no other way
    # to see them. Written into b, not o, so the off-board padding stays zero exactly as
    # training/encoding.py leaves it.
    b[26]=score_diff/10000.0
    b[27]=turn/100.0
    o[:,PT:PT+H,PL:PL+W]=b
    o[23]=1.0
    o[23,PT:PT+H,PL:PL+W]=inked_cell
    return o

turn=0
while True:
    try:
        my_score=int(input()); foe_score=int(input())
    except EOFError:
        break
    track=np.zeros((H,W),np.int32); inst=np.zeros((H,W),np.float32)
    inked_z=np.zeros(NZ,bool); active=np.zeros((H,W),bool)
    for y in range(H):
        for x in range(W):
            p=input().split()
            track[y,x]=int(p[0]); inst[y,x]=float(p[1])
            if p[2]!="0": inked_z[zone[y,x]]=True
            active[y,x]=(p[3]!="x")

    logits=_fwd(observe(track,inst,inked_z,active,my_score-foe_score,turn))

    # The observation is fixed for the whole turn - training only ever recomputes the MASK
    # between sub-actions - so one forward pass covers every decision below.
    paint=%%PAINT%%; acts=[]
    placeable=(~IS_TOWN)&(track==-1)&(~inked_z[zone])
    pl=logits[0,PT:PT+H,PL:PL+W].copy()
    while paint>0:
        legal=placeable&(COSTMAP<=paint)
        if not legal.any(): break
        m=np.where(legal,pl,-np.inf)
        idx=int(np.argmax(m)); y,x=divmod(idx,W)
        acts.append("PLACE_TRACKS %d %d"%(x,y))
        paint-=int(COSTMAP[y,x]); placeable[y,x]=False

    # Disrupting is optional: channel 2 spans the whole on-board plane and wins by max logit.
    ok=(~inked_z)&(~ZHASTOWN)
    dz=ok[zone]
    if dz.any():
        dl=logits[1,PT:PT+H,PL:PL+W]
        best=int(np.argmax(np.where(dz,dl,-np.inf))); by,bx=divmod(best,W)
        if %%SKIP%% and logits[2,PT:PT+H,PL:PL+W].max()>dl[by,bx]:
            pass
        else:
            acts.append("DISRUPT %d"%zone[by,bx])

    # flush=True is not optional. Python block-buffers stdout when it is a pipe, which is
    # exactly how a referee runs a bot, so an unflushed print can sit in the buffer past the
    # turn deadline. test_submission.py spawns this file with -u, which forces unbuffered
    # output and would hide the problem locally no matter how badly it behaved in the arena.
    print(";".join(acts) if acts else "WAIT", flush=True)
    turn+=1
'''


def build_source(blob, shapes, quant, policy_channels, passive_income, allow_skip=True):
    payload = base64.b85encode(zlib.compress(blob, 9)).decode("ascii")
    shape_src = "[" + ",".join(f'("{n}",{tuple(s)})' for n, s in shapes) + "]"
    src = RUNTIME
    src = src.replace("%%QUANT%%", f'"{quant}"')
    src = src.replace("%%SHAPES%%", shape_src)
    src = src.replace("%%BLOB%%", payload)
    src = src.replace("%%PAINT%%", str(passive_income))
    # A 3-channel head is NOT on its own a licence to skip. When a run trains with
    # force_disrupt the SKIP_DISRUPT channel is masked out of every decision, so it receives
    # exactly zero gradient and its weights stay frozen at whatever the warm start left there,
    # while the features feeding them drift for the whole run. Emitting the skip comparison
    # then lets the shipped agent decline a disrupt on stale logits - the precise failure mode
    # force_disrupt exists to remove. Disruption points are 1/turn and never carry over, so a
    # wrongly declined disrupt is a destroyed resource.
    src = src.replace("%%SKIP%%", "True" if (policy_channels >= 3 and allow_skip) else "False")
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", default="submission.py")
    ap.add_argument("--quant", choices=("int8", "int4", "fp16"), default="int4",
                    help="int4 is the default because it is the only mode that fits the 100k cap at a useful size, and it costs no score (see quantize())")
    ap.add_argument("--verify", type=int, default=0,
                    help="Replay N games and report agreement with the torch model")
    ap.add_argument("--skip-disrupt", dest="skip_disrupt", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Whether the baked agent may decline to disrupt. Defaults to what the "
                         "checkpoint recorded: a run trained with force_disrupt never gave its "
                         "SKIP_DISRUPT channel a gradient, so shipping the skip would run on "
                         "frozen weights. Checkpoints from before that field was recorded fall "
                         "back to allowing it - pass --no-skip-disrupt for those.")
    args = ap.parse_args()

    blob_ckpt = torch.load(args.checkpoint, map_location="cpu")
    kwargs = blob_ckpt["model_kwargs"]
    sd = blob_ckpt["model_state_dict"]
    print(f"{args.checkpoint}: {kwargs}")

    if args.skip_disrupt is None:
        allow_skip = not blob_ckpt.get("force_disrupt", False)
        source = ("checkpoint" if "force_disrupt" in blob_ckpt
                  else "default (checkpoint predates the field)")
    else:
        allow_skip, source = args.skip_disrupt, "--skip-disrupt flag"
    print(f"  SKIP_DISRUPT in the baked agent: {allow_skip}  [{source}]")

    baked = fold_batchnorm(sd)
    n_params = sum(w.size + b.size for _, w, b in baked)
    blob, shapes = quantize(baked, args.quant)
    src = build_source(blob, shapes, args.quant, kwargs["policy_channels"], 3,
                       allow_skip=allow_skip)

    Path(args.out).write_text(src)
    size = len(src)
    print(f"\npolicy params (value head dropped, BN folded): {n_params:,}")
    print(f"payload {len(blob):,} B -> {size:,} chars of source  [{args.quant}]")
    if size > CG_SOURCE_LIMIT:
        over = size / CG_SOURCE_LIMIT
        print(f"\n  !! {size:,} chars is {over:.1f}x CodinGame's {CG_SOURCE_LIMIT:,} limit.")
        print("     This checkpoint is too large to submit. Retrain smaller (base_channels)")
        print("     or use --quant int4; see the size table this script prints with --verify.")
    else:
        print(f"  fits CodinGame's {CG_SOURCE_LIMIT:,} char limit "
              f"({100 * size / CG_SOURCE_LIMIT:.0f}% used)")

    if args.verify:
        verify(args.out, sd, kwargs, args.verify, allow_skip=allow_skip)
    return 0


def verify(out_path, sd, kwargs, episodes, allow_skip=True):
    """Replay real games: does the baked numpy net pick the same cells as the torch model?

    `allow_skip` mirrors what was baked, so the torch reference is asked the same question the
    baked file answers - otherwise every disrupt turn reads as a disagreement."""
    from railroad_env import RailroadGymEnv
    from training.agent_model import RailroadUNet
    from training.simulator import GameSimulator
    from training.ppo import _decode_action

    ns = {}
    src = Path(out_path).read_text()
    head = src.split("# ---------------------------------------------------------------- game")[0]
    exec(compile(head, out_path, "exec"), ns)
    fwd = ns["_fwd"]

    model = RailroadUNet(**kwargs)
    model.load_state_dict(sd)
    model.eval()

    agree = total = 0
    max_err = 0.0
    for seed in range(episodes):
        env = RailroadGymEnv(max_turns=100, opponent_strategy="level2", seed=seed)
        env.reset()
        sim = GameSimulator.from_env(env, allow_skip_disrupt=kwargs["policy_channels"] >= 3,
                                     force_disrupt=not allow_skip)
        while not sim.is_game_over():
            state = sim.get_encoded_state()
            mask = sim.get_action_mask()
            with torch.no_grad():
                ref, _ = model(torch.from_numpy(state).unsqueeze(0),
                               torch.from_numpy(mask).unsqueeze(0))
            ref = ref[0].numpy()
            got = np.where(mask == 0, -np.inf, fwd(state))
            finite = np.isfinite(ref) & np.isfinite(got)
            if finite.any():
                max_err = max(max_err, float(np.abs(ref[finite] - got[finite]).max()))
            agree += int(np.argmax(ref) == np.argmax(got))
            total += 1
            sim.apply(_decode_action(int(np.argmax(ref)), mask.shape, sim.pad_offsets))
    print(f"\nverify: {agree}/{total} decisions identical ({100 * agree / max(total,1):.2f}%), "
          f"max logit error {max_err:.4g}")


if __name__ == "__main__":
    sys.exit(main())
