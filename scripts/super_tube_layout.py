#!/usr/bin/env python3
"""Layout and wave timing of the super-hard 1D tube: exact Riemann fronts of every interface, and when stages first meet."""
import math
g=1.4
def c(r,p): return math.sqrt(g*p/r)
def f(p,r,pk):
    A=2/((g+1)*r); B=(g-1)/(g+1)*pk; ck=c(r,pk)
    if p>pk: return (p-pk)*math.sqrt(A/(p+B))
    return 2*ck/(g-1)*((p/pk)**((g-1)/(2*g))-1)
def star(L,R):
    (rl,ul,pl),(rr,ur,pr)=L,R
    lo,hi=1e-14,1e14
    for _ in range(300):
        p=math.sqrt(lo*hi)
        v=f(p,rl,pl)+f(p,rr,pr)+ur-ul
        if v>0: hi=p
        else: lo=p
    p=math.sqrt(lo*hi); u=0.5*(ul+ur)+0.5*(f(p,rr,pr)-f(p,rl,pl)); return p,u
def speeds(L,R):
    (rl,ul,pl),(rr,ur,pr)=L,R
    p,u=star(L,R)
    if p>pl: sl=ul-c(rl,pl)*math.sqrt((g+1)/(2*g)*p/pl+(g-1)/(2*g)); left=('shock',sl,sl)
    else:
        cs=c(rl,pl)*(p/pl)**((g-1)/(2*g)); left=('rarefaction',ul-c(rl,pl),u-cs)
    if p>pr: sr=ur+c(rr,pr)*math.sqrt((g+1)/(2*g)*p/pr+(g-1)/(2*g)); right=('shock',sr,sr)
    else:
        cs=c(rr,pr)*(p/pr)**((g-1)/(2*g)); right=('rarefaction',u+cs,ur+c(rr,pr))
    # star densities
    def rstar(r,pk,side):
        if p>pk: return r*((p/pk+(g-1)/(g+1))/((g-1)/(g+1)*p/pk+1))
        return r*(p/pk)**(1/g)
    return dict(p=p,u=u,left=left,right=right,rl=rstar(rl,pl,'l'),rr=rstar(rr,pr,'r'))
amb=(1.0,0.0,1.0); mv=(1.0,1.0,1.0)
stages=[('A quiet',[(1.0,amb)]),
 ('B smooth waves',[(1.0,amb)]),
 ('C stationary peak',[(0.05,(10.0,0.0,1.0))]),
 ('C2 moving peak',[(0.75,mv),(0.05,(10.0,1.0,1.0)),(0.75,mv)]),
 ('D sonic rarefaction',[(0.5,(1.0,0.75,1.0)),(0.5,(0.125,0.0,0.1))]),
 ('E extreme jump',[(0.5,(1.0,0.0,0.1)),(0.5,(1e-3,0.0,1e-10))]),
 ('F slow shock',[(0.5,(3.86,-0.81,10.33)),(0.5,(1.0,-3.44,1.0))]),
 ('S colliding streams',[(0.3,(5.99924,19.5975,460.894)),(0.3,(5.99242,-6.19633,46.0950))]),
 ('G near vacuum',[(0.5,(1.0,-2.0,0.4)),(0.5,(1.0,2.0,0.4))]),
 ('H double blast',[(0.1,(1.0,0.0,1000.0)),(0.8,(1.0,0.0,0.01)),(0.1,(1.0,0.0,100.0))])]
def layout(bufs):
    segs=[];x=0.0
    for i,(n,parts) in enumerate(stages):
        if i>0: segs.append((x,'buffer',amb)); x+=bufs[i-1]
        for w,s in parts: segs.append((x,n,s)); x+=w
    return segs,x
def fronts(segs):
    fr=[]
    for (x,a,L),(x2,b,R) in zip(segs,segs[1:]):
        if L==R: continue
        s=speeds(L,R); fr.append((x2,min(s['left'][1],s['left'][2]),max(s['right'][1],s['right'][2]),a,b))
    return fr
def meets(fr):
    out=[]
    for A,B in zip(fr,fr[1:]):
        if A[2]>B[1] and not (A[3].startswith('H') and B[3].startswith('H')) and not (A[4]==B[3]==A[3] or A[3]==B[4]):
            out.append(((B[0]-A[0])/(A[2]-B[1]),A[3],A[4],B[3],B[4]))
    return sorted(out)
# buffers sized per neighbour speed: start uniform 2, then widen near fast stages
bufs=[2.0]*(len(stages)-1)
names=[s[0] for s in stages]
bufs[names.index('S colliding streams')-1]=4.0
bufs[names.index('S colliding streams')]=4.0
bufs[names.index('H double blast')-1]=4.0
segs,end=layout(bufs); fr=fronts(segs)
print('length',round(end,2))
for m in meets(fr)[:8]: print(f"t={m[0]:.4f}  {m[1][:20]}|{m[2][:20]}  meets  {m[3][:20]}|{m[4][:20]}")
for x,l,r,a,b in fr: print(f"{x:6.2f} {l:8.2f} {r:8.2f}  {a[:22]} | {b[:22]}")
