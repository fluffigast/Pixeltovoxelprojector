# Offline reference detector for scoring web/motion.js (see tools/motion_eval.js).
# Slow, OpenCV-based: feature-tracked similarity stabilisation, 3-frame
# aligned min-diff, static-edge mask, connected components, nearest-neighbour
# linking. Output tracks (>=4 frames, moved >15 px) are CANDIDATES: inspect
# them (e.g. crop sheets) and prune before treating them as ground truth.
# Usage: venv/bin/python -I tools/build_gt.py <video.mp4> <out.json> [thr]
import cv2, numpy as np, json, sys
VID=sys.argv[1]; OUT=sys.argv[2]; THR=float(sys.argv[3]) if len(sys.argv)>3 else 14
c=cv2.VideoCapture(VID); frames=[]
while True:
    ok,fr=c.read()
    if not ok: break
    frames.append(fr)
H,W=frames[0].shape[:2]; N=len(frames)
gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
blur=[cv2.GaussianBlur(g,(3,3),0) for g in gray]
def warp(gp,gc):
    pts=cv2.goodFeaturesToTrack(gp,400,0.01,12)
    nxt,st,_=cv2.calcOpticalFlowPyrLK(gp,gc,pts,None,winSize=(21,21),maxLevel=3)
    ok=st.ravel()==1
    M,_=cv2.estimateAffinePartial2D(pts[ok],nxt[ok],method=cv2.RANSAC,ransacReprojThreshold=1.5)
    return cv2.warpAffine(gp,M,(W,H),borderMode=cv2.BORDER_REPLICATE)
wp=[None]+[warp(blur[t-1],blur[t]) for t in range(1,N)]
wn=[warp(blur[t+1],blur[t]) for t in range(0,N-1)]+[None]
dets=[[] for _ in range(N)]; cnt=[]
for t in range(1,N-1):
    d=np.minimum(cv2.absdiff(blur[t],wp[t]),cv2.absdiff(blur[t],wn[t])).astype(np.float32)
    def G(img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3); return np.hypot(gx,gy)
    static=np.minimum(G(blur[t]),np.minimum(G(wp[t]),G(wn[t])))   # edge present in all three aligned frames = scenery
    d[cv2.dilate((static>60).astype(np.uint8),np.ones((5,5),np.uint8))>0]=0
    n,lab,stats,cent=cv2.connectedComponentsWithStats((d>THR).astype(np.uint8),connectivity=4)
    for i in range(1,n):
        x,y,w,h,a=stats[i]
        if 2<=a<=400 and max(w,h)<=40 and 8<=cent[i][0]<=W-9 and 8<=cent[i][1]<=H-9: dets[t].append((float(cent[i][0]),float(cent[i][1]),int(w),int(h),int(a)))
    cnt.append(len(dets[t]))
print("dets/frame: mean=%.1f p50=%d p90=%d max=%d"%(np.mean(cnt),np.median(cnt),np.percentile(cnt,90),max(cnt)))
done=[]; active=[]
for t in range(N):
    used=set()
    for tr in active:
        best=None;bd=60
        for j,dd in enumerate(dets[t]):
            if j in used: continue
            dist=np.hypot(dd[0]-tr['pts'][-1][1],dd[1]-tr['pts'][-1][2])
            if dist<bd: bd=dist;best=j
        if best is not None: used.add(best); dd=dets[t][best]; tr['pts'].append((t,)+dd); tr['miss']=0
        else: tr['miss']+=1
    done+=[tr for tr in active if tr['miss']>2]; active=[tr for tr in active if tr['miss']<=2]
    for j,dd in enumerate(dets[t]):
        if j not in used: active.append({'pts':[(t,)+dd],'miss':0})
done+=active
good=[tr['pts'] for tr in done if len(tr['pts'])>=4 and np.hypot(tr['pts'][-1][1]-tr['pts'][0][1],tr['pts'][-1][2]-tr['pts'][0][2])>15]
print("tracks >=6 frames & moved >15px:",len(good))
for i,tr in enumerate(good): print(f"  {i}: f{tr[0][0]}-{tr[-1][0]} len={len(tr)} ({tr[0][1]:.0f},{tr[0][2]:.0f})->({tr[-1][1]:.0f},{tr[-1][2]:.0f}) area={np.mean([p[5] for p in tr]):.1f} wh={np.mean([p[3] for p in tr]):.1f}x{np.mean([p[4] for p in tr]):.1f}")
json.dump({'w':W,'h':H,'n':N,'tracks':good,'alldets':[[list(map(float,dd)) for dd in d] for d in dets]},open(OUT,'w'))
vis=frames[N//2].copy()
for i,tr in enumerate(good):
    col=[(0,0,255),(0,255,0),(255,0,255),(255,255,0),(0,165,255)][i%5]
    for p in tr: cv2.circle(vis,(int(p[1]),int(p[2])),3,col,-1)
    cv2.putText(vis,str(i),(int(tr[0][1])+5,int(tr[0][2])),0,0.7,col,2)
cv2.imwrite(OUT.replace('.json','_overlay.jpg'),vis)
