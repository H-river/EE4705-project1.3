"""Known-shape RGB-D localization within a VLM box; no simulator state access."""
import numpy as np


def pixel_box(box, width=640, height=480):
    x1,y1,x2,y2=box
    return (int(np.floor(x1*width/1000)),int(np.floor(y1*height/1000)),
            min(width,int(np.ceil(x2*width/1000))),min(height,int(np.ceil(y2*height/1000))))


def localize(obs, bbox, name, color):
    """Return estimated center/support or None when geometry cannot be supported.

    Dimensions are the project's public class priors in assets/objects.yaml.
    Color masking refines pixels inside the model box; it cannot add detections.
    """
    x1,y1,x2,y2=bbox
    if x1<=0 or y1<=0 or x2>=640 or y2>=480:
        return None, 'box touches image boundary'
    yy,xx=np.mgrid[y1:y2,x1:x2]
    depth=obs.depth[y1:y2,x1:x2]
    rgb=obs.rgb[y1:y2,x1:x2].astype(float)
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    masks={'gray':(np.maximum.reduce([r,g,b])-np.minimum.reduce([r,g,b])<18)&(r>40),
           'dark_red':(r>1.7*g)&(r>1.8*b)&(r>55),
           'red':(r>160)&(g<110)&(b<110)&(r-g>100),
           'blue':(b>1.4*r)&(b>1.25*g)&(b>65),
           'green':(g>1.4*r)&(g>1.25*b)&(g>55)}
    if color not in masks: return None,'no supported color segmentation for metric estimate'
    valid=np.isfinite(depth)&(depth>0)&masks[color]
    if np.count_nonzero(valid)<24: return None,'too few valid foreground depth pixels'
    k,t=obs.intrinsics,obs.t_world_camera
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(t))
            or k[0,0] <= 0 or k[1,1] <= 0):
        return None,'invalid camera geometry'
    z=depth[valid];u=xx[valid];v=yy[valid]
    pc=np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z))
    cloud=pc@t[:3,:3].T+t[:3,3]
    if not np.all(np.isfinite(cloud)): return None,'invalid camera geometry'
    center=(cloud.min(axis=0)+cloud.max(axis=0))/2
    if name=='red_region':
        # Reject an incomplete support fit, rather than using its clipped midpoint.
        z0=np.percentile(cloud[:,2],30)
        cloud=cloud[np.abs(cloud[:,2]-z0)<.006]
        if len(cloud)<24 or np.any(np.ptp(cloud[:,:2],axis=0)<.13): return None,'region support is partly hidden'
        center=(cloud.min(axis=0)+cloud.max(axis=0))/2
        center[2]=np.median(cloud[:,2])
    elif name=='stone':
        mid=np.median(cloud,axis=0);p=cloud-mid;scale=np.array([.030,.025,.025])
        design=np.column_stack((2*p/(scale*scale),np.ones(len(p))))
        coef,_,rank,_=np.linalg.lstsq(design,np.sum((p/scale)**2,axis=1),rcond=None)
        estimate=mid+coef[:3]
        residual=np.median(np.abs(np.sum(((cloud-estimate)/scale)**2,axis=1)-1))
        if rank<4 or np.linalg.norm(estimate-center)>.04 or residual>.35:
            return None,'ellipsoid surface fit is unreliable'
        center=estimate
    else:
        height=.05 if name=='cube' else .12
        top_z=np.percentile(cloud[:,2],99)
        top=cloud[cloud[:,2]>top_z-.006]
        if len(top)<12: return None,'top surface is not visible'
        center[:2]=(top[:,:2].min(axis=0)+top[:,:2].max(axis=0))/2
        center[2]=top_z-height/2
    return tuple(float(v) for v in center),'RGB-D fit with public class geometry'
