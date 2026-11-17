"""Matched-volume statistics, fixed GT regions, and comparable standard scores."""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from metrics import bscan_scores, scores

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
HISTOGRAM_EDGES = np.linspace(-.25, 1.25, 301)
VIEW_NAMES = ('ILM_OPL', 'OPL_BM', 'FULL', 'ZMAX', 'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN')


def depth_masks(layers, depth):
    z = ((np.arange(depth, dtype=np.float64)+.5)*640/depth-.5)[:, None, None]
    ilm, opl, bm = [layers[i].astype(np.float64)-1 for i in [0, 2, 5]]
    return dict(ILM_OPL=(z >= ilm) & (z <= opl), OPL_BM=(z >= opl) & (z <= bm),
                ILM_OPL_MEAN=(z >= ilm) & (z < opl), OPL_BM_MEAN=(z >= opl) & (z <= bm),
                ILM_BM_MEAN=(z >= ilm) & (z <= bm))


class Views:
    def __init__(self, layers, shape):
        self.masks = depth_masks(layers, shape[0])
        self.counts = {key: mask.sum(0) for key, mask in self.masks.items() if key.endswith('_MEAN')}

    def __call__(self, volume):
        maps = dict(FULL=volume.mean(0, dtype=np.float64).T.astype(np.float32), ZMAX=volume.max(0).T)
        for name, mask in self.masks.items():
            if name.endswith('_MEAN'):
                numerator = np.where(mask, volume, 0).sum(0, dtype=np.float32)
                maps[name] = (numerator/(self.counts[name]+1e-6)).T.astype(np.float32)
            else:
                # Use a signed maximum for raw trajectory views. On [0,1] endpoints,
                # this equals the parent's zero-filled maximum.
                maps[name] = np.where(mask.any(0), np.where(mask, volume, -np.inf).max(0), 0).T
        return maps


def vessel_regions(labels, truth, views, diameter_threshold, faz_volume=None):
    annotated = labels['capillary'] | labels['large']
    centerline = skeletonize(annotated)
    small = np.zeros_like(annotated)
    large = np.zeros_like(annotated)
    diameter = np.zeros(annotated.shape, dtype=np.float32)
    if centerline.any():
        distance = ndi.distance_transform_edt(annotated)
        nearest = ndi.distance_transform_edt(~centerline, return_distances=False, return_indices=True)
        diameter = (2*distance[tuple(nearest)]).astype(np.float32)
        diameter[~annotated] = 0
        small = annotated & (diameter <= diameter_threshold)
        large = annotated & (diameter > diameter_threshold)
    inside = views.masks['ILM_BM_MEAN']
    shallow = views.masks['ILM_OPL']
    # A GT vessel-labeled ray is not a voxel segmentation. For a supplemental
    # 3D candidate and G6, mark the voxels with >=50% of that ray's GT maximum.
    gt_mip = np.where(shallow, truth, 0).max(0)
    vessel_candidate = shallow & (gt_mip > 0)[None] & (truth >= .5*gt_mip[None])
    vessel3d = vessel_candidate & annotated.T[None]
    regions = dict(background_all_rays=~annotated.T, background_ILM_OPL=(~annotated.T)[None] & shallow,
                   retina_exterior=~inside, small_vessel_rays=small.T[None] & shallow,
                   large_vessel_rays=large.T[None] & shallow,
                   small_vessel_candidate_3d=vessel3d & small.T[None],
                   large_vessel_candidate_3d=vessel3d & large.T[None],
                   annotated_large_rays=labels['large'].T[None] & shallow,
                   ILM_OPL=shallow, OPL_BM=views.masks['OPL_BM'], ILM_BM=inside)
    faz = labels.get('faz2d', np.zeros(annotated.shape, dtype=bool))
    regions['FAZ_ILM_OPL'] = faz.T[None] & shallow
    if faz_volume is not None:
        regions['FAZ3D_retinal'] = faz_volume & inside
    masks = dict(vessel=annotated, centerline=centerline, small=small, large=large,
                 diameter=diameter, faz=faz)
    return regions, masks


def distribution(volume):
    all_min, all_max = float(volume.min()), float(volume.max())
    quantiles = np.percentile(volume, PERCENTILES)
    count = np.histogram(volume, bins=HISTOGRAM_EDGES)[0].tolist()
    return dict(mean=float(volume.mean(dtype=np.float64)), std=float(volume.std(dtype=np.float64)),
                min=all_min, max=all_max, **{f'p{n}': float(v) for n, v in zip(PERCENTILES, quantiles)},
                negative_ratio=float(np.count_nonzero(volume < 0)/volume.size),
                overflow_ratio=float(np.count_nonzero(volume > 1)/volume.size),
                histogram=count, histogram_underflow=int((volume < HISTOGRAM_EDGES[0]).sum()),
                histogram_overflow=int((volume > HISTOGRAM_EDGES[-1]).sum()))


def intensity(volume, truth, regions):
    difference = volume - truth
    by_region = {}
    for name, mask in regions.items():
        mask = np.broadcast_to(mask, volume.shape)
        n = int(mask.sum())
        by_region[name] = dict(voxels=n,
                                mean=float(volume.mean(where=mask, dtype=np.float64)) if n else None,
                                gt_mean=float(truth.mean(where=mask, dtype=np.float64)) if n else None,
                                bias=float(difference.mean(where=mask, dtype=np.float64)) if n else None,
                                mae=float(np.abs(difference[mask]).mean(dtype=np.float64)) if n else None)
    axes = (1, 2)
    return dict(distribution=distribution(volume), regions=by_region,
                by_depth=dict(mean=volume.mean(axes, dtype=np.float64).tolist(),
                              std=volume.std(axes, dtype=np.float64).tolist(),
                              gt_mean=truth.mean(axes, dtype=np.float64).tolist(),
                              bias=difference.mean(axes, dtype=np.float64).tolist(),
                              mae=np.abs(difference).mean(axes, dtype=np.float64).tolist()))


def quality(volume, truth, views, truth_views, include_bscan_rows=False):
    if not np.isfinite(volume).all():
        raise FloatingPointError('Nonfinite metric input')
    projected = views(volume)
    bscan, slices = bscan_scores(volume, truth)
    metrics = dict(volume=scores(volume, truth), bscan2d=bscan)
    metrics.update({name: scores(projected[name], truth_views[name]) for name in VIEW_NAMES})
    return metrics, projected, slices if include_bscan_rows else None


def topology(pred, gt, masks):
    threshold = float(threshold_otsu(gt))
    estimated = pred > threshold
    vessel = masks['vessel']
    pred_centerline = skeletonize(estimated)
    gt_centerline = masks['centerline']
    dice = 2*(estimated & vessel).sum()/max(int(estimated.sum() + vessel.sum()), 1)
    precision = (pred_centerline & vessel).sum()/max(int(pred_centerline.sum()), 1)
    sensitivity = (gt_centerline & estimated).sum()/max(int(gt_centerline.sum()), 1)
    cldice = 2*precision*sensitivity/max(float(precision+sensitivity), 1e-12)
    # Derive a predicted FAZ solely by the connected no-vessel component at the
    # GT FAZ centroid. Unbounded or obstructed cases remain undefined.
    faz = masks['faz']
    area = None
    if faz.any():
        centroid = tuple(np.rint(np.argwhere(faz).mean(0)).astype(int))
        components, _ = ndi.label(~estimated)
        i = components[centroid]
        if i:
            component = components == i
            if not (component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any()):
                area = int(component.sum())
    return dict(dice=float(dice), cldice=float(cldice), gt_otsu_threshold=threshold,
                GT_FAZ_area_px=int(faz.sum()) if faz.any() else None, predicted_FAZ_area_px=area,
                FAZ_signed_area_error_px=area-int(faz.sum()) if area is not None else None,
                FAZ_absolute_area_error_px=abs(area-int(faz.sum())) if area is not None else None)


def corrections(raw, truth):
    delta = truth.mean(dtype=np.float64) - raw.mean(dtype=np.float64)
    yield 'global_mean', raw + np.float32(delta)
    raw_std = raw.std(dtype=np.float64)
    ratio = truth.std(dtype=np.float64)/raw_std if raw_std > 1e-12 else 0.
    yield 'mean_std', (raw - np.float32(raw.mean(dtype=np.float64)))*np.float32(ratio) + np.float32(truth.mean(dtype=np.float64))
    shift = truth.mean((1, 2), keepdims=True, dtype=np.float64) - raw.mean((1, 2), keepdims=True, dtype=np.float64)
    yield 'depth_mean', raw + shift.astype(np.float32)


def plane_views(volume, projected):
    _, height, width = volume.shape
    return dict(Bscan=volume[:, height//2, :], Orthogonal=volume[:, :, width//2], **projected)
