import numpy as np
import torch


def _log(*args):
    print("[seamless_loop]", *args)


def _log2x2(L):
    """Matrix logarithm of a 2x2 matrix via Sylvester's closed form.

    Handles real-distinct, complex-conjugate, and repeated eigenvalues without
    an eigendecomposition, so it stays well-conditioned for near-translation
    (where the linear part is ~identity) and near-pure-scale matrices.
    """
    a, b, c, d = L[0, 0], L[0, 1], L[1, 0], L[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc > 1e-12:
        s = np.sqrt(disc)
        l1 = (tr + s) / 2.0
        l2 = (tr - s) / 2.0
        return (np.log(l1) - np.log(l2)) / (l1 - l2) * L + \
               (l1 * np.log(l2) - l2 * np.log(l1)) / (l1 - l2) * np.eye(2)
    if disc < -1e-12:
        alpha = tr / 2.0
        beta = np.sqrt(-disc) / 2.0
        rho = np.sqrt(det)
        phi = np.arctan2(beta, alpha)
        return np.log(rho) * np.eye(2) + phi * (L - alpha * np.eye(2)) / beta
    return np.log(tr / 2.0) * np.eye(2)


def _exp2x2(X):
    """Matrix exponential of a 2x2 matrix via Sylvester's closed form."""
    a, b, c, d = X[0, 0], X[0, 1], X[1, 0], X[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc > 1e-12:
        s = np.sqrt(disc)
        l1 = (tr + s) / 2.0
        l2 = (tr - s) / 2.0
        e1, e2 = np.exp(l1), np.exp(l2)
        return (e1 - e2) / (l1 - l2) * X + (l1 * e2 - l2 * e1) / (l1 - l2) * np.eye(2)
    if disc < -1e-12:
        mu = np.sqrt(-disc) / 2.0
        return np.exp(tr / 2.0) * (np.cos(mu) * np.eye(2) + np.sin(mu) / mu * (X - tr / 2.0 * np.eye(2)))
    return np.exp(tr / 2.0) * (np.eye(2) + X - tr / 2.0 * np.eye(2))


def _affine_log(M):
    """Logarithm of a homogeneous 3x3 affine in the affine Lie algebra.

    For M = [[L, t], [0, 1]] the log is [[log(L), v], [0, 0]] with the
    translation correction v = log(L) * (L - I)^-1 * t, which couples the
    translation to the linear part so a zoom-about-centre stays a constant
    velocity zoom. Pure translation (L ~= I) degenerates to v = t.
    """
    L = M[:2, :2]
    t = M[:2, 2]
    X = _log2x2(L)
    LI = L - np.eye(2)
    v = t if np.linalg.norm(LI) < 1e-10 else X @ np.linalg.solve(LI, t)
    out = np.zeros((3, 3), dtype=np.float64)
    out[:2, :2] = X
    out[:2, 2] = v
    return out


def _affine_exp(l):
    """Exponential of an affine Lie-algebra element back to a 3x3 affine.

    Inverse of _affine_log: L = exp(X), t = (L - I) * X^-1 * v.
    """
    X = l[:2, :2]
    v = l[:2, 2]
    L = _exp2x2(X)
    t = v if np.linalg.norm(X) < 1e-10 else (L - np.eye(2)) @ np.linalg.solve(X, v)
    out = np.eye(3, dtype=np.float64)
    out[:2, :2] = L
    out[:2, 2] = t
    return out


def _radial_inv(rd, k):
    """Newton solve r_u*(1 + k*r_u^2) = rd for r_u, vectorised (numpy only)."""
    r = rd.copy()
    for _ in range(30):
        r2 = r * r
        r = r - (r + k * r2 * r - rd) / (1.0 + 3.0 * k * r2)
    return r


def _radial_remap(h, w, k, invert):
    """Build cv2 remap arrays (map_x, map_y) for a single-coefficient radial map.

    Model (normalised coords, corner radius == 1):  r' = r*(1 + k*r^2).
      invert=False -> 'apply'  distortion (ideal -> distorted).
      invert=True  -> 'remove' distortion (distorted -> ideal).
    In 'apply', k>0 pushes edge content outward (pincushion), k<0 = barrel.
    """
    import cv2
    cx, cy = w / 2.0, h / 2.0
    f = float(np.hypot(w, h) / 2.0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - cx) / f
    ny = (ys - cy) / f
    if invert:
        r2 = nx * nx + ny * ny
        scale = 1.0 + k * r2
        mx = cx + f * nx * scale
        my = cy + f * ny * scale
    else:
        rd = np.sqrt(nx * nx + ny * ny)
        ru = _radial_inv(rd, k)
        ratio = np.ones_like(ru)
        nz = rd > 1e-6
        ratio[nz] = ru[nz] / rd[nz]
        mx = cx + f * nx * ratio
        my = cy + f * ny * ratio
    return mx.astype(np.float32), my.astype(np.float32)


def apply_distortion(img, k):
    """Apply radial distortion with coefficient k (ideal -> distorted)."""
    import cv2
    h, w = img.shape[:2]
    mx, my = _radial_remap(h, w, k, invert=False)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_distortion(img, k):
    """Remove radial distortion with coefficient k (distorted -> ideal)."""
    import cv2
    h, w = img.shape[:2]
    mx, my = _radial_remap(h, w, k, invert=True)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def solve_radial(ref, drifted, k_range=(-0.4, 0.4), steps=49):
    """Estimate the radial coefficient k that removes the distortion present in
    ``drifted`` relative to the clean ``ref`` frame.

    Solves by GLOBAL image alignment: for each candidate k it removes the
    distortion from ``drifted`` and scores how well the result reproduces
    ``ref`` inside a shared central box, then refines around the best coarse
    bin. This is preferred over per-keypoint radius fitting because the
    periphery of a distorted frame is too transformed for ORB features to
    match, leaving only central correspondences where the radial effect is
    sub-pixel and the fit is noise-dominated.
    """
    h, w = drifted.shape[:2]
    c0, c1 = h // 4, 3 * h // 4
    r0, r1 = w // 4, 3 * w // 4
    ref_c = ref[c0:c1, r0:r1].astype(np.float32)

    ks = np.linspace(k_range[0], k_range[1], steps)
    costs = [np.abs(remove_distortion(drifted, kk)[c0:c1, r0:r1].astype(np.float32) - ref_c).mean()
             for kk in ks]
    i = int(np.argmin(costs))
    best_k, best_cost = ks[i], costs[i]

    lo = ks[max(0, i - 1)]
    hi = ks[min(len(ks) - 1, i + 1)]
    for kk in np.linspace(lo, hi, steps):
        cost = np.abs(remove_distortion(drifted, kk)[c0:c1, r0:r1].astype(np.float32) - ref_c).mean()
        if cost < best_cost:
            best_cost, best_k = cost, kk
    return float(best_k)


class CompensateDriftAlign:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    transform_methods = ["affine", "similarity", "radial"]
    interp_methods = ["log", "linear"]
    anchor_modes = ["auto", "first", "last"]
    mix_modes = ["off", "on"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_features": ("INT", {"default": 300, "min": 20, "max": 5000, "step": 10}),
                "min_matches": ("INT", {"default": 15, "min": 5, "max": 500, "step": 1}),
                "anchor_frame": (cls.anchor_modes, {"default": "auto"}),
                "transform": (cls.transform_methods, {"default": "affine"}),
                "interp": (cls.interp_methods, {"default": "log"}),
                "upscale_method": (cls.upscale_methods, {"default": "bicubic"}),
                "drop_last_frames": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
                "mix_first_last": (cls.mix_modes, {"default": "off"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"

    CATEGORY = "image/batch"
    SEARCH_ALIASES = ["seamless loop", "autoalign", "compensate drift", "auto zoom compensation", "feature align", "compensate zoom"]

    def apply(self, images, max_features, min_matches, anchor_frame, transform, interp, upscale_method, drop_last_frames=1, mix_first_last="off"):
        import cv2

        def to_numpy(t):
            return t.detach().to("cpu", dtype=torch.float32).numpy()

        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        drop = min(int(drop_last_frames), max(0, n - 1))
        out_n = n - drop
        _log(f"frames={n} size={w}x{h} max_features={max_features} min_matches={min_matches} "
             f"anchor_frame={anchor_frame} transform={transform} interp={interp} "
             f"drop_last_frames={drop} mix_first_last={mix_first_last}")

        if n < 2:
            _log("only one frame; returning unchanged")
            return (images[:out_n],)

        orig_dtype = images.dtype
        orig_device = images.device

        np_frames = [to_numpy(images[i]) for i in range(n)]

        # ---- Radial mode: corrects ONLY radial distortion, no affine. ----
        # solve_radial finds k such that remove_distortion(np_last, k) ~= np_first,
        # i.e. the differential distortion with the FIRST frame as the radial
        # reference. Every frame's coefficient interpolates 0 -> k across the
        # batch. This is the standalone radial solver applied per frame.
        if transform == "radial":
            k_ref = solve_radial(np_frames[0], np_frames[-1])
            _log(f"radial: solved differential k={k_ref:+.4f} (first frame is radial reference; no affine)")
            out_frames = []
            for i in range(n):
                f = i / (n - 1)
                ki = f * k_ref
                corr = np_frames[i] if ki == 0.0 else remove_distortion(np_frames[i], ki)
                out_frames.append(torch.from_numpy(corr.astype(np.float32)).to(orig_device).to(orig_dtype))
            if mix_first_last == "on" and drop == 1:
                f0, fl = out_frames[0], out_frames[-1]
                out_frames[0] = ((f0.float() + fl.float()) * 0.5).to(f0.dtype)
                _log("mixed first frame with dropped last frame (50/50)")
            out = torch.stack(out_frames, dim=0)[:out_n]
            _log("first and last frames now share the FIRST frame's framing (loop closes)")
            return (out,)

        np_first, np_last = np_frames[0], np_frames[-1]
        src_frames = np_frames

        gray_first = cv2.cvtColor((np_first * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_last = cv2.cvtColor((np_last * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=max_features)
        kp_first, des_first = orb.detectAndCompute(gray_first, None)
        kp_last, des_last = orb.detectAndCompute(gray_last, None)
        if des_first is None or des_last is None or len(kp_first) < min_matches or len(kp_last) < min_matches:
            _log(f"insufficient features ({0 if des_first is None else len(kp_first)} / "
                 f"{0 if des_last is None else len(kp_last)}); returning unchanged")
            return (images[:out_n],)

        # M maps first frame -> last frame framing: p_last = M * p_first
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des_first, des_last, k=2)
        src_pts, dst_pts = [], []
        for m, nn_ in knn:
            if m is not None and nn_ is not None and m.distance < 0.7 * nn_.distance:
                src_pts.append(kp_first[m.queryIdx].pt)
                dst_pts.append(kp_last[m.trainIdx].pt)
        if len(src_pts) < min_matches:
            _log(f"only {len(src_pts)} good matches < {min_matches}; returning unchanged")
            return (images[:out_n],)

        if transform == "affine":
            M, inliers = cv2.estimateAffine2D(
                np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=3.0)
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inliers is None or int(inliers.sum()) < min_matches:
            _log(f"RANSAC rejected ({0 if M is None else int(inliers.sum())} inliers); returning unchanged")
            return (images[:out_n],)

        M23 = M.astype(np.float64)
        M33 = np.vstack([M23, [[0.0, 0.0, 1.0]]])
        invM33 = np.linalg.inv(M33)

        # The raw estimate may come back in either orientation (M or its inverse
        # can both be valid affines). Disambiguate by warping the LAST frame with
        # each candidate and keeping the one that best reproduces the FIRST frame
        # in the shared central region.
        cen_y, cen_x, box = h // 2, w // 2, min(h, w) // 3
        sl = lambda a: a[cen_y - box:cen_y + box, cen_x - box:cen_x + box]

        def center_err(T):
            warped = cv2.warpAffine(np_last, T[:2].astype(np.float32), (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return float(np.abs(sl(warped) - sl(np_first)).mean())

        if center_err(M33) <= center_err(invM33):
            G = M33          # G warps last frame -> first framing (out->src)
        else:
            G = invM33
        detG = float(np.linalg.det(G[:2, :2]))
        _log(f"{len(src_pts)} matches, {int(inliers.sum())} inliers; "
             f"last->first framing det={detG:.4f}")

        # Decide which frame to anchor to. 'auto' picks whichever endpoint is more
        # zoomed-in so every interpolation is a magnification (zoom IN: crop/discard
        # border detail, never invent it). 'first'/'last' force that endpoint, even
        # if it means interpolations sometimes zoom OUT.
        # G maps last->first. detG > 1 means that map magnifies, i.e. the first
        # frame shows *larger* content than the last -> the first is the more
        # zoomed-in (tele/cropped) end, so it is the reference.
        if anchor_frame == "first":
            ref_is_first = True
        elif anchor_frame == "last":
            ref_is_first = False
        else:
            ref_is_first = detG > 1.0  # first frame is the more zoomed-in end

        if ref_is_first:
            A = np.eye(3)       # warp first (reference) -> identity
            B = G               # warp last frame -> first (reference) framing
            _log(f"anchoring to FIRST frame ({anchor_frame})")
        else:
            A = np.linalg.inv(G)  # warp first frame -> last (reference) framing
            B = np.eye(3)         # warp last (reference) -> identity
            _log(f"anchoring to LAST frame ({anchor_frame})")

        flags = {
            "nearest-exact": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[upscale_method]

        def per_frame(out_src3):
            return out_src3[:2].astype(np.float32)

        la = _affine_log(A)
        lb = _affine_log(B)
        out_frames = []
        for i in range(n):
            f = i / (n - 1)
            if interp == "linear":
                T = (1.0 - f) * A + f * B
            else:
                T = _affine_exp((1.0 - f) * la + f * lb)
            warp = cv2.warpAffine(src_frames[i], per_frame(T), (w, h),
                                  flags=flags, borderMode=cv2.BORDER_REPLICATE)
            out_frames.append(torch.from_numpy(warp.astype(np.float32)).to(orig_device).to(orig_dtype))

        # Seamless loop: the last frame is dropped (drop==1). Replace the first
        # output frame with the 50/50 blend of the aligned first and removed last
        # frame so the loop wrap is smooth.
        if mix_first_last == "on" and drop == 1:
            f0, fl = out_frames[0], out_frames[-1]
            out_frames[0] = ((f0.float() + fl.float()) * 0.5).to(f0.dtype)
            _log("mixed first frame with dropped last frame (50/50)")
        out = torch.stack(out_frames, dim=0)[:out_n]

        if not ref_is_first:
            _log("first and last frames now share the LAST frame's framing (loop closes)")
        else:
            _log("first and last frames now share the FIRST frame's framing (loop closes)")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "CompensateDriftAlign": CompensateDriftAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompensateDriftAlign": "Compensate Drift (Auto Align First+Last)",
}
