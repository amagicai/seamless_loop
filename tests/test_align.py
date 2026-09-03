import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

import importlib.util

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "comfyui_seamless_loop", os.path.join(_PKG_DIR, "__init__.py"))
_pkg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pkg)
SeamlessLoopAutoAlignFLV = _pkg.SeamlessLoopAutoAlignFLV
apply_distortion = _pkg.apply_distortion
remove_distortion = _pkg.remove_distortion
solve_radial = _pkg.solve_radial

ART = os.path.join(os.path.dirname(__file__), "artifacts_auto")


def _save_png(name, t):
    """Write a float [0,1] HWC tensor as a PNG (RGB->BGR for cv2) into ART."""
    os.makedirs(ART, exist_ok=True)
    img = (t.detach().clamp(0, 1).numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(ART, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_scene(size=384, seed=3):
    """Structured image with strong corners (quadrilaterals + grid) so ORB finds
    many unambiguous features, and gentle shading so borders recover cleanly."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), 90, np.uint8)
    for g in range(0, size, 64):
        cv2.line(img, (0, g), (size, g), (150, 150, 150), 1)
        cv2.line(img, (g, 0), (g, size), (150, 150, 150), 1)
    for _ in range(30):
        c = int(rng.randint(30, 220))
        color = (c, int(c * 0.8) % 255, int(c * 1.2) % 255)
        x, y = rng.randint(0, size - 100), rng.randint(0, size - 100)
        w, r = rng.randint(20, 120), rng.randint(20, 120)
        quad = np.array([[x + rng.randint(-14, 14), y + rng.randint(-14, 14)],
                         [x + w + rng.randint(-14, 14), y + rng.randint(-14, 14)],
                         [x + w + rng.randint(-14, 14), y + r + rng.randint(-14, 14)],
                         [x + rng.randint(-14, 14), y + r + rng.randint(-14, 14)]], np.int32)
        cv2.fillPoly(img, [quad], color)
    return img


def magnify(base, z, size):
    c = size / 2.0
    s = 1 / z
    M = np.array([[s, 0, c * (1 - s)], [0, s, c * (1 - s)]], np.float32)
    return cv2.warpAffine(base, M, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def make_first_zoomed_in(n=11, size=384, drift=0.05, seed=3):
    """The FIRST frame has the largest object (z=1, most zoomed-in); the last is
    progressively zoomed OUT (magnify(z>1) shrinks, so z rising = zoom out)."""
    base = make_scene(size, seed=seed)
    frames = [magnify(base, 1.0 + drift * (i / (n - 1)), size) for i in range(n)]
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def make_last_zoomed_in(n=11, size=384, drift=0.05, seed=3):
    """The LAST frame has the largest object (z=1, most zoomed-in); the first is
    the most zoomed OUT (z falling towards basis)."""
    base = make_scene(size, seed=seed)
    frames = [magnify(base, 1.0 + drift * (1 - i / (n - 1)), size) for i in range(n)]
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def make_anisotropic(n=9, size=384, drift=0.04):
    """x-only zoom: needs affine; similarity cannot represent it."""
    rng = np.random.RandomState(4)
    base = make_scene(size, seed=4)
    c = size / 2.0
    frames = []
    for i in range(n):
        zx = 1.0 + drift * (1 - i / (n - 1))
        sx, sy = 1 / zx, 1.0
        M = np.array([[sx, 0, c * (1 - sx)], [0, sy, c * (1 - sy)]], np.float32)
        frames.append(cv2.warpAffine(base, M, (size, size), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE))
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def make_scene_rect(h, w, seed=3):
    """Rectangular structured scene (grid + quads) for non-square coverage."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for gy in range(0, h, 64):
        cv2.line(img, (0, gy), (w, gy), (150, 150, 150), 1)
    for gx in range(0, w, 64):
        cv2.line(img, (gx, 0), (gx, h), (150, 150, 150), 1)
    for _ in range(30):
        c = int(rng.randint(30, 220))
        color = (c, int(c * 0.8) % 255, int(c * 1.2) % 255)
        x, y = rng.randint(0, max(1, w - 100)), rng.randint(0, max(1, h - 100))
        qw, qr = rng.randint(20, 120), rng.randint(20, 120)
        quad = np.array([[x + rng.randint(-14, 14), y + rng.randint(-14, 14)],
                         [x + qw + rng.randint(-14, 14), y + rng.randint(-14, 14)],
                         [x + qw + rng.randint(-14, 14), y + qr + rng.randint(-14, 14)],
                         [x + rng.randint(-14, 14), y + qr + rng.randint(-14, 14)]], np.int32)
        cv2.fillPoly(img, [quad], color)
    return img


def scale_about_center(base, zx, zy, h, w):
    """Zoom by (zx, zy) about the image centre (z>1 crops/zooms IN). Matches
    make_anisotropic/magnify conventions (s = 1/z)."""
    cx, cy = w / 2.0, h / 2.0
    sx, sy = 1.0 / zx, 1.0 / zy
    M = np.array([[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)]], np.float32)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rotate_about_center(base, angle_deg, h, w):
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def pan(base, dx, dy, h, w):
    M = np.array([[1, 0, float(dx)], [0, 1, float(dy)]], np.float32)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def shear_about_center(base, shx, shy, h, w):
    """Shear by (shx, shy) about the image centre: x += shx*y, y += shy*x.

    The linear part [[1, shx], [shy, 1]] has det == 1 (area-preserving), so like
    a pan it is invisible to the auto-anchor det test (near-tie); the loop must
    still close to a single framing."""
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[1.0, shx, -shx * cy], [shy, 1.0, -shy * cx]], np.float32)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def combined_zoom_pan_rotate(base, t, h, w, z_scale=0.06, rot_deg=12.0, dx=28.0, dy=18.0):
    """Zoom-IN + rotation + pan as ONE composed affine at progress t in [0,1].

    The single transform is affine, so it composes exactly, applied to the base
    as Mp (pan), then Mr (rotation), then Mz (zoom) — the ZOOM-IN IS LAST, so it
    crops away the border-replicated pixels that the pan/rotation would
    otherwise invent. This mirrors a real camera drifting several ways at once.
    z_scale/rot_deg/dx/dy give the total drift over the whole sequence."""
    cx, cy = w / 2.0, h / 2.0
    z = 1.0 + z_scale * t                     # zoom IN (crop): z rising
    s = 1.0 / z
    Mz = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
    Mr = cv2.getRotationMatrix2D((cx, cy), rot_deg * (t - 0.5), 1.0)
    Mp = np.array([[1, 0, dx * (t - 0.5)], [0, 1, dy * (t - 0.5)]], np.float32)

    def to3(M):
        return np.vstack([M, [[0.0, 0.0, 1.0]]])

    T = to3(Mz) @ to3(Mr) @ to3(Mp)
    return cv2.warpAffine(base, T[:2].astype(np.float32), (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _loop_err(out):
    # all outputs must share one framing; report max offset between them
    on = (out.detach().clamp(0, 1).numpy() * 255).astype(np.float32)
    ds = lambda x: cv2.resize(x, (80, 80), interpolation=cv2.INTER_AREA)
    anchor = ds(on[0])
    return max(float(np.abs(ds(on[i]) - anchor).mean()) for i in range(on.shape[0]))


def _down(x):
    return cv2.resize((x.detach().clamp(0, 1).numpy() * 255).astype(np.float32), (80, 80),
                      interpolation=cv2.INTER_AREA)


def test_align_preserves_shape_dtype_device():
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert out.shape == images.shape
    assert out.dtype == images.dtype
    assert out.device == images.device


def test_align_single_frame_is_identity():
    node = SeamlessLoopAutoAlignFLV()
    images = torch.rand(1, 96, 128, 3)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic")[0]
    assert torch.equal(out, images)


def test_align_no_drift_is_identity():
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    images = images[0].repeat(images.shape[0], 1, 1, 1)  # identical frames -> no drift
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert out.shape == images.shape
    assert np.abs((out - images).numpy()).mean() < 0.02


def test_align_closes_loop():
    # Both drift directions must close the loop to a single framing with both
    # interpolation modes. Anisotropic needs affine, so similarity over x-only
    # zoom is tested separately.
    node = SeamlessLoopAutoAlignFLV()
    for images in (make_first_zoomed_in(), make_last_zoomed_in()):
        for transform, interp in (("affine", "log"), ("affine", "linear"), ("similarity", "log")):
            out = node.apply(images, 800, 15, "auto", transform, interp, "bicubic", drop_last_frames=0)[0]
            assert _loop_err(out) < 2.5, (transform, interp)


def test_align_anisotropic_zoom():
    # x-only zoom needs affine: loop must close; similarity is expected to fail.
    node = SeamlessLoopAutoAlignFLV()
    images = make_anisotropic()
    assert _loop_err(node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]) < 2.5
    assert _loop_err(node.apply(images, 800, 15, "auto", "similarity", "log", "bicubic", drop_last_frames=0)[0]) > 3.0


def test_align_shear_drift():
    # A pure shear keeps det == 1 (area-preserving) so, like a pan, it is a
    # near-tie for the auto-anchor det test; the loop must still close. Shear is
    # affine (6 dof) so 'affine' registers it exactly; similarity (4 dof: uniform
    # scale + rotation + translation) cannot represent it and must fail.
    node = SeamlessLoopAutoAlignFLV()
    h = w = 384
    base = make_scene(h)
    n = 11
    frames = [shear_about_center(base, 0.06 * (i / (n - 1) - 0.5), 0.04 * (i / (n - 1) - 0.5), h, w)
              for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    assert _loop_err(node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]) < 2.5
    assert _loop_err(node.apply(images, 800, 15, "auto", "similarity", "log", "bicubic", drop_last_frames=0)[0]) > 3.0


def test_align_featureless_frame_passes_through():
    # Featureless (blank) frames provide no features to align: pass through.
    node = SeamlessLoopAutoAlignFLV()
    images = torch.full((5, 96, 96, 3), 0.5)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic")[0]
    assert out.shape[0] < images.shape[0] or torch.allclose(out, images)


def test_align_drop_last_frame_closes_loop():
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=1)[0]
    assert out.shape[0] == images.shape[0] - 1
    assert np.abs(_down(out[0]) - _down(out[-1])).mean() < 2.0


def test_align_forced_anchor_frame():
    # 'first'/'last' must override the auto decision and close the loop to that
    # endpoint's framing, in both zoom directions.
    node = SeamlessLoopAutoAlignFLV()
    for images in (make_first_zoomed_in(), make_last_zoomed_in()):
        for anchor in ("first", "last"):
            out = node.apply(images, 800, 15, anchor, "affine", "log", "bicubic", drop_last_frames=0)[0]
            assert _loop_err(out) < 2.5, (anchor,)


def test_align_mix_first_last_blend():
    # With drop_last_frames=1 the dropped last frame must be blended 50/50 into
    # the first output frame for a seamless loop wrap.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    full = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic",
                     drop_last_frames=1, mix_first_last="on")[0]
    assert out.shape[0] == images.shape[0] - 1
    expected = (full[0].float() + full[-1].float()) * 0.5
    assert torch.allclose(out[0], expected, atol=1e-5), "first frame is not the 50/50 blend"
    assert torch.allclose(out[1:], full[1:-1], atol=1e-5), "other frames changed"


def test_align_mix_first_last_gated():
    # The blend must be ignored unless drop_last_frames == 1.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    key = dict(max_features=800, min_matches=15, anchor_frame="auto", transform="affine",
               interp="log", upscale_method="bicubic")
    for drop in (0, 2):
        on = node.apply(images, drop_last_frames=drop, mix_first_last="on", **key)[0]
        off = node.apply(images, drop_last_frames=drop, mix_first_last="off", **key)[0]
        assert torch.equal(on, off), drop


def test_align_auto_anchor_direction_blackbox():
    # Pure black box: only inputs and outputs, no internals. 'auto' must anchor
    # to the MORE zoomed-in endpoint (largest object). The anchor itself is the
    # identity warp, so it shows up as the one output frame that is byte-for-byte
    # unchanged while the far endpoint is re-registered into that framing.
    node = SeamlessLoopAutoAlignFLV()
    key = dict(max_features=800, min_matches=15, anchor_frame="auto", transform="affine",
               interp="log", upscale_method="bicubic")

    # FIRST is most zoomed-in (largest object) -> auto anchors to it.
    im = make_first_zoomed_in()
    out = node.apply(im, drop_last_frames=0, **key)[0]
    assert out.shape == im.shape
    assert torch.equal(out[0], im[0]), "auto should anchor to FIRST (most zoomed-in)"
    assert np.abs((out[-1] - im[-1]).float().numpy()).mean() > 0.02, "last frame should be re-warped"
    _save_png("first_zoomed_in__input_first.png", im[0])
    _save_png("first_zoomed_in__input_last.png", im[-1])
    _save_png("first_zoomed_in__output_first.png", out[0])
    _save_png("first_zoomed_in__output_last.png", out[-1])

    # LAST is most zoomed-in (largest object) -> auto anchors to it.
    im = make_last_zoomed_in()
    out = node.apply(im, drop_last_frames=0, **key)[0]
    assert out.shape == im.shape
    assert torch.equal(out[-1], im[-1]), "auto should anchor to LAST (most zoomed-in)"
    assert np.abs((out[0] - im[0]).float().numpy()).mean() > 0.02, "first frame should be re-warped"
    _save_png("last_zoomed_in__input_first.png", im[0])
    _save_png("last_zoomed_in__input_last.png", im[-1])
    _save_png("last_zoomed_in__output_first.png", out[0])
    _save_png("last_zoomed_in__output_last.png", out[-1])


# ---------------------------------------------------------------------------
# Edge cases & robustness (README: "Edge cases & fallbacks", parameter table)
# ---------------------------------------------------------------------------

def test_align_zero_frames():
    # README: "1 frame, or 0 frames: returned unchanged". An empty batch must
    # come back empty without touching the detection path.
    node = SeamlessLoopAutoAlignFLV()
    images = torch.zeros(0, 128, 128, 3)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic")[0]
    assert out.shape == (0, 128, 128, 3)


def test_align_two_frames_minimal_batch():
    # Smallest meaningful batch: f = i/(n-1) hits exactly 0 and 1; the whole
    # pipeline (match -> RANSAC -> interpolate -> warp) must still run.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in(n=2)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert out.shape == images.shape
    assert _loop_err(out) < 3.0


def test_align_drop_clamped_to_n_minus_1():
    # README: drop_last_frames >= n is clamped to n-1 so one frame always remains.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in(n=11)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=100)[0]
    assert out.shape[0] == 1  # drop clamps to n-1 = 10 -> n - drop == 1


def test_align_min_matches_gate_pass_through():
    # README: below min_matches confident matches the batch passes through
    # unchanged. Forcing the gate to the max makes any rich scene fail it.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in(n=11)
    out = node.apply(images, 800, 500, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert torch.equal(out, images), "batch should be returned unchanged when the match gate fails"


def test_align_mismatched_endpoints_pass_through():
    # Endpoint frames with no shared structure yield (near-)zero confident
    # matches -> pass-through, not a bogus warp of the whole batch. (Pure noise
    # vs a structured scene: cross-scene ORB matches are negligible.)
    node = SeamlessLoopAutoAlignFLV()
    size = 256
    first = make_scene(size, seed=1)
    rng = np.random.RandomState(7)
    last = rng.randint(0, 256, (size, size, 3)).astype(np.uint8)
    mid = np.stack([first] * 4 + [last])
    images = torch.from_numpy(mid.astype(np.float32) / 255.0)
    out = node.apply(images, 500, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert torch.equal(out, images) or out.shape[0] < images.shape[0]


def test_align_preserves_dtype_float64():
    # README: input dtype/device round-trip. float64 in -> float64 out (CPU).
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in().to(torch.float64)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert out.dtype == torch.float64
    assert out.device == images.device


def test_align_all_upscale_methods():
    # Every resampling filter in the parameter list must run, preserve shape,
    # emit finite pixels, and still close the loop (coarse filters allowed to
    # be blockier, hence the lenient bound).
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    for method in SeamlessLoopAutoAlignFLV.upscale_methods:
        out = node.apply(images, 800, 15, "auto", "affine", "log", method, drop_last_frames=0)[0]
        assert out.shape == images.shape, method
        assert torch.isfinite(out).all(), method
        assert _loop_err(out) < 5.0, (method, _loop_err(out))


def test_align_nonsquare_images():
    # H != W exercises the min(h,w)//3 centring box and the h x w res-field grid.
    node = SeamlessLoopAutoAlignFLV()
    h, w, n, drift = 320, 480, 11, 0.05
    base = make_scene_rect(h, w, seed=5)
    frames = [scale_about_center(base, 1.0 + drift * (i / (n - 1)), 1.0 + drift * (i / (n - 1)), h, w)
              for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]
    assert out.shape == images.shape
    assert _loop_err(out) < 3.0


def test_align_rotation_drift():
    # A pure in-plane rotation (no scale) must be registered. Similarity holds 4
    # dof (uniform scale + rotation + tx + ty) and must cope; affine as well.
    node = SeamlessLoopAutoAlignFLV()
    h = w = 384
    base = make_scene(h)
    n = 11
    frames = [rotate_about_center(base, (i / (n - 1) - 0.5) * 16.0, h, w) for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    for transform in ("affine", "similarity"):
        out = node.apply(images, 800, 15, "auto", transform, "log", "bicubic", drop_last_frames=0)[0]
        assert _loop_err(out) < 3.5, (transform, _loop_err(out))


def test_align_pan_drift():
    # A pure translation (no scale, no rotation) must be registered. detG ~= 1
    # so auto-anchor is decided by a near-tie; the loop must still close either way.
    # Magnitude is kept modest: large pans shift different border content into the
    # frame (border-replicated), which raises the whole-image mean without being a
    # registration failure.
    node = SeamlessLoopAutoAlignFLV()
    h = w = 384
    base = make_scene(h)
    n = 11
    frames = [pan(base, (i / (n - 1) - 0.5) * 20.0, (i / (n - 1) - 0.5) * 14.0, h, w) for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    for transform in ("affine", "similarity"):
        out = node.apply(images, 800, 15, "auto", transform, "log", "bicubic", drop_last_frames=0)[0]
        assert _loop_err(out) < 1.0, (transform, _loop_err(out))


def test_align_output_finite():
    # No NaN/Inf may leak out of the eig-based log/exp path.
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    for interp in ("log", "linear"):
        out = node.apply(images, 800, 15, "auto", "affine", interp, "bicubic", drop_last_frames=0)[0]
        assert torch.isfinite(out).all(), interp


def test_align_deterministic():
    # Same input + params -> byte-identical output (ORB/RANSAC are deterministic).
    node = SeamlessLoopAutoAlignFLV()
    images = make_first_zoomed_in()
    key = dict(max_features=800, min_matches=15, anchor_frame="auto", transform="affine",
               interp="log", upscale_method="bicubic")
    o1 = node.apply(images, **key)[0]
    o2 = node.apply(images, **key)[0]
    assert torch.equal(o1, o2)


def test_align_corrects_drift():
    # Strong correctness check: a colour-coded marker must land at (nearly) the
    # same pixel in every aligned frame. Before alignment its centre follows the
    # zoom; after alignment the drift is removed, so the marker is stationary.
    node = SeamlessLoopAutoAlignFLV()
    h = w = 384
    base = make_scene(w)
    cy, cx = 140, 240
    cv2.rectangle(base, (cx - 4, cy - 40), (cx + 4, cy + 40), (255, 0, 0), -1)
    cv2.rectangle(base, (cx - 40, cy - 4), (cx + 40, cy + 4), (255, 0, 0), -1)
    n = 11
    frames = [magnify(base, 1.0 + 0.08 * (i / (n - 1)), w) for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    out = node.apply(images, 1200, 20, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]

    def marker_centroid(img):
        a = (img.detach().clamp(0, 1).numpy() * 255).astype(np.int32)
        # the marker was drawn (255,0,0) into an RGB array -> red: R>180, G,B<90
        m = (a[:, :, 0] > 180) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
        ys, xs = np.nonzero(m)
        assert xs.size >= 20, f"marker not found ({xs.size} px); centroid test is invalid"
        return xs.mean(), ys.mean()

    cents = [marker_centroid(out[i]) for i in range(n)]
    ref = cents[0]
    spread = max(float(np.hypot(c[0] - ref[0], c[1] - ref[1])) for c in cents)
    assert spread < 4.0, (spread, cents)


def test_align_combined_zoom_rotate_pan():
    # Real footage drifts several ways at once, and a single affine can represent
    # zoom + rotation + translation together. Unlike pure rotation/pan (which push
    # different border content into the frame), the zoom-in keeps every sample
    # strictly inside the source, so the composite motion must register *nearly*
    # exactly. Uber test: rectangular image + zoom + rotation + pan.
    node = SeamlessLoopAutoAlignFLV()
    h, w, n = 320, 480, 11
    base = make_scene_rect(h, w, seed=11)
    cx, cy = w // 2, h // 2
    cv2.rectangle(base, (cx - 4, cy - 40), (cx + 4, cy + 40), (255, 0, 0), -1)
    cv2.rectangle(base, (cx - 40, cy - 4), (cx + 40, cy + 4), (255, 0, 0), -1)
    frames = [combined_zoom_pan_rotate(base, i / (n - 1), h, w) for i in range(n)]
    images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    out = node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0]

    # (1) every output frame shares one framing (loop closes)
    assert out.shape == images.shape
    assert _loop_err(out) < 2.0, _loop_err(out)

    # (2) the marker must be (nearly) stationary -> the drift is actually removed,
    #     not merely consistent frame-to-frame.
    def marker_centroid(img):
        a = (img.detach().clamp(0, 1).numpy() * 255).astype(np.int32)
        m = (a[:, :, 0] > 180) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
        ys, xs = np.nonzero(m)
        assert xs.size >= 20, f"marker not found ({xs.size} px); centroid test is invalid"
        return xs.mean(), ys.mean()

    cents = [marker_centroid(out[i]) for i in range(n)]
    ref = cents[0]
    spread = max(float(np.hypot(c[0] - ref[0], c[1] - ref[1])) for c in cents)
    assert spread < 4.0, (spread, cents)

    # eyes-on review: input first/last vs output first/last
    _save_png("combined__input_first.png", images[0])
    _save_png("combined__input_last.png", images[-1])
    _save_png("combined__output_first.png", out[0])
    _save_png("combined__output_last.png", out[-1])


# ---------------------------------------------------------------------------
# Radial distortion: apply -> remove round-trip (ground-truth k, no solver yet)
# ---------------------------------------------------------------------------

def test_radial_roundtrip():
    # Applying a radial distortion and then removing it with the SAME k must
    # return the original image. Exercises both barrel (k<0) and pincushion
    # (k>0) at 2%, 5% and 10%. k is the corner displacement fraction (the
    # model normalises so the image corner sits at radius 1, so k=0.1 means
    # the corner moves 10% of the half-diagonal).
    base = make_scene(256, seed=3)
    cases = [
        ("pincushion", 0.10),
        ("pincushion", 0.05),
        ("pincushion", 0.02),
        ("barrel", -0.02),
        ("barrel", -0.05),
        ("barrel", -0.10),
    ]
    for label, k in cases:
        distorted = apply_distortion(base, k)
        restored = remove_distortion(distorted, k)
        restored_err = float(np.abs(restored.astype(np.int32) - base.astype(np.int32)).mean())
        # The inverse must bring us back to (near) the original. At 10% the
        # distortion moves corners ~10% of the half-diagonal (~10-20px), so a
        # wrong/inverted k would leave errors of many pixels; the <1.8 bound
        # (the residual is bicubic ringing at the sharp quad edges) confirms
        # the removal works.
        assert restored_err < 1.8, (label, k, restored_err)

        os.makedirs(ART, exist_ok=True)
        stem = f"radial_{label}_{abs(k) * 100:.0f}pct"
        for tag, img in (("_distorted", distorted), ("_restored", restored)):
            cv2.imwrite(os.path.join(ART, stem + tag + ".png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def test_radial_solve_recovers_k():
    # Solve for k from first/last-frame combos across the same range, using NO
    # ground truth (global-image alignment, not keypoint radii -- the periphery
    # of a distorted frame won't yield ORB correspondences). The recovered k
    # must be close to the true k, and removing distortion with the *solved* k
    # must bring the frame back to the reference.
    base = make_scene(256, seed=3)
    cases = [
        ("pincushion", 0.10),
        ("pincushion", 0.05),
        ("pincushion", 0.02),
        ("barrel", -0.02),
        ("barrel", -0.05),
        ("barrel", -0.10),
    ]
    for label, k_gt in cases:
        # first = clean reference, last = distorted
        drifted = apply_distortion(base, k_gt)
        k_est = solve_radial(base, drifted)

        assert abs(k_est - k_gt) < 0.02, (label, k_gt, k_est)

        # removal with the SOLVED k (not ground truth) recovers the reference
        restored = remove_distortion(drifted, k_est)
        restored_err = float(np.abs(restored.astype(np.int32) - base.astype(np.int32)).mean())
        assert restored_err < 3.0, (label, k_gt, k_est, restored_err)


def test_radial_corrects_drift():
    # End-to-end, like the pan/zoom correctness tests: given first (clean) and
    # last (radially distorted) frames, solve for k, remove it from the last
    # frame, and check the drift is actually gone -- the whole-frame error vs
    # the reference collapses and an off-centre marker becomes stationary.
    base = make_scene(256, seed=3)
    cy, cx = 96, 176
    cv2.rectangle(base, (cx - 4, cy - 40), (cx + 4, cy + 40), (255, 0, 0), -1)
    cv2.rectangle(base, (cx - 40, cy - 4), (cx + 40, cy + 4), (255, 0, 0), -1)

    def marker_centroid(img):
        a = img.astype(np.int32)
        m = (a[:, :, 0] > 180) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
        ys, xs = np.nonzero(m)
        assert xs.size >= 20, f"marker not found ({xs.size} px)"
        return xs.mean(), ys.mean()

    for k_gt in (0.10, 0.05, -0.05, -0.10):
        first = base
        last = apply_distortion(base, k_gt)
        c_first = marker_centroid(first)

        k_est = solve_radial(first, last)
        corrected = remove_distortion(last, k_est)

        before = float(np.abs(last.astype(np.int32) - first.astype(np.int32)).mean())
        after = float(np.abs(corrected.astype(np.int32) - first.astype(np.int32)).mean())
        assert abs(k_est - k_gt) < 0.02, (k_gt, k_est)
        # error removed: corrected last is near the reference, and clearly
        # closer than the raw distorted last was.
        assert after < 3.0, (k_gt, k_est, after)
        assert after < before, (k_gt, k_est, before, after)

        # off-centre marker is (nearly) stationary after correction
        c_corr = marker_centroid(corrected)
        spread = float(np.hypot(c_corr[0] - c_first[0], c_corr[1] - c_first[1]))
        assert spread < 3.0, (k_gt, spread)

        os.makedirs(ART, exist_ok=True)
        stem = f"radial_drift_{abs(k_gt) * 100:.0f}pct_{'pincushion' if k_gt > 0 else 'barrel'}"
        cv2.imwrite(os.path.join(ART, stem + "_first.png"), cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(ART, stem + "_last.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(ART, stem + "_corrected.png"), cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))


def test_align_radial_closes_loop():
    # Plumbed into the node: with transform="radial", a batch that drifts only
    # by radial distortion must close the loop to the first frame's framing.
    # 'radial' corrects ONLY radial (no affine), and 'affine' corrects ONLY
    # affine -- they are mutually exclusive. (The affine mode can PARTIALLY mask
    # radial as a zoom, and vice-versa, so we compare relative residual rather
    # than an absolute bound.)
    node = SeamlessLoopAutoAlignFLV()
    base = make_scene(384, seed=3)
    n = 11
    for k in (0.10, 0.05, -0.05, -0.10):
        frames = [apply_distortion(base, k * (i / (n - 1))) for i in range(n)]
        images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)

        rad = _loop_err(node.apply(images, 800, 15, "auto", "radial", "log", "bicubic", drop_last_frames=0)[0])
        aff = _loop_err(node.apply(images, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0])
        assert rad < 2.5, (k, rad)          # radial mode removes the radial drift
        assert aff > rad + 0.5, (k, rad, aff)  # affine mode leaves most of it

    # radial does NOT fix a pure affine zoom
    zoom = make_first_zoomed_in(n=n, size=384)
    rad = _loop_err(node.apply(zoom, 800, 15, "auto", "radial", "log", "bicubic", drop_last_frames=0)[0])
    aff = _loop_err(node.apply(zoom, 800, 15, "auto", "affine", "log", "bicubic", drop_last_frames=0)[0])
    assert aff < 2.5, aff                    # affine mode fixes the zoom
    assert rad > 3.0, rad                    # radial mode cannot


# ---------------------------------------------------------------------------
# Numerical error baseline: report the loop-closure error for every drift type
# and assert it stays within +1% of the recorded baseline, so any change that
# makes registration worse is caught immediately.
# ---------------------------------------------------------------------------

def test_align_numerical_error_baseline():
    node = SeamlessLoopAutoAlignFLV()
    key = dict(max_features=800, min_matches=15, anchor_frame="auto",
               transform="affine", interp="log", upscale_method="bicubic")

    cases = []

    # zoom in / out / anisotropic, both interpolation modes
    for name, im in (("zoom-in", make_first_zoomed_in()),
                     ("zoom-out", make_last_zoomed_in()),
                     ("anisotropic", make_anisotropic())):
        for interp in ("log", "linear"):
            out = node.apply(im, drop_last_frames=0, **dict(key, interp=interp))[0]
            cases.append((f"{name} ({interp})", _loop_err(out)))

    # shear
    h = w = 384
    base = make_scene(h)
    n = 11
    frames = [shear_about_center(base, 0.06 * (i / (n - 1) - 0.5), 0.04 * (i / (n - 1) - 0.5), h, w)
              for i in range(n)]
    im = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    cases.append(("shear (log)", _loop_err(node.apply(im, drop_last_frames=0, **key)[0])))

    # rotation
    frames = [rotate_about_center(base, (i / (n - 1) - 0.5) * 16.0, h, w) for i in range(n)]
    im = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    cases.append(("rotation (log)", _loop_err(node.apply(im, drop_last_frames=0, **key)[0])))

    # pan
    frames = [pan(base, (i / (n - 1) - 0.5) * 20.0, (i / (n - 1) - 0.5) * 14.0, h, w) for i in range(n)]
    im = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    cases.append(("pan (log)", _loop_err(node.apply(im, drop_last_frames=0, **key)[0])))

    # combined zoom + rotate + pan
    h, w, n = 320, 480, 11
    base = make_scene_rect(h, w, seed=11)
    cx, cy = w // 2, h // 2
    cv2.rectangle(base, (cx - 4, cy - 40), (cx + 4, cy + 40), (255, 0, 0), -1)
    cv2.rectangle(base, (cx - 40, cy - 4), (cx + 40, cy + 4), (255, 0, 0), -1)
    frames = [combined_zoom_pan_rotate(base, i / (n - 1), h, w) for i in range(n)]
    im = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    cases.append(("combined (log)", _loop_err(node.apply(im, drop_last_frames=0, **key)[0])))

    # radial
    base = make_scene(384, seed=3)
    n = 11
    radial_key = dict(key, transform="radial")
    for k in (0.10, 0.05, -0.05, -0.10):
        frames = [apply_distortion(base, k * (i / (n - 1))) for i in range(n)]
        im = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        out = node.apply(im, drop_last_frames=0, **radial_key)[0]
        cases.append((f"radial k={k} (log)", _loop_err(out)))

    baseline = {
        "zoom-in (log)": 0.3014,
        "zoom-in (linear)": 0.3014,
        "zoom-out (log)": 0.2798,
        "zoom-out (linear)": 0.2798,
        "anisotropic (log)": 0.2005,
        "anisotropic (linear)": 0.2005,
        "shear (log)": 0.7753,
        "rotation (log)": 2.2579,
        "pan (log)": 0.3004,
        "combined (log)": 1.5028,
        "radial k=0.1 (log)": 0.3924,
        "radial k=0.05 (log)": 0.5431,
        "radial k=-0.05 (log)": 0.5802,
        "radial k=-0.1 (log)": 0.1044,
    }

    print("\n=== numerical error (loop_err, lower is better) ===")
    for name, err in cases:
        ref = baseline[name]
        print(f"  {name:24s} {err:.4f}  (delta {err - ref:+.4f})")
        assert err <= ref * 1.01, f"{name}: {err:.4f} exceeds baseline {ref:.4f} by >1%"
