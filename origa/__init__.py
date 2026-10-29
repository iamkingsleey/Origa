# Origa — shape paper inside Blender, with first-class tools for money
# Turns one flat sheet into folded, rolled, curled, arched, waving or crumpled
# paper. Built-in banknote sizes (Dollar, Naira, Euro, Pound), flying-bill and
# rolled-cash presets, and a Money Rain generator — plus receipts, letters,
# posters, scrolls and anything else sheet-like.
# Everything stays live: change a slider, the sheet rebuilds. Paper length is
# preserved along the bend axis, so folds and rolls behave like real paper.
#
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Origa",
    "author": "Kingsley Mkpandiok",
    "version": (0, 5, 3),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Origa  |  Add > Mesh > Paper Sheet",
    "description": "Turn a plane mesh into paper: fold, roll, curl, arch, wave, crumple. Made for money: banknote sizes, flying & rolled cash, Money Rain",
    "category": "Add Mesh",
}

import math
import random

import bpy
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from mathutils import Vector, noise

_SUSPEND = False  # batch property edits without rebuilding each time


# --------------------------------------------------------------------------
# Sheet sizes (width:height ratios of real things)
# --------------------------------------------------------------------------
SIZES = {
    "USD":    ("US Dollar",    156.0, 66.0),
    "NAIRA":  ("Naira",        151.0, 78.0),
    "EURO":   ("Euro (€50)",   140.0, 77.0),
    "GBP":    ("Pound (£20)",  139.0, 73.0),
    "JPY":    ("Yen",          160.0, 76.0),
    "GHS":    ("Cedi",         145.0, 70.0),
    "KES":    ("Kenyan Shilling", 157.0, 76.0),
    "ZAR":    ("Rand",         150.0, 70.0),
    "INR":    ("Rupee",        150.0, 66.0),
    "A4":     ("A4 Paper",     297.0, 210.0),
    "SQUARE": ("Square",       150.0, 150.0),
    "RECEIPT": ("Receipt",     240.0, 80.0),
    "RECEIPT_LONG": ("Long Receipt", 560.0, 80.0),
}


def _apply_size(p):
    if p.size_preset == "CUSTOM":
        return
    _, w, h = SIZES[p.size_preset]
    p["width"] = p.length * h / w


# --------------------------------------------------------------------------
# Core maths
# --------------------------------------------------------------------------
def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _roll_angle(t, r0, gap, rmin):
    """Angle turned after travelling t along an Archimedean spiral that starts
    at radius r0 and tightens by `gap` per turn."""
    if t <= 0.0:
        return 0.0
    b = max(gap, 1e-6) / (2.0 * math.pi)
    # spiral reaches rmin at phi_c
    phi_c = max((r0 - rmin) / b, 0.0)
    t_c = r0 * phi_c - 0.5 * b * phi_c * phi_c
    if t < t_c:
        return (r0 - math.sqrt(max(r0 * r0 - 2.0 * b * t, 0.0))) / b
    return phi_c + (t - t_c) / max(rmin, 1e-4)


def _arch_segments(L, p):
    """Arc-length segments (length, theta_from, theta_to) for a run of tall
    rounded arches: rise, up, round over, down, round under, ... settle flat."""
    r = max(p.arch_width * L * 0.5, 1e-4)
    lean = math.radians(p.arch_lean)
    up = math.pi / 2 - lean
    down = up - math.pi
    segs = [(p.arch_start * L, 0.0, 0.0)]           # flat lead-in
    segs.append((r * up, 0.0, up))                    # rise
    for k in range(p.arch_count):
        h = max(p.arch_height * L * (p.arch_decay ** k), 0.0)
        segs.append((h, up, up))                      # straight up
        segs.append((math.pi * r, up, down))          # round over the top
        segs.append((h, down, down))                  # straight down
        if k < p.arch_count - 1:
            segs.append((math.pi * r, down, up))      # round under
    segs.append((r * abs(down), down, 0.0))           # settle back to flat
    return segs


def _arch_theta(s, L, p):
    pos = 0.0
    for length, a, b in _arch_segments(L, p):
        if s <= pos + length:
            t = (s - pos) / length if length > 1e-9 else 1.0
            return a + (b - a) * t
        pos += length
    return 0.0


def _fit_radius(r0, gap, rmin, span):
    """Grow the roll's outer radius just enough that all the rolled paper fits
    in a spiral down to the core, so turns never land on top of each other."""
    b = max(gap, 1e-6) / (2.0 * math.pi)
    return max(r0, math.sqrt(rmin * rmin + 2.0 * b * span))


def _theta(s, L, p, phase, rad_k=1.0):
    """Tangent angle of the paper's centre-line at arc length s."""
    th = 0.0

    # Wave / flutter (flying money)
    if p.wave_amount:
        th += math.radians(p.wave_amount) * math.sin(
            2.0 * math.pi * p.wave_count * s / L + phase)

    # Folds / creases
    n = p.fold_count
    if n and p.fold_angle:
        a = math.radians(p.fold_angle)
        w = max(p.crease_softness * L / (n + 1), 1e-4)
        for i in range(1, n + 1):
            s0 = L * i / (n + 1) + p.fold_offset * L / (n + 1)
            if p.fold_style == "ACCORDION":
                sign = 1.0 if i % 2 else -1.0
            else:
                sign = 1.0
            if p.fold_flip:
                sign = -sign
            th += sign * a * _smooth((s - s0) / w + 0.5)

    # Curled edges (both ends lift)
    if p.edge_curl:
        e = max(p.edge_curl_size * L, 1e-4)
        c = math.radians(p.edge_curl)
        if s < e:
            th -= c * (1.0 - s / e) ** 2
        if s > L - e:
            th += c * ((s - (L - e)) / e) ** 2

    # Separate curl at each end (receipt tops, lifted tails)
    if p.curl_start:
        e = max(p.curl_start_size * L, 1e-4)
        if s < e:
            th -= math.radians(p.curl_start) * (1.0 - s / e) ** 2
    if p.curl_end:
        e = max(p.curl_end_size * L, 1e-4)
        if s > L - e:
            th += math.radians(p.curl_end) * ((s - (L - e)) / e) ** 2

    # Lift: the sheet turns and stands up partway along
    if p.lift_angle:
        w = max(p.lift_softness * L, 1e-4)
        th += math.radians(p.lift_angle) * _smooth((s - p.lift_at * L) / w + 0.5)

    # Arches: tall standing loops (receipt piling up)
    if p.arch_count:
        th += _arch_theta(s, L, p)

    # Roll (scroll / rolled cash)
    if p.roll_amount > 0.0:
        r0 = p.roll_radius * rad_k
        # layers can never sit closer than the paper is thick
        gap = max(p.roll_tightness * r0, p.thickness * 1.25, L * 0.0015)
        rmin = 0.3 * r0
        span = L * p.roll_amount
        span_end = min(span * (1.0 + p.roll_balance), L)
        span_start = min(span * (1.0 - p.roll_balance), L)
        if p.roll_from != "BOTH":
            span_end = span_start = min(span, L)
        sign = -1.0 if p.roll_flip else 1.0
        if p.roll_from in ("END", "BOTH"):
            th += sign * _roll_angle(s - (L - span_end), _fit_radius(r0, gap, rmin, span_end), gap, rmin)
        if p.roll_from in ("START", "BOTH"):
            th -= sign * _roll_angle(span_start - s, _fit_radius(r0, gap, rmin, span_start), gap, rmin)

    return th


def _tear(s, L, W, p, side):
    """How far a torn side edge is eaten in at arc length s."""
    if p.tear_depth <= 0.0:
        return 0.0
    x = s / L * p.tear_detail
    o = side * 13.7 + p.seed * 3.1
    n = noise.noise(Vector((x, o, 0.3)))
    hi = noise.noise(Vector((x * 5.3, o + 7.1, 1.1)))
    notch = max(0.0, noise.noise(Vector((x * 2.2, o + 2.9, 4.7))) - 0.25) * 3.0
    d = 0.15 + max(0.0, 0.5 + n) + 0.3 * hi + notch
    return max(0.0, d) * p.tear_depth * W


def _integrate(thetas, ds):
    cx, cz = [0.0], [0.0]
    for k in range(len(thetas) - 1):
        ta, tb = thetas[k], thetas[k + 1]
        cx.append(cx[-1] + 0.5 * (math.cos(ta) + math.cos(tb)) * ds)
        cz.append(cz[-1] + 0.5 * (math.sin(ta) + math.sin(tb)) * ds)
    return cx, cz


# Detail levels: max bend per edge loop (degrees), longest flat span (share of
# length), loops across a cupped/skewed width, crumple grid (along, across).
DETAIL = {
    "LOW":    dict(angle=15.0, span=1 / 3, across=4, crumple=(40, 10), anim=24),
    "MEDIUM": dict(angle=8.0, span=1 / 6, across=8, crumple=(80, 20), anim=48),
    "HIGH":   dict(angle=4.0, span=1 / 12, across=12, crumple=(160, 40), anim=96),
}


def _is_animated(obj):
    p = obj.origa
    return bool(p.flutter) or _has_origa_keys(obj)


def _pick_rows(thetas, ds, L, max_angle, max_len, twist_rate):
    """Edge loops only where the paper turns: accumulate bend (plus twist)
    and drop a loop each time it passes max_angle, or after max_len."""
    last = len(thetas) - 1
    ks, acc_a, acc_l = [0], 0.0, 0.0
    for k in range(1, last + 1):
        acc_a += abs(thetas[k] - thetas[k - 1]) + twist_rate * ds
        acc_l += ds
        if acc_a >= max_angle or acc_l >= max_len - 1e-9:
            ks.append(k)
            acc_a = acc_l = 0.0
    if ks[-1] != last:
        if last - ks[-1] < 3 and len(ks) > 1:
            ks[-1] = last
        else:
            ks.append(last)
    return ks


def build_sheet(obj, frame=None):
    p = obj.origa
    if not p.is_paper or obj.type != "MESH":
        return

    long_l, short_w = p.length, p.width
    if p.bend_axis == "LENGTH":
        L, W = long_l, short_w
    else:
        L, W = short_w, long_l

    uniform = p.detail == "UNIFORM"
    lvl = DETAIL.get(p.detail, DETAIL["MEDIUM"])
    animated = _is_animated(obj)

    if uniform:
        nx = max(4, p.res_bend)
        ny = max(1, p.res_cross)
    else:
        # across the width: a straight line unless something bends it that way
        ny = 1
        if abs(p.cross_bend) > 1e-3:
            ny = max(ny, math.ceil(abs(p.cross_bend) / lvl["angle"]))
        if p.roll_amount > 0 and (abs(p.roll_skew) > 1e-4 or abs(p.roll_flare) > 1e-4):
            ny = max(ny, lvl["across"])
        if p.crumple:
            ny = max(ny, lvl["crumple"][1])
        ny = max(ny, p.cross_loops)
        anim_rows = max(lvl["anim"], lvl["crumple"][0] if p.crumple else 0)
        nx = anim_rows if animated else 300   # 300 = dense sampling, thinned below

    phase = math.radians(p.wave_phase)
    if p.flutter and frame is not None:
        phase += frame * p.flutter_speed * 0.1

    # ---- integrate the centre-line in the XZ plane (arc-length exact) ----
    sub = 8
    m = nx * sub
    ds = L / m
    thetas = [_theta(k * ds, L, p, phase) for k in range(m + 1)]
    cx, cz = _integrate(thetas, ds)

    if uniform or animated:
        # fixed loop count: steady topology for animation and uniform mode
        ks = [i * sub for i in range(nx + 1)]
    else:
        max_len = L * lvl["span"]
        if p.crumple:
            max_len = min(max_len, L / lvl["crumple"][0])
        if p.tear_depth > 0:
            max_len = min(max_len, L / max(p.tear_detail * 5.0, 4.0))
        if p.wave_amount:
            max_len = min(max_len, L / max(p.wave_count * 8.0, 4.0))
        twist_rate = abs(math.radians(p.twist)) / L
        ks = _pick_rows(thetas, ds, L, math.radians(lvl["angle"]), max_len, twist_rate)

    # ---- anchor: keep one point fixed & flat so the sheet doesn't swing ----
    ak = {"START": 0, "CENTER": m // 2, "END": m, "GROUND": 0}[p.anchor]
    a0 = 0.0 if p.anchor == "GROUND" else thetas[ak]
    ca, sa = math.cos(-a0), math.sin(-a0)
    ox, oz = cx[ak], cz[ak]

    # ---- uneven roll: each row rolls a different amount (cone-shaped rolls) ----
    row_delta = None
    if p.roll_amount > 0.0 and (abs(p.roll_skew) > 1e-4 or abs(p.roll_flare) > 1e-4):
        row_delta = []
        for j in range(ny + 1):
            t = 2.0 * j / ny - 1.0
            f = 1.0 + p.roll_skew * t + p.roll_flare * (t * t - 0.33)
            th_j = [_theta(k * ds, L, p, phase, max(f, 0.1)) for k in range(m + 1)]
            jx, jz = _integrate(th_j, ds)
            jox, joz = jx[ak], jz[ak]
            row_delta.append([((jx[k] - jox) - (cx[k] - ox),
                               (jz[k] - joz) - (cz[k] - oz)) for k in ks])

    twist = math.radians(p.twist)
    cross = math.radians(p.cross_bend)
    kc = cross / W if abs(cross) > 1e-6 else 0.0
    seed_v = Vector((p.seed * 17.31, p.seed * 5.77, p.seed * 3.13))

    verts, uvs_per_vert = [], []
    for i, k in enumerate(ks):
        s = k * ds
        # centre point & frame (rotated to anchor)
        x0, z0 = cx[k] - ox, cz[k] - oz
        px, pz = x0 * ca - z0 * sa, x0 * sa + z0 * ca
        th = thetas[k] - a0
        T = Vector((math.cos(th), 0.0, math.sin(th)))
        B = Vector((0.0, 1.0, 0.0))
        N = Vector((-math.sin(th), 0.0, math.cos(th)))
        tau = twist * (s / L - 0.5)
        if p.flutter and frame is not None and p.twist_flutter:
            tau += math.radians(p.twist_flutter) * math.sin(
                2.0 * math.pi * s / L + phase * 1.37)
        Bt = B * math.cos(tau) + N * math.sin(tau)
        Nt = N * math.cos(tau) - B * math.sin(tau)
        C = Vector((px, 0.0, pz))

        v_lo = -W / 2.0 + _tear(s, L, W, p, 0)
        v_hi = W / 2.0 - _tear(s, L, W, p, 1)
        if v_hi - v_lo < W * 0.05:
            mid = 0.5 * (v_hi + v_lo)
            v_lo, v_hi = mid - W * 0.025, mid + W * 0.025
        for j in range(ny + 1):
            v = v_lo + (v_hi - v_lo) * j / ny
            if kc:
                off_b = math.sin(kc * v) / kc
                off_n = (1.0 - math.cos(kc * v)) / kc
            else:
                off_b, off_n = v, 0.0
            P = C + Bt * off_b + Nt * off_n
            if row_delta is not None:
                dx, dz = row_delta[j][i]
                P += Vector((dx * ca - dz * sa, 0.0, dx * sa + dz * ca))

            if p.crumple:
                q = Vector((s, v, 0.0)) * (p.crumple_scale / max(W, 1e-4)) + seed_v
                n1 = noise.noise(q)
                n2 = noise.noise(q * 2.7 + Vector((3.1, 1.7, 0.4)))
                d = n1 + 0.45 * n2
                if p.crumple_sharp:
                    d = (1.0 - p.crumple_sharp) * d + p.crumple_sharp * (0.6 - abs(d) * 1.6)
                P += Nt * d * p.crumple * W * 0.08

            if p.bend_axis == "LENGTH":
                verts.append((P.x, P.y, P.z))
                uvs_per_vert.append((s / L, (v + W / 2.0) / W))
            else:  # rotate so the long side still runs along X
                verts.append((P.y, -P.x, P.z))
                uvs_per_vert.append(((v + W / 2.0) / W, 1.0 - s / L))

    faces = []
    r = ny + 1
    for i in range(len(ks) - 1):
        for j in range(ny):
            a = i * r + j
            faces.append((a, a + r, a + r + 1, a + 1))

    me = obj.data
    me.clear_geometry()
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap") if not me.uv_layers else me.uv_layers[0]
    loop_uvs = []
    for poly in me.polygons:
        for li in poly.loop_indices:
            loop_uvs.extend(uvs_per_vert[me.loops[li].vertex_index])
    if p.uv_flip or p.uv_mirror:
        for q in range(0, len(loop_uvs), 2):
            if p.uv_flip:
                loop_uvs[q] = 1.0 - loop_uvs[q]
                loop_uvs[q + 1] = 1.0 - loop_uvs[q + 1]
            if p.uv_mirror:
                loop_uvs[q + 1] = 1.0 - loop_uvs[q + 1]
    uv.data.foreach_set("uv", loop_uvs)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()

    p["bend_len"] = L
    _sync_thickness(obj)
    _sync_smooth(obj)


def _sync_smooth(obj):
    p = obj.origa
    mod = obj.modifiers.get("Origa Smooth")
    if p.smooth_levels > 0:
        if mod is None:
            mod = obj.modifiers.new("Origa Smooth", "SUBSURF")
            # smooth first, then give it thickness
            idx = list(obj.modifiers).index(mod)
            if idx > 0:
                try:
                    obj.modifiers.move(idx, 0)
                except Exception:
                    try:
                        with bpy.context.temp_override(object=obj):
                            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
                    except Exception:
                        pass
        mod.levels = p.smooth_levels
        mod.render_levels = p.smooth_levels
    elif mod is not None:
        obj.modifiers.remove(mod)


def _sync_thickness(obj):
    p = obj.origa
    mod = obj.modifiers.get("Origa Thickness")
    if p.thickness > 0.0:
        if mod is None:
            mod = obj.modifiers.new("Origa Thickness", "SOLIDIFY")
            mod.offset = 0.0
            mod.use_even_offset = False   # even offset spikes on tight rolls
            mod.use_quality_normals = True
        mod.thickness = p.thickness
        mod.use_even_offset = False
    elif mod is not None:
        obj.modifiers.remove(mod)


def _upd(self, context):
    if _SUSPEND:
        return
    obj = self.id_data
    if self.flutter and context:
        _lock_for_render(context.scene)
    if self.size_preset != "CUSTOM":
        _apply_size(self)
    build_sheet(obj, context.scene.frame_current if context else None)


def _bend_len(p):
    return p.length if p.bend_axis == "LENGTH" else p.width


def _keep_shape(p):
    """Resizing the sheet scales the roll with it, so a roll stays rolled
    (same number of turns) instead of unrolling as the paper gets shorter."""
    old, new = p.bend_len, _bend_len(p)
    if p.keep_shape and old > 0 and abs(new - old) > 1e-9:
        p["roll_radius"] = max(p.roll_radius * new / old, 0.0005)


def _upd_size(self, context):
    if _SUSPEND:
        return
    _apply_size(self)
    _keep_shape(self)
    _upd(self, context)


def _upd_length(self, context):
    if _SUSPEND:
        return
    _apply_size(self)
    _keep_shape(self)
    _upd(self, context)


def _upd_width(self, context):
    global _SUSPEND
    if _SUSPEND:
        return
    if self.size_preset != "CUSTOM":
        # typing a width means the sheet is no longer a standard size
        _SUSPEND = True
        self.size_preset = "CUSTOM"
        _SUSPEND = False
    _keep_shape(self)
    _upd(self, context)


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------
class OrigaProps(bpy.types.PropertyGroup):
    is_paper: BoolProperty(default=False)

    # sheet
    size_preset: EnumProperty(
        name="Size",
        items=[(k, v[0], "") for k, v in SIZES.items()] + [("CUSTOM", "Custom", "")],
        default="USD", update=_upd_size)
    length: FloatProperty(name="Length", default=2.0, min=0.01, soft_max=20, unit="LENGTH", update=_upd_length)
    width: FloatProperty(name="Width", default=0.846, min=0.01, soft_max=20, unit="LENGTH", update=_upd_width)
    keep_shape: BoolProperty(
        name="Keep Shape When Resizing", default=True,
        description="Rolls scale with the sheet, so changing Length or Width keeps the same shape instead of unrolling")
    bend_len: FloatProperty(default=0.0, options={"HIDDEN"})
    thickness: FloatProperty(name="Thickness", default=0.006, min=0.0, soft_max=0.1,
                             precision=4, unit="LENGTH", update=_upd)
    uv_flip: BoolProperty(name="Flip Texture", default=False, update=_upd,
                          description="Run the image the other way along the sheet")
    uv_mirror: BoolProperty(name="Mirror Texture", default=False, update=_upd)
    detail: EnumProperty(
        name="Detail",
        items=[("LOW", "Low Poly", "Fewest faces: edge loops only where the paper bends"),
               ("MEDIUM", "Medium", "Smoother curves, still light"),
               ("HIGH", "High", "Very smooth curves"),
               ("UNIFORM", "Uniform Grid", "Even grid using the resolution values below")],
        default="LOW", update=_upd)
    smooth_levels: IntProperty(
        name="Smooth", default=0, min=0, max=3, update=_upd,
        description="Adds a Subdivision Surface modifier: low-poly mesh to edit, smooth result")
    cross_loops: IntProperty(
        name="Cross Loops", default=1, min=1, max=200, update=_upd,
        description="Minimum number of segments across the sheet, so there are vertices to grab along the whole length of a roll or fold")
    res_bend: IntProperty(name="Resolution", default=120, min=4, max=1000, update=_upd)
    res_cross: IntProperty(name="Cross Resolution", default=24, min=1, max=400, update=_upd)
    bend_axis: EnumProperty(name="Bend Across",
                            items=[("LENGTH", "Length", "Folds/waves run along the long side"),
                                   ("WIDTH", "Width", "Folds/waves run along the short side")],
                            default="LENGTH", update=_upd)
    anchor: EnumProperty(name="Anchor",
                         items=[("START", "Start", ""), ("CENTER", "Center", ""), ("END", "End", ""),
                                ("GROUND", "Ground", "Start sits at the origin, nothing is re-levelled "
                                 "(use for standing receipts and arches)")],
                         default="CENTER", update=_upd)

    # wave
    wave_amount: FloatProperty(name="Wave", default=0.0, soft_min=-90, soft_max=90,
                               description="Peak tilt of the wave in degrees", update=_upd)
    wave_count: FloatProperty(name="Waves", default=1.0, min=0.0, soft_max=6.0, update=_upd)
    wave_phase: FloatProperty(name="Phase", default=0.0, update=_upd)

    # twist / cross bend
    twist: FloatProperty(name="Twist", default=0.0, soft_min=-360, soft_max=360, update=_upd)
    cross_bend: FloatProperty(name="Cup", default=0.0, soft_min=-180, soft_max=180,
                              description="Bend across the other axis (like a held bill)", update=_upd)

    # folds
    fold_count: IntProperty(name="Folds", default=0, min=0, max=12, update=_upd)
    fold_angle: FloatProperty(name="Fold Angle", default=150.0, min=0.0, max=180.0, update=_upd)
    fold_style: EnumProperty(name="Style",
                             items=[("ACCORDION", "Accordion (Z)", "Alternate mountain/valley"),
                                    ("SAME", "Same Way", "Every fold turns the same direction")],
                             default="ACCORDION", update=_upd)
    fold_flip: BoolProperty(name="Flip", default=False, update=_upd)
    fold_offset: FloatProperty(name="Offset", default=0.0, min=-0.9, max=0.9, update=_upd)
    crease_softness: FloatProperty(name="Softness", default=0.08, min=0.005, max=1.0, update=_upd)

    # roll
    roll_amount: FloatProperty(name="Roll", default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=_upd)
    roll_radius: FloatProperty(name="Radius", default=0.18, min=0.005, soft_max=2.0, unit="LENGTH", update=_upd)
    roll_tightness: FloatProperty(name="Layer Gap", subtype="FACTOR", default=0.18, min=0.0, max=1.0, update=_upd,
                                  description="How much the roll tightens each turn")
    roll_from: EnumProperty(name="From",
                            items=[("END", "End", ""), ("START", "Start", ""), ("BOTH", "Both (Scroll)", "")],
                            default="END", update=_upd)
    roll_flip: BoolProperty(name="Roll Down", default=False, update=_upd)
    roll_balance: FloatProperty(name="Balance", default=0.0, min=-1.0, max=1.0, update=_upd,
                                description="Scroll: roll more at one end than the other")
    roll_skew: FloatProperty(name="Unevenness", default=0.0, min=-0.9, max=0.9, update=_upd,
                             description="Make rolls fatter on one side than the other, like a hand-rolled scroll")

    roll_flare: FloatProperty(name="Flare", default=0.0, min=-0.9, max=1.5, update=_upd,
                              description="Rolls swell at their ends (+) or bulge in the middle (-)")

    # torn edges
    tear_depth: FloatProperty(name="Torn Edges", default=0.0, min=0.0, max=0.3, update=_upd,
                              description="How deep the ragged side edges bite in")
    tear_detail: FloatProperty(name="Tear Detail", default=8.0, min=0.5, soft_max=40.0, update=_upd)

    # edges
    edge_curl: FloatProperty(name="Edge Curl", default=0.0, soft_min=-180, soft_max=180, update=_upd)
    edge_curl_size: FloatProperty(name="Curl Size", default=0.2, min=0.01, max=0.5, update=_upd)

    # independent end curls
    curl_start: FloatProperty(name="Start Curl", default=0.0, soft_min=-360, soft_max=360, update=_upd)
    curl_start_size: FloatProperty(name="Size", default=0.1, min=0.01, max=0.9, update=_upd)
    curl_end: FloatProperty(name="End Curl", default=0.0, soft_min=-360, soft_max=360, update=_upd)
    curl_end_size: FloatProperty(name="Size", default=0.15, min=0.01, max=0.9, update=_upd)

    # lift / stand up
    lift_angle: FloatProperty(name="Lift", default=0.0, soft_min=-180, soft_max=180,
                              description="Turn the sheet up (90 = stands upright)", update=_upd)
    lift_at: FloatProperty(name="At", default=0.3, min=0.0, max=1.0, subtype="FACTOR", update=_upd)
    lift_softness: FloatProperty(name="Bend Size", default=0.08, min=0.005, max=1.0, update=_upd)

    # arches
    arch_count: IntProperty(name="Arches", default=0, min=0, max=40, update=_upd)
    arch_height: FloatProperty(name="Height", default=0.1, min=0.0, max=1.0, update=_upd,
                               description="Height of the first arch, as a share of sheet length")
    arch_width: FloatProperty(name="Width", default=0.03, min=0.002, max=0.5, update=_upd,
                              description="How wide each rounded top is")
    arch_decay: FloatProperty(name="Shrink", default=0.85, min=0.1, max=2.0, update=_upd,
                              description="Each next arch is this times the previous (>1 grows)")
    arch_start: FloatProperty(name="Flat Lead-in", default=0.3, min=0.0, max=0.95, update=_upd)
    arch_lean: FloatProperty(name="Lean", default=0.0, soft_min=-45, soft_max=45, update=_upd)

    # crumple
    crumple: FloatProperty(name="Crumple", default=0.0, min=0.0, soft_max=2.0, update=_upd)
    crumple_scale: FloatProperty(name="Detail", default=3.0, min=0.1, soft_max=20.0, update=_upd)
    crumple_sharp: FloatProperty(name="Sharp Origas", default=0.5, min=0.0, max=1.0, update=_upd)
    seed: IntProperty(name="Seed", default=0, update=_upd)

    # animation
    flutter: BoolProperty(name="Animate Flutter", default=False, update=_upd,
                          description="Wave travels over time (flying money)")
    flutter_speed: FloatProperty(name="Speed", default=1.5, soft_min=-10, soft_max=10, update=_upd)
    twist_flutter: FloatProperty(name="Twist Wobble", default=15.0, soft_max=90, update=_upd)


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
_RESET = dict(wave_amount=0.0, wave_count=1.0, wave_phase=0.0, twist=0.0, cross_bend=0.0,
              fold_count=0, fold_angle=150.0, fold_style="ACCORDION", fold_flip=False,
              fold_offset=0.0, crease_softness=0.08, roll_amount=0.0, roll_radius=0.18,
              roll_tightness=0.18, roll_from="END", roll_flip=False, roll_balance=0.0,
              roll_skew=0.0, roll_flare=0.0, tear_depth=0.0, tear_detail=8.0, edge_curl=0.0,
              edge_curl_size=0.2, crumple=0.0, bend_axis="LENGTH", anchor="CENTER",
              curl_start=0.0, curl_start_size=0.1, curl_end=0.0, curl_end_size=0.15,
              lift_angle=0.0, lift_at=0.3, lift_softness=0.08, arch_count=0,
              arch_height=0.1, arch_width=0.03, arch_decay=0.85, arch_start=0.3, arch_lean=0.0)

PRESETS = {
    "FLAT":      ("Flat", {}),
    "FLYING":    ("Flying Bill", dict(wave_amount=38, wave_count=1.3, wave_phase=30, twist=25, cross_bend=12)),
    "RIBBON":    ("Wavy Ribbon", dict(wave_amount=55, wave_count=2.2, twist=60)),
    "HALF":      ("Folded in Half", dict(fold_count=1, fold_angle=172, crease_softness=0.06, anchor="START")),
    "HALF_OPEN": ("Half Open", dict(fold_count=1, fold_angle=95, crease_softness=0.1, cross_bend=8)),
    "TRIFOLD":   ("Tri-Fold (Z)", dict(fold_count=2, fold_angle=115, crease_softness=0.1, cross_bend=6)),
    "ACCORDION": ("Accordion", dict(fold_count=7, fold_angle=110, crease_softness=0.15)),
    "LENGTHWISE": ("Folded Lengthwise", dict(bend_axis="WIDTH", fold_count=1, fold_angle=120, crease_softness=0.1)),
    "ROLLED":    ("Rolled Cash", dict(roll_amount=0.82, roll_radius=0.2, roll_tightness=0.2,
                                       anchor="START")),
    "ROLL_TIGHT": ("Rolled Tight", dict(roll_amount=1.0, roll_radius=0.16, roll_tightness=0.16,
                                        anchor="START")),
    "SCROLL":    ("Scroll", dict(roll_amount=0.35, roll_from="BOTH", roll_radius=0.12, roll_tightness=0.2)),
    "PARCHMENT": ("Parchment Scroll", dict(roll_amount=0.34, roll_from="BOTH", roll_radius=0.06,
                                           roll_tightness=0.38, roll_skew=0.2, roll_flare=0.6,
                                           roll_balance=0.08,
                                           tear_depth=0.05, tear_detail=6.0, crumple=0.12,
                                           crumple_scale=4.0, res_bend=200, res_cross=40)),
    "CURLED":    ("Curled Edges", dict(edge_curl=55, edge_curl_size=0.22, cross_bend=10)),
    "CRUMPLED":  ("Crumpled", dict(crumple=0.9, crumple_scale=3.5, crumple_sharp=0.7, wave_amount=10)),
    "RECEIPT_STAND": ("Receipt: Standing", dict(
        size_preset="RECEIPT_LONG", anchor="GROUND", res_bend=360, lift_angle=90, lift_at=0.03,
        lift_softness=0.05, curl_end=240, curl_end_size=0.2, cross_bend=10)),
    "RECEIPT_S":     ("Receipt: S-Curve", dict(
        size_preset="RECEIPT_LONG", anchor="GROUND", res_bend=360, lift_angle=90, lift_at=0.32,
        lift_softness=0.3, curl_start=110, curl_start_size=0.16, curl_end=170, curl_end_size=0.25,
        cross_bend=8)),
    "RECEIPT_ARCH":  ("Receipt: Arches", dict(
        size_preset="RECEIPT_LONG", anchor="GROUND", res_bend=500, arch_count=4, arch_height=0.13,
        arch_width=0.045, arch_decay=0.82, arch_start=0.22, arch_lean=8, curl_start=20,
        curl_start_size=0.06)),
    "WORN":      ("Used / Worn", dict(fold_count=1, fold_angle=18, crease_softness=0.02,
                                      crumple=0.25, crumple_scale=5.0, crumple_sharp=0.8, edge_curl=12)),
}


def apply_preset(obj, key, frame=None):
    global _SUSPEND
    p = obj.origa
    _SUSPEND = True
    try:
        shape = {**_RESET, **PRESETS[key][1]}
        # A shape never changes the sheet's size; resolution only ever goes up.
        shape.pop("size_preset", None)
        if shape.get("res_bend", 0) <= p.res_bend:
            shape.pop("res_bend", None)
        for k, v in shape.items():
            setattr(p, k, v)
    finally:
        _SUSPEND = False
    build_sheet(obj, frame)


# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------
def _set_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _paper_material(image=None, name="Origa Paper"):
    mat = bpy.data.materials.get(name) if image is None else None
    if mat is None:
        mat = bpy.data.materials.new(name if image is None else f"Origa {image.name}")
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.65
        if image is None:
            bsdf.inputs["Base Color"].default_value = (0.93, 0.93, 0.9, 1)
        else:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = image
            tex.location = (bsdf.location.x - 320, bsdf.location.y)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def _preset_items(self, context):
    return [(k, v[0], "") for k, v in PRESETS.items()]


def new_sheet(context, preset="FLAT", size="AUTO", name="Paper"):
    if size == "AUTO":
        size = PRESETS[preset][1].get("size_preset", "USD")
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj.location = context.scene.cursor.location
    p = obj.origa
    global _SUSPEND
    _SUSPEND = True
    p.is_paper = True
    p.size_preset = size
    _apply_size(p)
    _SUSPEND = False
    me.materials.append(_paper_material())
    apply_preset(obj, preset, context.scene.frame_current)
    return obj


class ORIGA_OT_add(bpy.types.Operator):
    bl_idname = "origa.add_sheet"
    bl_label = "Paper Sheet"
    bl_description = "Add a live paper / banknote sheet"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(name="Shape", items=_preset_items)
    size: EnumProperty(name="Size",
                       items=[("AUTO", "Shape's Default", "Bills for money shapes, long receipt for receipt shapes")]
                       + [(k, v[0], "") for k, v in SIZES.items()],
                       default="AUTO")

    def execute(self, context):
        for o in context.selected_objects:
            o.select_set(False)
        obj = new_sheet(context, self.preset, self.size)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


class ORIGA_OT_preset(bpy.types.Operator):
    bl_idname = "origa.apply_preset"
    bl_label = "Apply Shape"
    bl_description = "Change the shape of the selected sheets. Their size and texture stay as they are"
    bl_options = {"REGISTER", "UNDO"}
    preset: EnumProperty(name="Shape", items=_preset_items)

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.origa.is_paper:
                apply_preset(obj, self.preset, context.scene.frame_current)
        return {"FINISHED"}


class ORIGA_OT_make_paper(bpy.types.Operator):
    bl_idname = "origa.make_paper"
    bl_label = "Convert Plane to Paper"
    bl_description = "Turn the selected plane into a live Origa sheet using its size"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh plane")
            return {"CANCELLED"}
        dims = obj.dimensions
        a, b = sorted([dims.x, dims.y], reverse=True)
        if a < 1e-6:
            self.report({"ERROR"}, "Plane has no size")
            return {"CANCELLED"}
        global _SUSPEND
        _SUSPEND = True
        p = obj.origa
        p.is_paper = True
        p.size_preset = "CUSTOM"
        p.length, p.width = a, max(b, 1e-3)
        _SUSPEND = False
        obj.scale = (1, 1, 1)
        if not obj.data.materials:
            obj.data.materials.append(_paper_material())
        build_sheet(obj, context.scene.frame_current)
        return {"FINISHED"}


class ORIGA_OT_texture(bpy.types.Operator):
    bl_idname = "origa.load_texture"
    bl_label = "Load Texture"
    bl_description = "Put an image (front of a banknote, poster, etc.) on the selected sheets"
    bl_options = {"REGISTER", "UNDO"}
    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        img = bpy.data.images.load(self.filepath, check_existing=True)
        mat = _paper_material(img)
        for obj in context.selected_objects:
            if obj.type == "MESH":
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
        return {"FINISHED"}


class ORIGA_OT_rain(bpy.types.Operator):
    bl_idname = "origa.money_rain"
    bl_label = "Money Rain"
    bl_description = "Scatter many flying bills, each with its own shape"
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(name="Count", default=20, min=1, max=500)
    spread: FloatProperty(name="Spread", default=8.0, min=0.1)
    height: FloatProperty(name="Height", default=6.0, min=0.0)
    variety: FloatProperty(name="Variety", default=1.0, min=0.0, max=2.0)
    animate: BoolProperty(name="Animate Flutter", default=True)
    fall: BoolProperty(name="Animate Falling", default=False)
    seed: IntProperty(name="Seed", default=1)
    def execute(self, context):
        rng = random.Random(self.seed)
        if self.animate or self.fall:
            _lock_for_render(context.scene)
        src = context.active_object if context.active_object and context.active_object.origa.is_paper else None
        size = src.origa.size_preset if src and src.origa.size_preset != "CUSTOM" else "USD"
        mat = src.active_material if src else None
        col = bpy.data.collections.new("Money Rain")
        context.scene.collection.children.link(col)
        base = context.scene.cursor.location.copy()
        scn = context.scene
        v = self.variety
        for i in range(self.count):
            me = bpy.data.meshes.new(f"Bill.{i:03d}")
            obj = bpy.data.objects.new(f"Bill.{i:03d}", me)
            col.objects.link(obj)
            p = obj.origa
            global _SUSPEND
            _SUSPEND = True
            p.is_paper = True
            p.size_preset = size
            if src:
                p.length = src.origa.length
            _apply_size(p)
            for k, val in _RESET.items():
                setattr(p, k, val)
            p.wave_amount = rng.uniform(15, 50) * v * rng.choice((-1, 1))
            p.wave_count = rng.uniform(0.6, 1.8)
            p.wave_phase = rng.uniform(0, 360)
            p.twist = rng.uniform(-50, 50) * v
            p.cross_bend = rng.uniform(-20, 20) * v
            if rng.random() < 0.25 * v:
                p.fold_count = 1
                p.fold_angle = rng.uniform(20, 90)
            p.res_bend = 72
            p.res_cross = 12
            p.flutter = self.animate
            p.flutter_speed = rng.uniform(0.8, 2.5) * rng.choice((-1, 1))
            p.twist_flutter = rng.uniform(5, 25)
            _SUSPEND = False
            me.materials.append(mat or _paper_material())
            obj.location = base + Vector((rng.uniform(-1, 1) * self.spread / 2,
                                          rng.uniform(-1, 1) * self.spread / 2,
                                          rng.uniform(0, self.height)))
            obj.rotation_euler = (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                                  rng.uniform(0, math.tau))
            build_sheet(obj, scn.frame_current)
            if self.fall:
                f0, f1 = scn.frame_start, scn.frame_end
                obj.keyframe_insert("location", frame=f0)
                obj.keyframe_insert("rotation_euler", frame=f0)
                obj.location.z -= self.height + 2.0
                obj.location.x += rng.uniform(-1, 1)
                obj.rotation_euler.z += rng.uniform(-3, 3)
                obj.keyframe_insert("location", frame=f1)
                obj.keyframe_insert("rotation_euler", frame=f1)
        return {"FINISHED"}


class ORIGA_OT_freeze(bpy.types.Operator):
    bl_idname = "origa.freeze"
    bl_label = "Freeze to Mesh"
    bl_description = "Stop live updates so you can edit the mesh by hand"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            obj.origa.is_paper = False
            obj.origa.flutter = False
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
class ORIGA_PT_panel(bpy.types.Panel):
    bl_label = "Origa"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Origa"

    def draw(self, context):
        lay = self.layout
        obj = context.active_object
        row = lay.row(align=True)
        row.operator_menu_enum("origa.add_sheet", "preset", text="Add Sheet", icon="ADD")
        lay.operator("origa.money_rain", icon="PARTICLES")
        if obj is None or obj.type != "MESH":
            return
        p = obj.origa
        if not p.is_paper:
            lay.operator("origa.make_paper", icon="MESH_PLANE")
            return

        lay.operator_menu_enum("origa.apply_preset", "preset", text="Shape Preset", icon="PRESET")

        b = lay.box()
        b.label(text="Sheet", icon="FILE_BLANK")
        b.prop(p, "size_preset")
        r = b.row(align=True)
        r.prop(p, "length")
        r.prop(p, "width")
        b.prop(p, "keep_shape")
        b.prop(p, "thickness")
        r = b.row(align=True)
        r.prop(p, "bend_axis", expand=True)
        b.prop(p, "anchor")
        r = b.row(align=True)
        r.prop(p, "detail", text="")
        r.prop(p, "smooth_levels")
        if p.detail == "UNIFORM":
            r = b.row(align=True)
            r.prop(p, "res_bend")
            r.prop(p, "res_cross", text="Cross")
        else:
            b.prop(p, "cross_loops")
        b.label(text=f"{len(obj.data.vertices):,} vertices · {len(obj.data.polygons):,} faces",
                icon="MESH_DATA")
        b.operator("origa.load_texture", icon="IMAGE_DATA")
        r = b.row(align=True)
        r.prop(p, "uv_flip", toggle=True)
        r.prop(p, "uv_mirror", toggle=True)

        b = lay.box()
        b.label(text="Wave & Twist", icon="MOD_WAVE")
        b.prop(p, "wave_amount")
        r = b.row(align=True)
        r.prop(p, "wave_count")
        r.prop(p, "wave_phase")
        b.prop(p, "twist")
        b.prop(p, "cross_bend")

        b = lay.box()
        b.label(text="Folds", icon="MOD_SIMPLEDEFORM")
        b.prop(p, "fold_count")
        if p.fold_count:
            b.prop(p, "fold_angle")
            b.prop(p, "fold_style", text="")
            r = b.row(align=True)
            r.prop(p, "crease_softness")
            r.prop(p, "fold_offset")
            b.prop(p, "fold_flip")

        b = lay.box()
        b.label(text="Roll & Curl", icon="MOD_SCREW")
        b.prop(p, "roll_amount")
        if p.roll_amount > 0:
            r = b.row(align=True)
            r.prop(p, "roll_radius")
            r.prop(p, "roll_tightness")
            b.prop(p, "roll_from")
            r = b.row(align=True)
            r.prop(p, "roll_skew")
            r.prop(p, "roll_flare")
            r = b.row(align=True)
            sub = r.row(align=True)
            sub.enabled = p.roll_from == "BOTH"
            sub.prop(p, "roll_balance")
            b.prop(p, "roll_flip")
        b.prop(p, "edge_curl")
        if p.edge_curl:
            b.prop(p, "edge_curl_size")

        r = b.row(align=True)
        r.prop(p, "curl_start")
        r.prop(p, "curl_start_size")
        r = b.row(align=True)
        r.prop(p, "curl_end")
        r.prop(p, "curl_end_size")

        b = lay.box()
        b.label(text="Stand & Arches", icon="IPO_EASE_IN_OUT")
        b.prop(p, "lift_angle")
        if p.lift_angle:
            r = b.row(align=True)
            r.prop(p, "lift_at")
            r.prop(p, "lift_softness")
        b.prop(p, "arch_count")
        if p.arch_count:
            r = b.row(align=True)
            r.prop(p, "arch_height")
            r.prop(p, "arch_width")
            r = b.row(align=True)
            r.prop(p, "arch_decay")
            r.prop(p, "arch_lean")
            b.prop(p, "arch_start")
        if (p.lift_angle or p.arch_count) and p.anchor != "GROUND":
            b.label(text="Tip: set Anchor to Ground", icon="INFO")

        b = lay.box()
        b.label(text="Torn Edges", icon="MOD_EDGESPLIT")
        b.prop(p, "tear_depth")
        if p.tear_depth:
            r = b.row(align=True)
            r.prop(p, "tear_detail")
            r.prop(p, "seed")

        b = lay.box()
        b.label(text="Crumple", icon="FORCE_TURBULENCE")
        b.prop(p, "crumple")
        if p.crumple:
            r = b.row(align=True)
            r.prop(p, "crumple_scale")
            r.prop(p, "crumple_sharp", text="Sharp")
            b.prop(p, "seed")

        b = lay.box()
        b.label(text="Animation", icon="PLAY")
        b.prop(p, "flutter")
        if p.flutter:
            r = b.row(align=True)
            r.prop(p, "flutter_speed")
            r.prop(p, "twist_flutter")
        b.label(text="Tip: every slider can be keyframed (I key).", icon="INFO")
        if p.flutter or _has_origa_keys(obj):
            b.prop(context.scene.render, "use_lock_interface", text="Lock UI while rendering (needed)")

        lay.operator("origa.freeze", icon="FREEZE")


def _convert_menu(self, context):
    self.layout.separator()
    self.layout.operator("origa.make_paper", text="Paper (Origa)", icon="FILE_BLANK")


def _add_menu(self, context):
    self.layout.operator_menu_enum("origa.add_sheet", "preset", text="Paper Sheet", icon="FILE_BLANK")


# --------------------------------------------------------------------------
# Live updates during playback (flutter + keyframed sliders)
# --------------------------------------------------------------------------
def _fcurves(action):
    """F-curves of an action on both legacy and slotted (4.4+/5.x) actions."""
    fcs = getattr(action, "fcurves", None)
    if fcs is not None:
        try:
            return list(fcs)
        except Exception:
            pass
    out = []
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                out.extend(bag.fcurves)
    return out


def _has_origa_keys(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    return any(fc.data_path.startswith("origa.") for fc in _fcurves(ad.action))


@persistent
def _frame_handler(scene, depsgraph=None):
    f = scene.frame_current
    for obj in scene.objects:
        p = obj.origa
        if p.is_paper and (p.flutter or _has_origa_keys(obj)):
            build_sheet(obj, f)


def _lock_for_render(scene):
    # Rebuilding meshes on frame change is only render-safe with a locked UI.
    if scene and not scene.render.use_lock_interface:
        scene.render.use_lock_interface = True


classes = (OrigaProps, ORIGA_OT_add, ORIGA_OT_preset, ORIGA_OT_make_paper,
           ORIGA_OT_texture, ORIGA_OT_rain, ORIGA_OT_freeze, ORIGA_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.origa = PointerProperty(type=OrigaProps)
    bpy.types.VIEW3D_MT_mesh_add.append(_add_menu)
    bpy.types.VIEW3D_MT_object_convert.append(_convert_menu)
    if _frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_handler)


def unregister():
    if _frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_handler)
    bpy.types.VIEW3D_MT_object_convert.remove(_convert_menu)
    bpy.types.VIEW3D_MT_mesh_add.remove(_add_menu)
    del bpy.types.Object.origa
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
