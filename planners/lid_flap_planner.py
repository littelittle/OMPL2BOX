import math


def _make_uniform_grid(lo, hi, resolution):
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    if math.isclose(lo, hi):
        return [float(lo)] * resolution
    step = (hi - lo) / float(resolution - 1)
    return [float(lo + k * step) for k in range(resolution)]


def _seed_signature(q, ndigits=3):
    if q is None:
        return None
    return tuple(round(float(x), ndigits) for x in q)

def search_traj(left_angle_tuple: tuple, right_angle_tuple: tuple, is_feasible=None, num_sample:int=10, verbose=False):
    """
    find the feasible lid/flap angle trajectory 
    *_angle_tuple: (lid_angle, flap_angle)
    """
    middle_flap_angle = (left_angle_tuple[1]+right_angle_tuple[1])/2 
    step = (right_angle_tuple[1]-left_angle_tuple[1])/(num_sample-1)
    samples = [left_angle_tuple[1]+i*step for i in range(num_sample)]
    mid = (num_sample-1)/2.0
    ordered = sorted(range(num_sample), key=lambda i:abs(i-mid))
    flap_angle_candidate = [middle_flap_angle] + [samples[i] for i in ordered]
    # flap_angle_candidate = samples
    middle_angle_tuple_list = [((left_angle_tuple[0]+right_angle_tuple[0])/2, i) for i in flap_angle_candidate]
    for i, candidate_tuple in enumerate(middle_angle_tuple_list):
        if verbose:
            print(i)
        q_candidate = is_feasible(candidate_tuple, q_reset=is_feasible(right_angle_tuple)) # q_reset=is_feasible(right_angle_tuple)
        if q_candidate is not None:
            if abs(right_angle_tuple[0]-left_angle_tuple[0]) > 20:
                left_traj, left_q = search_traj(left_angle_tuple, candidate_tuple, is_feasible, num_sample, verbose=False)
                if left_traj is None:
                    continue
                right_traj, right_q = search_traj(candidate_tuple, right_angle_tuple, is_feasible, num_sample, verbose=False)
                if right_traj is None:
                    continue
                return left_traj + right_traj, left_q + right_q
            else:
                return [candidate_tuple, right_angle_tuple], [q_candidate, is_feasible(right_angle_tuple)]
    return None, None

def search_traj_cache(
    left_angle_tuple: tuple,
    right_angle_tuple: tuple,
    is_feasible=None,
    num_sample: int = 10,
    verbose: bool = False,
    *,
    resolution=None,
    cache=None,
    max_seed_trials_per_cell: int = 1,
):
    """
    Cached / quantized version of search_traj.

    Return convention is kept compatible with the old version:
      - returned traj does NOT include left_angle_tuple
      - returned traj DOES include right_angle_tuple (snapped onto the grid)
      - q_list aligns with traj

    Parameters
    ----------
    resolution:
        number of grid points per axis.
        If None, fall back to num_sample, so the old call style still works.
    cache:
        a mutable dict. Reuse the same dict across repeated searches in the
        same environment to get high cache hit rate.
    max_seed_trials_per_cell:
        1  -> strict at-most resolution^2 calls to is_feasible
        >1 -> allow a few different q_reset seeds per grid cell
    """
    if is_feasible is None:
        raise ValueError("is_feasible must not be None.")
    if resolution is None:
        resolution = num_sample
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("resolution must be >= 2.")
    if max_seed_trials_per_cell < 1:
        raise ValueError("max_seed_trials_per_cell must be >= 1.")

    lid_min, lid_max = sorted((float(left_angle_tuple[0]), float(right_angle_tuple[0])))
    flap_min, flap_max = sorted((float(left_angle_tuple[1]), float(right_angle_tuple[1])))

    if cache is None:
        cache = {}

    if not cache:
        cache.update(
            {
                "resolution": resolution,
                "lid_min": lid_min,
                "lid_max": lid_max,
                "flap_min": flap_min,
                "flap_max": flap_max,
                "lid_grid": _make_uniform_grid(lid_min, lid_max, resolution),
                "flap_grid": _make_uniform_grid(flap_min, flap_max, resolution),
                "point_cache": {},   # (lid_idx, flap_idx) -> {"q": ..., "tried_seed_sigs": set(...)}
                "segment_cache": {}, # ((left_idx),(right_idx)) -> {"version": int, "traj_idx": tuple(...) or None}
                "version": 0,
                "stats": {
                    "is_feasible_calls": 0,
                    "point_cache_hits": 0,
                    "segment_cache_hits": 0,
                },
            }
        )
    else:
        expected = (resolution, lid_min, lid_max, flap_min, flap_max)
        actual = (
            cache.get("resolution"),
            cache.get("lid_min"),
            cache.get("lid_max"),
            cache.get("flap_min"),
            cache.get("flap_max"),
        )
        if actual != expected:
            raise ValueError(
                "cache was created for a different angle range or resolution. "
                "Clear it or use a fresh dict."
            )
        cache.setdefault("lid_grid", _make_uniform_grid(lid_min, lid_max, resolution))
        cache.setdefault("flap_grid", _make_uniform_grid(flap_min, flap_max, resolution))
        cache.setdefault("point_cache", {})
        cache.setdefault("segment_cache", {})
        cache.setdefault("version", 0)
        cache.setdefault(
            "stats",
            {
                "is_feasible_calls": 0,
                "point_cache_hits": 0,
                "segment_cache_hits": 0,
            },
        )

    def angle_to_idx(angle_tuple):
        lid_angle, flap_angle = float(angle_tuple[0]), float(angle_tuple[1])

        if math.isclose(lid_max, lid_min):
            lid_idx = 0
        else:
            lid_idx = int(round((lid_angle - lid_min) / (lid_max - lid_min) * (resolution - 1)))

        if math.isclose(flap_max, flap_min):
            flap_idx = 0
        else:
            flap_idx = int(round((flap_angle - flap_min) / (flap_max - flap_min) * (resolution - 1)))

        lid_idx = max(0, min(resolution - 1, lid_idx))
        flap_idx = max(0, min(resolution - 1, flap_idx))
        return (lid_idx, flap_idx)

    def idx_to_angle(idx):
        lid_idx, flap_idx = idx
        return (
            float(cache["lid_grid"][lid_idx]),
            float(cache["flap_grid"][flap_idx]),
        )

    def center_out_indices(a, b):
        lo, hi = sorted((a, b))
        mid = 0.5 * (lo + hi)
        return sorted(range(lo, hi + 1), key=lambda k: abs(k - mid))

    def middle_lid_indices(i0, i1):
        lo, hi = sorted((i0, i1))
        if hi - lo <= 1:
            return []
        mid = 0.5 * (i0 + i1)
        candidates = []
        for i in (int(math.floor(mid)), int(math.ceil(mid))):
            if lo < i < hi and i not in candidates:
                candidates.append(i)
        candidates.sort(key=lambda i: abs(i - mid))
        return candidates

    def evaluate_point(idx, preferred_q=None):
        entry = cache["point_cache"].setdefault(
            idx,
            {
                "q": None,
                "tried_seed_sigs": set(),
            },
        )

        # Already know one representative IK solution for this cell.
        if entry["q"] is not None:
            cache["stats"]["point_cache_hits"] += 1
            return entry["q"]

        seed_sig = _seed_signature(preferred_q)
        if seed_sig in entry["tried_seed_sigs"]:
            return None
        if len(entry["tried_seed_sigs"]) >= max_seed_trials_per_cell:
            return None

        entry["tried_seed_sigs"].add(seed_sig)
        cache["stats"]["is_feasible_calls"] += 1

        angle_tuple = idx_to_angle(idx)
        q = is_feasible(angle_tuple, q_reset=preferred_q)

        if q is not None:
            entry["q"] = q
            cache["version"] += 1
        # import ipdb; ipdb.set_trace()
        return entry["q"]

    def store_segment(seg_key, traj_idx):
        cache["segment_cache"][seg_key] = {
            "version": cache["version"],
            "traj_idx": None if traj_idx is None else tuple(traj_idx),
        }

    def search_rec(left_idx, right_idx):
        seg_key = (left_idx, right_idx)
        seg_entry = cache["segment_cache"].get(seg_key)

        # Conservative invalidation:
        # whenever a new q is discovered anywhere, version changes and stale
        # segment results are recomputed.
        if seg_entry is not None and seg_entry["version"] == cache["version"]:
            cache["stats"]["segment_cache_hits"] += 1
            traj_idx = seg_entry["traj_idx"]
            if traj_idx is None:
                return None
            return list(traj_idx)

        # Keep old behavior: the right endpoint itself should be feasible.
        q_right = evaluate_point(right_idx, preferred_q=None)
        if q_right is None:
            import ipdb; ipdb.set_trace()
            store_segment(seg_key, None)
            return None

        lid_gap = abs(right_idx[0] - left_idx[0])

        # No interior lid column remains on this discretized grid.
        if lid_gap <= 1:
            traj_idx = [right_idx]
            store_segment(seg_key, traj_idx)
            return traj_idx

        lid_candidates = middle_lid_indices(left_idx[0], right_idx[0])
        flap_candidates = center_out_indices(left_idx[1], right_idx[1])

        for mid_lid_idx in lid_candidates:
            for flap_idx in flap_candidates:
                candidate_idx = (mid_lid_idx, flap_idx)
                q_candidate = evaluate_point(candidate_idx, preferred_q=q_right)

                if verbose:
                    print(
                        f"try {idx_to_angle(candidate_idx)} -> "
                        f"{'OK' if q_candidate is not None else 'FAIL'}"
                    )

                if q_candidate is None:
                    continue

                left_traj = search_rec(left_idx, candidate_idx)
                if left_traj is None:
                    continue

                right_traj = search_rec(candidate_idx, right_idx)
                if right_traj is None:
                    continue

                traj_idx = left_traj + right_traj
                store_segment(seg_key, traj_idx)
                return traj_idx

        store_segment(seg_key, None)
        return None

    left_idx = angle_to_idx(left_angle_tuple)
    right_idx = angle_to_idx(right_angle_tuple)

    traj_idx = search_rec(left_idx, right_idx)
    if traj_idx is None:
        return None, None

    # Remove accidental consecutive duplicates.
    dedup_traj_idx = []
    for idx in traj_idx:
        if not dedup_traj_idx or idx != dedup_traj_idx[-1]:
            dedup_traj_idx.append(idx)

    traj = [idx_to_angle(idx) for idx in dedup_traj_idx]
    q_list = [cache["point_cache"][idx]["q"] for idx in dedup_traj_idx]
    return traj, q_list