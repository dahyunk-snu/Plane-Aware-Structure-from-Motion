import collections
from typing import Dict, List
import itertools
import networkx as nx


def merge_planes(
    feature_to_planes: Dict[int, Dict[int, List[str]]],
    min_overlap: int,
    min_global_inliers: int,
):
    """
    1) connecting local planes with feature overlap
    2) filtering ambiguous/noisy features
    3) selecting only robust global planes

    @return feature_to_global_plane[img_id][feat_idx] = global_plane_id
    """

    # 1. Compute overlap: local-plane pair → #features overlapping
    pair_overlap = collections.Counter()

    for img_id, feat_dict in feature_to_planes.items():
        for feat_idx, plane_list in feat_dict.items():
            if len(plane_list) < 2:
                continue

            uniq = sorted(set(plane_list))
            for p_i, p_j in itertools.combinations(uniq, 2):
                pair_overlap[(p_i, p_j)] += 1

    # 2. Graph construction
    plane_graph = nx.Graph()
    for (p_i, p_j), count in pair_overlap.items():
        if count >= min_overlap:
            plane_graph.add_edge(p_i, p_j, weight=count)

    if plane_graph.number_of_nodes() == 0:
        print("[INFO] No robust plane connections found.")
        return {}

    # 3. Connected-components to global plane id
    components = list(nx.connected_components(plane_graph))
    local_to_global = {}
    for gpid, comp in enumerate(components):
        for lp in comp:
            local_to_global[lp] = gpid

    # 4. Keep only features that map cleanly to ONE global plane
    feature_to_global = collections.defaultdict(dict)
    global_plane_support = collections.Counter()

    for img_id, feat_dict in feature_to_planes.items():
        for feat_idx, plane_list in feat_dict.items():
            gp_ids = {local_to_global[p] for p in plane_list if p in local_to_global}

            if len(gp_ids) == 1:
                gp = next(iter(gp_ids))
                feature_to_global[img_id][feat_idx] = gp
                global_plane_support[gp] += 1

    # 5. Filter weak global planes
    valid_gps = {
        gp for gp, cnt in global_plane_support.items() if cnt >= min_global_inliers
    }
    if not valid_gps:
        print("[INFO] No global plane passed min_global_inliers threshold.")
        return {}

    # Filter all outputs
    filtered_feature_to_global = {}
    for img_id, fmap in feature_to_global.items():
        new_map = {f: gp for f, gp in fmap.items() if gp in valid_gps}
        if new_map:
            filtered_feature_to_global[img_id] = new_map

    return filtered_feature_to_global
