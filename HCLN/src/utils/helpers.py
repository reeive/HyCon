import torch
import random
import numpy as np
import logging
from dhg import Hypergraph 
import torch.nn.functional as F 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def derive_node_label_from_seg_mask(mask_tensor, num_classes_teacher):
    if num_classes_teacher == 2:
        return 1 if torch.any(mask_tensor > 0) else 0
    elif num_classes_teacher > 2:

        unique_vals = torch.unique(mask_tensor)
        foreground_classes = unique_vals[unique_vals > 0]
        if len(foreground_classes) > 0:

            # counts = torch.bincount(mask_tensor[mask_tensor > 0].flatten())
            # if len(counts) > 0: return torch.argmax(counts).item()
            return foreground_classes[0].item() 
        return 0 # 仅背景
    return 0 # 默认情况

def fix_iso_v(hg: Hypergraph) -> Hypergraph:
    num_v = hg.num_v
    current_e_list = list(hg.e[0]) 
   
    degrees = hg.deg_v 
    
    isolated_vertices_found = False
    for v_idx in range(num_v):
        if degrees[v_idx] == 0: 
            current_e_list.append(tuple([v_idx]))
            isolated_vertices_found = True
            logging.debug(f"fix_iso_v: Added self-loop for isolated vertex {v_idx}")

    if isolated_vertices_found:
        return Hypergraph(num_v, current_e_list)


def ho_topology_score(emb: torch.Tensor, hg: Hypergraph) -> float:
    if hg.num_e == 0:
        logging.warning("ho_topology_score: Hypergraph has no edges. Returning 0.0.")
        return 0.0

    total_hyperedge_similarity_score = 0.0
    processed_hyperedges_count = 0


    for hyperedge_nodes in hg.e[0]:
        if len(hyperedge_nodes) < 2:
            continue


        try:
            node_indices = torch.tensor(list(hyperedge_nodes), dtype=torch.long, device=emb.device)
            if torch.any(node_indices >= emb.shape[0]) or torch.any(node_indices < 0):
                logging.warning(f"ho_topology_score: Invalid node index in hyperedge {hyperedge_nodes}. Max index: {emb.shape[0]-1}. Skipping.")
                continue
            
            current_hyperedge_embeddings = emb[node_indices]
        except IndexError as e:
            logging.warning(f"ho_topology_score: Error accessing embeddings for hyperedge {hyperedge_nodes}: {e}. Skipping.")
            continue

        num_nodes_in_hyperedge = current_hyperedge_embeddings.shape[0]
        pairwise_similarity_sum = 0.0
        num_pairs = 0

        for i in range(num_nodes_in_hyperedge):
            for j in range(i + 1, num_nodes_in_hyperedge):
                sim = F.cosine_similarity(current_hyperedge_embeddings[i].unsqueeze(0), 
                                          current_hyperedge_embeddings[j].unsqueeze(0), 
                                          dim=1)
                pairwise_similarity_sum += sim.item()
                num_pairs += 1
        
        if num_pairs > 0:
            average_similarity_for_hyperedge = pairwise_similarity_sum / num_pairs
            total_hyperedge_similarity_score += average_similarity_for_hyperedge
            processed_hyperedges_count += 1

    if processed_hyperedges_count == 0:
        logging.warning("ho_topology_score: No hyperedges with 2 or more nodes found to calculate similarity. Returning 0.0.")
        return 0.0

    final_score = total_hyperedge_similarity_score / processed_hyperedges_count
    logging.debug(f"ho_topology_score: Calculated score {final_score:.4f} over {processed_hyperedges_count} hyperedges.")
    return final_score