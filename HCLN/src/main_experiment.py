import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from copy import deepcopy
import os

from dhg import Hypergraph, Graph
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.models import HGNNP 

from .datasets.modal_dataset import MultiModalDataset, DEFAULT_RESNET_TRANSFORM
from .models.hypergraph_models import MyHGNN, MyMLPs, MyGCN 
from .training.teacher_trainer import train_teacher, valid_teacher # test_teacher 
from .training.student_trainer import HighOrderConstraint, train_stu, test_stu # valid_stu
from .utils.helpers import set_seed, derive_node_label_from_seg_mask, fix_iso_v, ho_topology_score


def exp(cfg, feature_extractor):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)

    logging.info("Initializing datasets...")
    labeled_dataset = MultiModalDataset(
        data_dir=cfg.data.data_dir, mode='train_labeled',
        mode1_folder_suffix=cfg.data.mode1_suffix, mode2_folder_suffix=cfg.data.mode2_suffix,
        mask_folder_name=cfg.data.mask_folder, sample_list_file=cfg.data.labeled_list_file,
        image_load_rate=cfg.data.labeled_image_rate, transform=DEFAULT_RESNET_TRANSFORM, load_mask=True
    )
    all_nodes_dataset = MultiModalDataset(
        data_dir=cfg.data.data_dir, mode='all_nodes_for_hypergraph',
        mode1_folder_suffix=cfg.data.mode1_suffix, mode2_folder_suffix=cfg.data.mode2_suffix,
        mask_folder_name=cfg.data.mask_folder,
        sample_list_file=cfg.data.all_nodes_list_file,
        image_load_rate=1.0, transform=DEFAULT_RESNET_TRANSFORM, load_mask=True
    )
    val_dataset = MultiModalDataset(
        data_dir=cfg.data.data_dir, mode='validation',
        mode1_folder_suffix=cfg.data.mode1_suffix, mode2_folder_suffix=cfg.data.mode2_suffix,
        mask_folder_name=cfg.data.mask_folder, sample_list_file=cfg.data.val_list_file,
        image_load_rate=1.0, transform=DEFAULT_RESNET_TRANSFORM, load_mask=True
    )

    logging.info("Performing feature extraction for all hypergraph nodes...")
    global_sample_ids = all_nodes_dataset.sample_ids
    num_vertices = len(global_sample_ids)
    if num_vertices == 0:
        raise ValueError("No samples found in all_nodes_dataset.")
        
    sample_id_to_idx = {sid: i for i, sid in enumerate(global_sample_ids)}

    X_teacher_features = torch.zeros((num_vertices, feature_extractor.output_dim), dtype=torch.float32)
    X_student_features = torch.zeros((num_vertices, feature_extractor.output_dim), dtype=torch.float32)
    
    extraction_loader = DataLoader(all_nodes_dataset, batch_size=cfg.train.feature_extraction_batch_size,
                                   shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)
    processed_count = 0
    for batch_idx, batch in enumerate(extraction_loader):
        logging.debug(f"Feature extraction: batch {batch_idx + 1}/{len(extraction_loader)}")
        batch_sids = batch['id']
        batch_mode1_imgs = batch['mode1'].to(device)
        batch_mode2_imgs = batch['mode2'].to(device)
        batch_is_valid = batch['is_valid']

        with torch.no_grad():
            valid_indices_in_batch = [i for i, v_flag in enumerate(batch_is_valid) if v_flag]
            if not valid_indices_in_batch:
                continue

            sub_batch_mode1 = batch_mode1_imgs[valid_indices_in_batch]
            sub_batch_mode2 = batch_mode2_imgs[valid_indices_in_batch]
            sub_batch_sids = [batch_sids[i] for i in valid_indices_in_batch]

            if sub_batch_mode1.numel() > 0 :
                feat_m1 = feature_extractor(sub_batch_mode1)
                feat_m2 = feature_extractor(sub_batch_mode2)

                for i, sid in enumerate(sub_batch_sids):
                    node_idx = sample_id_to_idx.get(sid)
                    if node_idx is not None:
                        X_teacher_features[node_idx] = feat_m1[i].cpu()
                        X_student_features[node_idx] = feat_m2[i].cpu()
                        processed_count +=1
    
    logging.info(f"Successfully extracted features for {processed_count}/{num_vertices} nodes.")
    X_teacher = X_teacher_features.to(device)
    X_student = X_student_features.to(device)
    resnet_feature_dim = feature_extractor.output_dim

    logging.info("Preparing teacher labels and masks...")
    lbls_teacher_task = torch.full((num_vertices,), -1, dtype=torch.long, device=device)
    train_mask_teacher = torch.zeros(num_vertices, dtype=torch.bool, device=device)
    val_mask_teacher = torch.zeros(num_vertices, dtype=torch.bool, device=device)

    for sample_idx_in_labeled_set in range(len(labeled_dataset)):
        labeled_sample = labeled_dataset[sample_idx_in_labeled_set]
        if not labeled_sample['is_valid']: continue
        global_node_idx = sample_id_to_idx.get(labeled_sample['id'])
        if global_node_idx is not None:
            node_label = derive_node_label_from_seg_mask(labeled_sample['mask'], cfg.data.num_classes_teacher)
            lbls_teacher_task[global_node_idx] = node_label
            train_mask_teacher[global_node_idx] = True

    for sample_idx_in_val_set in range(len(val_dataset)):
        val_sample = val_dataset[sample_idx_in_val_set]
        if not val_sample['is_valid']: continue
        global_node_idx = sample_id_to_idx.get(val_sample['id'])
        if global_node_idx is not None:
            node_label_val = derive_node_label_from_seg_mask(val_sample['mask'], cfg.data.num_classes_teacher)
            lbls_teacher_task[global_node_idx] = node_label_val
            val_mask_teacher[global_node_idx] = True
            if train_mask_teacher[global_node_idx]:
                 train_mask_teacher[global_node_idx] = False
    
    logging.info(f"Teacher masks: Train {train_mask_teacher.sum().item()}, Val {val_mask_teacher.sum().item()}")

    logging.info("Constructing hypergraph G...")
    if not os.path.exists(cfg.data.edge_list_file):
        raise FileNotFoundError(f"Edge list file not found: {cfg.data.edge_list_file}")
    edge_list = torch.load(cfg.data.edge_list_file)

    if cfg.model.teacher_model_type == 'Graph': # Assuming config distinguishes graph/hypergraph model type
        G = Graph(num_vertices, edge_list)
        G.add_extra_selfloop()
    else:
        G = Hypergraph(num_vertices, edge_list)
        G = fix_iso_v(G)
    G = G.to(device)
    logging.info(f"Hypergraph G constructed: {G}")

    evaluator_teacher = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    num_classes_teacher = cfg.data.num_classes_teacher

    if cfg.model.teacher_nn == "hgnn":
        net_t = MyHGNN(resnet_feature_dim, cfg.model.teacher_hid, num_classes_teacher, use_bn=False)
    elif cfg.model.teacher_nn == "hgnnp":
        net_t = HGNNP(resnet_feature_dim, cfg.model.teacher_hid, num_classes_teacher, use_bn=False)
    elif cfg.model.teacher_nn == "gcn":
        net_t = MyGCN(resnet_feature_dim, cfg.model.teacher_hid, num_classes_teacher, use_bn=False)
    else:
        raise NotImplementedError(f"Teacher model {cfg.model.teacher_nn} not configured.")
    
    net_t = net_t.to(device)
    optimizer_t = optim.Adam(net_t.parameters(), lr=cfg.train.lr_teacher, weight_decay=cfg.train.weight_decay_teacher)

    best_state_t, best_epoch_t, best_val_t = None, 0, 0.0
    logging.info("Starting Teacher Training...")
    for epoch in range(cfg.train.num_epochs_teacher):
        if not train_mask_teacher.any():
            logging.warning("Teacher training mask is empty. Skipping teacher training.")
            break
        train_teacher(net_t, X_teacher, G, lbls_teacher_task, train_mask_teacher, optimizer_t)
        if val_mask_teacher.any():
            with torch.no_grad():
                val_res_t = valid_teacher(net_t, X_teacher, G, lbls_teacher_task, val_mask_teacher, evaluator_teacher)
            if val_res_t > best_val_t:
                best_epoch_t, best_val_t = epoch, val_res_t
                best_state_t = deepcopy(net_t.state_dict())
                logging.debug(f"Teacher Epoch {epoch}: New best val: {best_val_t:.4f}")
        else:
            best_state_t = deepcopy(net_t.state_dict())
            best_val_t = -1.0

    if best_state_t: net_t.load_state_dict(best_state_t)
    
    if train_mask_teacher.any() and best_state_t:
        out_t_logits = net_t(X_teacher, G).detach()
    else:
        out_t_logits = torch.randn(num_vertices, cfg.data.num_classes_teacher, device=device)
        logging.warning("Using random teacher logits.")

    num_segmentation_classes_student = cfg.data.num_segmentation_classes_student
    hc = None
    if cfg.model.student_nn == "light_hgnnp" and cfg.loss.lamb < 1.0 :
        hc = HighOrderConstraint(net_t, X_teacher, G, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau_hc).to(device)

    net_s = MyMLPs(resnet_feature_dim, cfg.model.student_hid, num_segmentation_classes_student).to(device)
    optimizer_s = optim.Adam(net_s.parameters(), lr=cfg.train.lr_student, weight_decay=cfg.train.weight_decay_student)
    
    lbls_student_direct_sup = None
    train_mask_student_direct_sup = None
    
    best_state_s, best_epoch_s, best_val_s_metric = None, 0, -float('inf')
    logging.info("Starting Student Training...")
    for epoch in range(cfg.train.num_epochs_student):
        loss_stu_train = train_stu(
            net_s, X_student, G, lbls_student_direct_sup, out_t_logits, 
            train_mask_student_direct_sup, optimizer_s, hc=hc, lamb=cfg.loss.lamb, tau_kd=cfg.loss.tau_kd
        )
        current_val_s_metric = -loss_stu_train

        if best_state_s is None or current_val_s_metric > best_val_s_metric:
            best_epoch_s, best_val_s_metric = epoch, current_val_s_metric
            best_state_s = deepcopy(net_s.state_dict())
            logging.debug(f"Student Epoch {epoch}: New best val metric proxy: {best_val_s_metric:.4f}")

    if best_state_s: net_s.load_state_dict(best_state_s)

    all_nodes_mask = torch.ones(num_vertices, dtype=torch.bool, device=device)
    final_soft_pseudo_labels, _ = test_stu(
        net_s, X_student, None, all_nodes_mask, None, 
        num_segmentation_classes_student, cfg.data.ft_noise_level
    )
    logging.info(f"Student generated final soft pseudo-labels of shape: {final_soft_pseudo_labels.shape}")
    
    # emb_t = net_t(X_teacher, G, get_emb=True).detach()
    # emb_s = net_s(X_student, get_emb=True).detach()
    # tos_t = ho_topology_score(emb_t, G) if G.num_e > 0 else 0.0
    # tos_s = ho_topology_score(emb_s, G) if G.num_e > 0 else 0.0
    # logging.info(f"Teacher topology score: {tos_t:.4f}")
    # logging.info(f"Student topology score: {tos_s:.4f}\n")

    return {"student_soft_pseudo_labels": final_soft_pseudo_labels.cpu()}