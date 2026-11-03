# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder
from models.motip.equivariant_set_decoder import EquivariantSetDecoder
from models.motip.equivariant_word_decoder import EquivariantWordDecoder
from models.motip.entity_pointer import CausalEntityPointer


def build(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used):
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
        use_causal_history_prototype=config.get("USE_CAUSAL_HISTORY_PROTOTYPE", False),
        use_symmetric_history_prototype=config.get("USE_SYMMETRIC_HISTORY_PROTOTYPE", False),
    ) if config["ONLY_DETR"] is False else None
    # 2. ID decoder:
    if config.get("USE_EQUIVARIANT_WORD_DECODER", False):
        decoder_class = EquivariantWordDecoder
    elif config.get("USE_EQUIVARIANT_SET_DECODER", False):
        decoder_class = EquivariantSetDecoder
    else:
        decoder_class = IDDecoder
    _id_decoder = decoder_class(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None
    _entity_pointer = CausalEntityPointer(
        feature_dim=config["FEATURE_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        aggregation_mode=config.get("POINTER_AGGREGATION_MODE", "max"),
        init_tau=config.get("POINTER_INIT_TAU", 0.5),
        consensus_residual_scale=config.get("POINTER_CONSENSUS_RESIDUAL_SCALE", 2.0),
        entity_state_residual_scale=config.get("POINTER_ENTITY_STATE_RESIDUAL_SCALE", 2.0),
        entity_state_hidden_dim=config.get("POINTER_ENTITY_STATE_HIDDEN_DIM", 64),
        id_memory_dim=config.get("POINTER_ID_MEMORY_DIM", 128),
        new_head_context=config.get("POINTER_NEW_CONTEXT_HEAD", False),
        new_context_init=config.get("POINTER_NEW_CONTEXT_INIT", 0.0),
    ) if config["ONLY_DETR"] is False and config.get("USE_CAUSAL_ENTITY_POINTER", False) else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
        entity_pointer=_entity_pointer,
        use_entity_pointer=config.get("USE_CAUSAL_ENTITY_POINTER", False),
    )

    return motip_model, detr_criterion
