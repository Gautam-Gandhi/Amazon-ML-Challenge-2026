"""Every constant of the final pipeline in one place: seeds, model settings, thresholds and pretrained models."""

SEED = 0

# Countries with training labels. Every other country (France in the test set) is handled as an unseen country.
TRAINING_COUNTRIES = ("US", "India")

# ---------------------------------------------------------------------------------------------- dense training world
# A random 20% of the training S1 entities is removed so that their S2/S3 records become distractors, which
# reproduces the test set's distractor density. Models are trained and validated in this world.
KEEP_FRACTION = 0.8
WORLD_SEED = 0

# ---------------------------------------------------------------------------------------------- blocking
HASH_BUCKETS = 1 << 19          # hashed character 3-gram / word features per field
EMBED_DIM = 128                 # per tower (name, address)
BLOCKER_EPOCHS = 3
BLOCKER_BATCH = 2048
BLOCKER_LR = 0.01
BLOCKER_TAU = 0.05
BLOCKER_HARD_FRACTION = 0.5
SEARCH_K = {"name": 20, "addr": 20, "joint": 30}   # nearest records per S1 entity, per channel
REVERSE_K = 3                                       # nearest S1 entities per record (reverse channel)
NPROBE = 32
PRUNE_TOP_S1 = 15               # keep a pair if it is in the S1 entity's top 15 by joint similarity ...
PRUNE_TOP_RECORD = 2            # ... or the S1 entity is among the record's top 2
FEATURE_CHUNK_ROWS = 3_000_000  # pair features are computed (and kept on disk) in parts of this many rows

# ---------------------------------------------------------------------------------------------- matchers (XGBoost)
NEG_FRACTION = 0.3              # all positives + this share of negatives, re-weighted by 1 / NEG_FRACTION
XGB_DEPTH = 8
XGB_ETA = 0.1
XGB_ROUNDS = 2000
XGB_DEVICE = "cuda"
N_ANCHORS = 3                   # confident candidates used by the consistency features
CANDIDATE_FILTER = {"seen": 0.01, "unseen": 0.003}   # stage-1 probability needed to stay a candidate

# ---------------------------------------------------------------------------------------------- cross-encoders
CE_BAND = (0.01, 0.99)          # only pairs with a stage-2 probability in this band are re-scored
CE_MAX_LEN = 96
CE_LR = 3e-5
CE_SCORE_BATCH = 128
E5_SMALL = ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3")   # MIT, 118M parameters
BERT_BASE = ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594")               # Apache-2.0, 110M parameters
CROSS_ENCODERS = {
    # name: (pretrained model, revision, epochs, batch size)
    "ce1": E5_SMALL + (2, 64),      # trained on the uncertain band of the token matcher
    "ce2": E5_SMALL + (3, 64),      # trained on the uncertain band of the stage-2 matcher
    "ce3": BERT_BASE + (2, 32),     # same band, different architecture
}

# ---------------------------------------------------------------------------------------------- stacking
STACK_STRUCT_FEATURES = ["num1_eq", "cos_a", "cos_n", "ad_tset", "ncl_tset", "nc_tset", "r_margin2", "num_jacc",
                         "sfx_eq", "b_addr_empty", "a_s1cnt", "b_s1cnt"]
GBDT_STACK_PARAMS = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 5, "eta": 0.05,
                     "nthread": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0}
GBDT_STACK_ROUNDS = 500

# ---------------------------------------------------------------------------------------------- decoding
DECODE_GRID = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)   # threshold chosen on the dense out-of-fold predictions
UNSEEN_THRESHOLD = 0.9          # unseen countries (France)
EMPTY_THRESHOLD = 0.6           # training countries: an S1 entity without any match may take its best pair above this

# ---------------------------------------------------------------------------------------------- rules
ROLE_RULE = {"min_role": 0.1, "min_count": 200, "min_over": 0.1, "p_floor": 0.01, "p_set": 0.95}
SIBLING_RULE = {"countries": ("US",), "t_near": 0.98, "t_far": 0.95, "demote": 0.5}
ACRONYM_RULE = {"p_floor": 0.1, "p_set": 0.95}
DESCRIPTOR_RULE = {"min_s1": 200, "max_over": 1.0, "min_swaps": 200, "exclude": ("compagnie",), "demote": 0.5}
ACRONYM_CANDIDATES_P = 0.95
CANDIDATE_GATE = 0.01           # final candidate set: pairs the post-matcher stages can still accept
