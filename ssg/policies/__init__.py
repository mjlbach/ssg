from .gcn import GNN
from .graph_multiset_attention import GMA
from .soft_attention import SAM
from .hetero.gnn import HGNN
from .hetero.fixed_attention import HFAM
from .hetero.soft_attention import HSAM
from .hetero.hsam_limited_features import HSAML
from .hetero.hsam_goal_embeding import HSAMGE

REGISTERED_MODELS = {
    "GNN": GNN,
    "GMA": GMA,
    "SAM": SAM,
    "HGNN": HGNN,
    "HFAM": HFAM,
    "HSAM": HSAM,
    "HSAML": HSAML,
    "HSAMGE": HSAMGE
}
