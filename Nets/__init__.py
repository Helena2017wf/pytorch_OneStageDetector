from .pdn import PDN
from .ssd import SSD
from .config import get_config

net_factory = {
    "PDN" : PDN,
    "SSD" : SSD
}
def get_net(name):
    return net_factory[name]