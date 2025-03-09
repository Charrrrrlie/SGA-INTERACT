from datasets.volleyball_composer import Volleyball_COMPOSER
from datasets.volleyball_mpgcn import Volleyball_MPGCN
from datasets.basketball import BasketballGAR, BasketballGAL

__all__ = {
    'Volleyball': Volleyball_COMPOSER,
    'Volleyball_MPGCN': Volleyball_MPGCN,
    'BasketballGAR': BasketballGAR,
    'BasketballGAL': BasketballGAL
}