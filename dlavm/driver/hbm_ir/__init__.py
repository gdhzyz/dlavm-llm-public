from ...adr import Op
from ._hbm_compute import *


Op.Get("accel.hbm.mvm").attrs["compute"] = MVM
Op.Get("accel.hbm.mvm_bn").attrs["compute"] = MVM
Op.Get("accel.hbm.mvm_bn_res").attrs["compute"] = MVM