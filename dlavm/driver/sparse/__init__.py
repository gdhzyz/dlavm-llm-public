from ...adr import Op
from ._sparse_driver import *


Op.Get("accel.sparse.conv2d").attrs["driver"] = Conv2d