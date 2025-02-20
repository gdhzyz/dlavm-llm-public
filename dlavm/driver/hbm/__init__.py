from ...adr import Op
from ._hbm_driver import *
from ._hbm_process import *
from ._hbm_testbench import *


Op.Get("accel.hbm.mvm").attrs["testbench"] = MVMTestbench
Op.Get("accel.hbm.mvm").attrs["cfg_id"] = [0b00001000, 2]
Op.Get("accel.hbm.mvm").attrs["driver"] = MVMDriver
Op.Get("accel.hbm.mvm").attrs["process"] = MVMProcess


Op.Get("accel.hbm.mvm_bn").attrs["testbench"] = MVMBNTestbench
Op.Get("accel.hbm.mvm_bn").attrs["cfg_id"] = [0b01001000, 2] # 2*AXI_DAT_WIDTH
Op.Get("accel.hbm.mvm_bn").attrs["driver"] = MVMBNDriver
Op.Get("accel.hbm.mvm_bn").attrs["process"] = MVMBNProcess


Op.Get("accel.hbm.mvm_bn_res").attrs["testbench"] = MVMBNResTestbench
Op.Get("accel.hbm.mvm_bn_res").attrs["cfg_id"] = [0b11001000, 2]
Op.Get("accel.hbm.mvm_bn_res").attrs["driver"] = MVMBNResDriver


Op.Get("accel.hbm.mvm_afterTRP").attrs["testbench"] = MVMafterTRPTestbench
Op.Get("accel.hbm.mvm_afterTRP").attrs["cfg_id"] = [0b00000011, 1]
Op.Get("accel.hbm.mvm_afterTRP").attrs["driver"] = MVMafterTRPDriver


Op.Get("accel.hbm.mvm_afterF2W").attrs["testbench"] = MVMafterF2WTestbench
Op.Get("accel.hbm.mvm_afterF2W").attrs["cfg_id"] = [0b00000010, 1]
Op.Get("accel.hbm.mvm_afterF2W").attrs["driver"] = MVMafterF2WDriver


Op.Get("accel.hbm.add").attrs["driver"] = AddDriver
Op.Get("accel.hbm.add").attrs["cfg_id"] = [0b00000001, 1]
Op.Get("accel.hbm.add").attrs["driver"] = AddDriver


Op.Get("accel.hbm.softmax").attrs["testbench"] = SoftmaxTestbench
Op.Get("accel.hbm.softmax").attrs["cfg_id"] = [0b00000101, 1]
Op.Get("accel.hbm.softmax").attrs["driver"] = SoftmaxDriver


Op.Get("accel.hbm.layer_norm").attrs["testbench"] = LayerNormTestbench
Op.Get("accel.hbm.layer_norm").attrs["cfg_id"] = [0b00000111, 1]
Op.Get("accel.hbm.layer_norm").attrs["driver"] = LayerNormDriver
Op.Get("accel.hbm.layer_norm").attrs["process"] = LayerNormProcess


Op.Get("accel.hbm.pos_emb").attrs["testbench"] = PosEmbTestbench
Op.Get("accel.hbm.pos_emb").attrs["cfg_id"] = [0b00000100, 1]
Op.Get("accel.hbm.pos_emb").attrs["driver"] = PosEmbDriver


Op.Get("accel.hbm.activate").attrs["testbench"] = ActivateTestbench
Op.Get("accel.hbm.activate").attrs["cfg_id"] = [0b00000110, 1]
Op.Get("accel.hbm.activate").attrs["driver"] = ActivateDriver


Op.Get("accel.hbm.trp_mvm").attrs["cfg_id"] = [0b00000011, 1]
Op.Get("accel.hbm.trp_mvm").attrs["driver"] = TRPMVMDriver


Op.Get("accel.hbm.f2w_mvm").attrs["cfg_id"] = [0b00000010, 1]
Op.Get("accel.hbm.f2w_mvm").attrs["driver"] = F2WMVMDriver


Op.Get("accel.hbm.dat2hbm").attrs["cfg_id"] = [0b0011_0000, 1]
Op.Get("accel.hbm.dat2hbm").attrs["driver"] = DAT2HBMDriver


Op.Get("accel.hbm.dat_hbm").attrs["cfg_id"] = [0b0011_0000, 1]