import dlavm
from dlavm import adr
from dlavm import transform
from dlavm import codegen
from dlavm.driver import config

config.tb_sim_path = "/home/shenao/accel/hbm0725/HBM_sv"


def check_mvm_afterTRP(head, token, kvcache):
    input0 = adr.hbm.var_ddr("input0", (head[0], token, 128))
    input1 = adr.hbm.var_ddr("input1", (head[1], token, 128))
    output = adr.hbm.mvm_afterTRP(input0, input1, padding=1, kvcache=kvcache)

    output = transform.infer_type(output, dlavm.device.EdgeLLMv1)

    addr_assign = {"runtime": 0x0}
    expr, source, storage, mod = codegen.csb_test_head_ops(output, "test", addr_assign)
    _, _, _, tag = codegen.testbench_test_head_ops(output, "test", addr_assign)

    fail = 0
    for m, t in zip(mod[2]["csb_regs"], tag[2]["csb_regs"]):
        if m[0] == t[0] and m[1] == t[1]:
            if str(m[2]) != str(t[2]):
                print(m[0], m[1] - 192, m[2], t[2])
                fail += 1
        else:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if fail:
        print("/*****************************************\\")
        print("|           mvm_afterTRP Fail             |")
        print("\*****************************************/")
    else:
        print("/*****************************************\\")
        print("|           mvm_afterTRP Success          |")
        print("\*****************************************/")


def check_mvm_afterF2W(head, token, kvcache):
    input0 = adr.hbm.var_ddr("input0", (head[0], token, token))
    input1 = adr.hbm.var_ddr("input1", (head[1], token, 128))
    output = adr.hbm.mvm_afterF2W(input0, input1, padding=1, kvcache=kvcache)

    output = transform.infer_type(output, dlavm.device.EdgeLLMv1)

    addr_assign = {"runtime": 0x0}
    expr, source, storage, mod = codegen.csb_test_head_ops(output, "test", addr_assign)
    _, _, _, tag = codegen.testbench_test_head_ops(output, "test", addr_assign)

    fail = 0
    for m, t in zip(mod[2]["csb_regs"], tag[2]["csb_regs"]):
        if m[0] == t[0] and m[1] == t[1]:
            if str(m[2]) != str(t[2]):
                print(m[0], m[1] - 192, m[2], t[2])
                fail += 1
        else:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if fail:
        print("/*****************************************\\")
        print("|           mvm_afterF2W Fail             |")
        print("\*****************************************/")
    else:
        print("/*****************************************\\")
        print("|           mvm_afterF2W Success          |")
        print("\*****************************************/")


check_mvm_afterTRP(head=[32, 2], token=19, kvcache=0)
check_mvm_afterTRP(head=[32, 2], token=19, kvcache=1)
check_mvm_afterF2W(head=[32, 2], token=19, kvcache=0)
check_mvm_afterF2W(head=[32, 2], token=19, kvcache=1)