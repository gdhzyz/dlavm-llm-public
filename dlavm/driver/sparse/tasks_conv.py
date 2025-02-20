Sparsity = 4 # 2 4 8 16

if Sparsity == 2:
    DAT_DW_L0 = 8
    Sparsity_num = 8
elif Sparsity == 4:
    DAT_DW_L0 = 8
    Sparsity_num = 4
elif Sparsity == 8:
    DAT_DW_L0 = 8
    Sparsity_num = 2
elif Sparsity == 16:
    DAT_DW_L0 = 8 # 8 4 2 for 8 4 2 bit ???
    Sparsity_num = 1

Tin_L0 = 32 * (8 // DAT_DW_L0)
slice_of_CHin_L0 = (512 + Tin_L0 * 8 - 1) // (Tin_L0 * 8)
slice_of_CHin_L0_no_minus1 = (512 + Tin_L0 * 8) // (Tin_L0 * 8)
CH_in_res_Tin_div_Tin16_L0 = ((512 % (Tin_L0 * 8)) + Tin_L0 - 1) // Tin_L0
slice_of_Tin_div_Tout_L0 = (Tin_L0 + 31) // 32
WT_DW_L0 = DAT_DW_L0

MAX_DW2 = 16
base_log2Tin = 5
log2_KyKx = 8
log2_other = 11
DAT_DW_L1 = 8
log2_scale = 6
Wout_L0 = 7
Hout_L0 = 7
slice_of_CHout_L0 = 8  # (256+32-1)/32
Max_Sparsity = 8
CHout_L0 = 256
CHin_L0 = 512
Ky_L0 = 3
Kx_L0 = 3
Tout = 32
log2_Max_Sparsity = 3
PE_NUM_Per_Block = 4
MAX_DAT_DW = 8
base_Tin = 32
BRAM_NUM = 2
BRAM_DEPTH = (1<<24) // base_Tin // MAX_DAT_DW // BRAM_NUM // Max_Sparsity


def MapWeightTout(weight_in):

    weight_reorg = [[[[[[0 for _ in range(Tin_L0)] for _ in range(Tout)] for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0))] for _ in range(slice_of_CHout_L0)]
    weight_Tin = [[[[[0 for _ in range(Tout)] for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0))] for _ in range(slice_of_CHout_L0)]
    weight_Tout = [[[[[[0 for _ in range(slice_of_Tin_div_Tout_L0)] for _ in range(Tout)] for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0))] for _ in range(slice_of_CHout_L0)]
    output_wt = [0] * slice_of_CHout_L0 * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Ky_L0 * Kx_L0 * Tout * slice_of_Tin_div_Tout_L0

    addr = 0

    for chout in range(slice_of_CHout_L0):
        for chin in range((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        for tin in range(Tin_L0):
                            if chout * Tout + tout < CHout_L0 and chin * Tin_L0 + tin < CHin_L0:
                                tp1 = weight_in[ky][kx][chin * Tin_L0 + tin][chout * Tout + tout]
                            else:
                                tp1 = 0
                            weight_reorg[chout][chin][ky][kx][tout][tin] = tp1

    for chout in range(slice_of_CHout_L0):
        for chin in range((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout * Tout + tout < CHout_L0:
                            tp2 = 0
                            for tin in range(Tin_L0):
                                tp2 |= weight_reorg[chout][chin][ky][kx][tout][tin] << (WT_DW_L0 * tin)
                            weight_Tin[chout][chin][ky][kx][tout] = tp2

    for chout in range(slice_of_CHout_L0):
        for chin in range(slice_of_CHin_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout * Tout + tout < CHout_L0:
                            tp3 = 0
                            tp4 = 0
                            if chin == (slice_of_CHin_L0 - 1):
                                CNT_MAX = Max_Sparsity if CHin_L0 % (Tin_L0 * Max_Sparsity) == 0 else CH_in_res_Tin_div_Tin16_L0
                            else:
                                CNT_MAX = Max_Sparsity
                            for cnt in range(CNT_MAX):
                                tp3 = weight_Tin[chout][chin * Max_Sparsity + cnt][ky][kx][tout]
                                for tin_out in range(slice_of_Tin_div_Tout_L0):
                                    weight_Tout[chout][chin * Max_Sparsity + cnt][ky][kx][tout][tin_out] = tp3 >> (WT_DW_L0 * Tout * tin_out)
                                    tp4 = weight_Tout[chout][chin * Max_Sparsity + cnt][ky][kx][tout][tin_out]
                                    output_wt[addr] = tp4
                                    addr += 1
    return output_wt


import random

def SparseAddrGeneratorBlock64S1(sparsity, depth):

    random_addr = [0] * (depth * CHout_L0 * Ky_L0 * Kx_L0 * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Tin_L0 // 16)

    for i in range(depth):
        random_addr[i] = random.getrandbits(4)  
        random_addr[i] <<= 12

    return random_addr


def SparseWtGeneratorBlock64S1(original_index):

    original_wt = [[[[0 for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(CHin_L0)] for _ in range(CHout_L0)]

    sparse_index_cnt = 0
    for i in range(CHout_L0):
        for k in range(Ky_L0):
            for l in range(Kx_L0):
                for j in range((((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Tin_L0) // 16):
                    addr0 = original_index[sparse_index_cnt] & 0xF
                    for jj in range(16):
                        original_wt[i][j*16+jj][k][l] = 0
                    original_wt[i][j*16+addr0][k][l] = random.randint(0, 254)
                    sparse_index_cnt += 1
    return original_wt


def MapWeightTin(weight_in):
 
    weight_reorg = [[[[[[0 for _ in range(Tin_L0)] for _ in range(Tout)] for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0))] for _ in range(slice_of_CHout_L0)]
    weight_Tin = [[[[[0 for _ in range(Tout)] for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0))] for _ in range(slice_of_CHout_L0)]
    output_wt_Tin = [0 for _ in range(slice_of_CHout_L0*((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0)*Ky_L0*Kx_L0*Tout)]

    addr = 0

    for chout in range(slice_of_CHout_L0):
        for chin in range((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        for tin in range(Tin_L0):
                            tp1 = 0
                            if (chout*Tout+tout < CHout_L0) and (chin*Tin_L0+tin < CHin_L0):
                                tp1 = weight_in[chout*Tout+tout][chin*Tin_L0+tin][ky][kx]
                            weight_reorg[chout][chin][ky][kx][tout][tin] = tp1

    for chout in range(slice_of_CHout_L0):
        for chin in range((slice_of_CHin_L0_no_minus1-1)*Max_Sparsity+CH_in_res_Tin_div_Tin16_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout*Tout+tout < CHout_L0:
                            tp2 = 0
                            for tin in range(Tin_L0):
                                # Packing each tp1 into tp2, simulating bit concatenation
                                tp2 |= weight_reorg[chout][chin][ky][kx][tout][tin] << (WT_DW_L0 * tin)
                            weight_Tin[chout][chin][ky][kx][tout] = tp2

    for chout in range(slice_of_CHout_L0):
        for chin in range(slice_of_CHin_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout * Tout + tout < CHout_L0:
                            if chin == slice_of_CHin_L0 - 1 and CHin_L0 % (Tin_L0 * Max_Sparsity) != 0:
                                CNT_MAX = CH_in_res_Tin_div_Tin16_L0
                            else:
                                CNT_MAX = Max_Sparsity

                            for cnt in range(CNT_MAX):
                                output_wt_Tin[addr] = weight_Tin[chout][chin*Max_Sparsity+cnt][ky][kx][tout]
                                addr += 1
    return output_wt_Tin


def GetSparseAddrBlock64S1(reshape_wt):

    block_size = slice_of_CHout_L0 * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Ky_L0 * Kx_L0 * Tout
    group_size = Tin_L0 // 16
    block_wt = [0 for _ in range(block_size)]
    tp_addr = [[0 for _ in range(group_size)] for _ in range(block_size)]
    tp_addr_wt = [[0 for _ in range(group_size)] for _ in range(block_size)]
    reshape_wt_sparse = [0 for _ in range(block_size)]

    for i in range(block_size):
        for m in range(group_size):
            tp_weight = reshape_wt[i] >> (WT_DW_L0 * 16 * m) & ((1 << (WT_DW_L0 * 16)) - 1)
            if tp_weight == 0:
                tp_addr[i][m] = 0
                tp_addr_wt[i][m] = 0
            else:
                for k in range(16):
                    weight = tp_weight >> (WT_DW_L0 * k) & ((1 << WT_DW_L0) - 1)
                    if weight != 0:
                        tp_addr[i][m] = k
                        tp_addr_wt[i][m] = weight
                        break

        sparse_wt = 0
        for j in range(4):
            sparse_wt |= (tp_addr[i][j] >> 3) << (j+44)
            sparse_wt |= (tp_addr[i][j] & 0x7) << j*3+32
        
        for j in range(4):
            sparse_wt |= tp_addr_wt[i][j] << (8*j)

        reshape_wt_sparse[i] = sparse_wt
        block_wt[i] = sparse_wt

    return block_wt


def sort2(x, y):
    if x > y:
        x, y = y, x
    return x, y

def sort4(i_dat):
    va = (i_dat >> 9) & 0b111
    vb = (i_dat >> 6) & 0b111
    vc = (i_dat >> 3) & 0b111
    vd = i_dat & 0b111
    
    va, vc = sort2(va, vc)
    vb, vd = sort2(vb, vd)
    va, vb = sort2(va, vb)
    vc, vd = sort2(vc, vd)
    vb, vc = sort2(vb, vc)
    
    o_dat = (va << 9) | (vb << 6) | (vc << 3) | vd
    
    return o_dat

def SparseAddrGeneratorBlock32(sparsity, depth):

    block_size = CHout_L0 * Ky_L0 * Kx_L0 * (((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Tin_L0 // 8)
    random_addr = [0 for _ in range(block_size)]

    for i in range(depth):
        tp_addr = random.randint(0, 2**12 - 1)
        
        if sparsity == 2:
            tp_addr &= 0b111
        elif sparsity == 4:
            tp_addr &= 0b111111
            if (tp_addr & 0b111) == ((tp_addr >> 3) & 0b111):
                tp_addr = (tp_addr & 0b111111000111) | (((((tp_addr >> 3) & 0b111) + 1) & 0b111) << 3)
        elif sparsity == 6:
            tp_addr &= 0b111111111
            if ((tp_addr & 0b111) == ((tp_addr >> 3) & 0b111) or
                ((tp_addr >> 6) & 0b111) == ((tp_addr >> 3) & 0b111) or
                ((tp_addr >> 6) & 0b111) == (tp_addr & 0b111)):
                tp_addr = (tp_addr & 0b111111000111) | ((((tp_addr & 0b111) + 1) & 0b111) << 3)
                tp_addr = (tp_addr & 0b111000111111) | ((((tp_addr & 0b111) + 2) & 0b111) << 6)
        elif sparsity == 8:
            groups = [tp_addr & 0b111, (tp_addr >> 3) & 0b111, (tp_addr >> 6) & 0b111, (tp_addr >> 9) & 0b111]
            if len(set(groups)) < len(groups): 
                tp_addr = (tp_addr & 0b111111000111) | ((((tp_addr & 0b111) + 1) & 0b111) << 3)
                tp_addr = (tp_addr & 0b111000111111) | ((((tp_addr & 0b111) + 2) & 0b111) << 6)
                tp_addr = (tp_addr & 0b000111111111) | ((((tp_addr & 0b111) + 4) & 0b111) << 9)

        tp_out_addr = sort4(tp_addr)
        random_addr[i] = tp_out_addr & 0xFFFF

    return random_addr


def SparseWtGeneratorBlock32(original_index):

    block_size = CHout_L0 * Ky_L0 * Kx_L0 * (((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Tin_L0 // 8)
    original_wt = [[[[0 for _ in range(Kx_L0)] for _ in range(Ky_L0)] for _ in range(CHin_L0)] for _ in range(CHout_L0)]
    sparse_index_cnt = 0
    
    for i in range(CHout_L0):
        for k in range(Ky_L0):
            for l in range(Kx_L0):
                for j in range((((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Tin_L0) // 8):
                    index = original_index[sparse_index_cnt]
                    addr0 = index & 0b111
                    addr1 = (index >> 3) & 0b111
                    addr2 = (index >> 6) & 0b111
                    addr3 = (index >> 9) & 0b111
                    
                    if Sparsity == 2:
                        original_wt[i][j * 8 + addr0][k][l] = random.randint(0, 254)
                    
                    elif Sparsity == 4:
                        original_wt[i][j * 8 + addr0][k][l] = random.randint(0, 253)
                        original_wt[i][j * 8 + addr1][k][l] = random.randint(0, 252)
                    
                    elif Sparsity == 6:
                        original_wt[i][j * 8 + addr0][k][l] = random.randint(0, 251)
                        original_wt[i][j * 8 + addr1][k][l] = random.randint(0, 250)
                        original_wt[i][j * 8 + addr2][k][l] = random.randint(0, 249)
                    
                    elif Sparsity == 8:
                        original_wt[i][j * 8 + addr0][k][l] = random.randint(0, 248)
                        original_wt[i][j * 8 + addr1][k][l] = random.randint(0, 247)
                        original_wt[i][j * 8 + addr2][k][l] = random.randint(0, 246)
                        original_wt[i][j * 8 + addr3][k][l] = random.randint(0, 245)
                    
                    sparse_index_cnt += 1

    return original_wt


def GetSparseAddrBlock32(reshape_wt):

    reshape_wt_sparse = [[0 for _ in range(Tin_L0 // 8)] for _ in range(slice_of_CHout_L0 * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Ky_L0 * Kx_L0 * Tout)]
    
    for i in range(len(reshape_wt_sparse)):
        for j in range(Tin_L0 // 8):
            tp_weight = reshape_wt[i] >> (WT_DW_L0 * 8 * j) & ((1 << (WT_DW_L0 * 8)) - 1)

            addr_cnt = 0
            tp_addr = 0
            tp_addr_wt = 0
            weight = [0] * 8
            for k in range(8):
                weight[k] = (tp_weight >> (WT_DW_L0 * k)) & ((1 << WT_DW_L0) - 1)
                if weight[k] != 0:
                    tp_addr += k * (1 << (addr_cnt * 3))
                    tp_addr_wt += weight[k] * (1 << (addr_cnt * 8))
                    addr_cnt += 1

            sort_addr = sort4(tp_addr)
            
            addr0, addr1, addr2, addr3 = (sort_addr >> 0) & 7, (sort_addr >> 3) & 7, (sort_addr >> 6) & 7, (sort_addr >> 9) & 7
            if Sparsity == 2:
                reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, 0, weight[addr0])
            elif Sparsity == 4:
                if addr1 == 0 and addr0 == 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, 0, weight[addr0])
                else:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, weight[addr1], weight[addr0])
            elif Sparsity == 6:
                if addr2 == 0 and addr1 == 0 and addr0 == 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, 0, weight[addr0])
                elif addr2 == 0 and addr1 == 0 and addr0 != 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, weight[addr1], weight[addr0])
                else:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, weight[addr2], weight[addr1], weight[addr0])
            elif Sparsity == 8:
                if addr3 == 0 and addr2 == 0 and addr1 == 0 and addr0 == 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, 0, weight[addr0])
                elif addr3 == 0 and addr2 == 0 and addr1 == 0 and addr0 != 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, 0, weight[addr1], weight[addr0])
                elif addr3 == 0 and addr2 == 0 and addr1 != 0 and addr0 != 0:
                    reshape_wt_sparse[i][j] = (0, sort_addr, 0, weight[addr2], weight[addr1], weight[addr0])
                else:
                    reshape_wt_sparse[i][j] = (0, sort_addr, weight[addr3], weight[addr2], weight[addr1], weight[addr0])
    
    return reshape_wt_sparse


def PruningBlockWeightBlock32(sparsity, block8_wt):

    num_blocks = slice_of_CHout_L0 * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Ky_L0 * Kx_L0 * Tout
    block_wt = [0 for _ in range(num_blocks)]

    for i in range(num_blocks):
        block32_wt = []
        for j in range(sparsity // 2):
            tp_block8_wt = [block8_wt[i][j * (Tin_L0 // 8 // (sparsity // 2)) + m] for m in range(Tin_L0 // 8 // (sparsity // 2))]

            if sparsity == 2:
                combined_data = 0
                for k in range(4):
                    tp_dat = tp_block8_wt[k] & 0xFF
                    tp_addr = (tp_block8_wt[k] >> 32) & 0x7
                    combined_data |= tp_dat << (k * 8)
                    combined_data |= tp_addr << (32 + k * 3)
                block32_wt.append(combined_data)
            
            elif sparsity == 4:
                combined_data = 0
                for k in range(2):
                    tp_dat = tp_block8_wt[k] & 0xFFFF
                    tp_addr = (tp_block8_wt[k] >> 32) & 0x3F
                    combined_data |= tp_dat << (k * 16)
                    combined_data |= tp_addr << (32 + k * 6)
                block32_wt.append(combined_data)
            
            elif sparsity == 8:
                block32_wt.append(tp_block8_wt[0])

        combined_wt = 0
        for index, value in enumerate(block32_wt):
            shift_amount = index * (WT_DW_L0 * 4 + 16)
            combined_wt |= value << shift_amount
        
        block_wt[i] = combined_wt

    return block_wt


def CompressSparseWtAndAddrBlock32(block_wt):

    out_wt_max_sparsity = [0] * (slice_of_CHout_L0 * slice_of_CHin_L0 * Ky_L0 * Kx_L0 * Tout)
    addr = 0

    for chout in range(slice_of_CHout_L0):
        for chin in range(slice_of_CHin_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout * Tout + tout < CHout_L0:
                            tp3 = 0
                            if chin == (slice_of_CHin_L0 - 1):
                                if CHin_L0 % (Tin_L0 * Max_Sparsity) == 0:
                                    CNT_MAX = Max_Sparsity
                                else:
                                    CNT_MAX = CH_in_res_Tin_div_Tin16_L0
                            else:
                                CNT_MAX = Max_Sparsity
                            for cnt in range(CNT_MAX):
                                index = (chout * ((slice_of_CHin_L0_no_minus1 - 1) * Max_Sparsity + CH_in_res_Tin_div_Tin16_L0) * Ky_L0 * Kx_L0 * Tout) + (chin * Ky_L0 * Kx_L0 * Tout * Max_Sparsity) + (ky * Kx_L0 * Tout * CNT_MAX) + (kx * Tout * CNT_MAX) + (tout * CNT_MAX) + cnt
                                tp3_segment = block_wt[index] & ((1 << (PE_NUM_Per_Block * WT_DW_L0 * 2 * (Sparsity / (16 / Max_Sparsity)))) - 1)
                                shift_amount = PE_NUM_Per_Block * WT_DW_L0 * (Sparsity / (16 / Max_Sparsity)) * 2 * (cnt + 1) - 1
                                tp3 |= (tp3_segment << shift_amount)
                            out_wt_max_sparsity[addr] = tp3
                            addr += 1
    return out_wt_max_sparsity


def ExtendSparseWtAndAddrBlock32(out_wt_max_sparsity):

    out_wt_final = [0 for _ in range(slice_of_CHout_L0 * ((slice_of_CHin_L0 - 1) * Sparsity + (CH_in_res_Tin_div_Tin16_L0 + Sparsity_num - 1) // Sparsity_num * 2) * Ky_L0 * Kx_L0 * Tout)]
    addr = 0

    for chout in range(slice_of_CHout_L0):
        for chin in range(slice_of_CHin_L0):
            for ky in range(Ky_L0):
                for kx in range(Kx_L0):
                    for tout in range(Tout):
                        if chout * Tout + tout < CHout_L0:
                            tp4 = [0 for _ in range(Tin_L0)]
                            CNT_MAX = Sparsity if chin != (slice_of_CHin_L0 - 1) or CHin_L0 % (Tin_L0 * Max_Sparsity) == 0 else (CH_in_res_Tin_div_Tin16_L0 + Sparsity_num - 1) // Sparsity_num * 2
                            
                            for cnt in range(CNT_MAX):
                                index = (chout * slice_of_CHin_L0 * Ky_L0 * Kx_L0 * Tout) + (chin * Ky_L0 * Kx_L0 * Tout) + (ky * Kx_L0 * Tout) + (kx * Tout) + tout
                                segment = (cnt + 1) * WT_DW_L0 * Tin_L0 - 1
                                start = cnt * WT_DW_L0 * Tin_L0
                                tp4_segment = out_wt_max_sparsity[index][start:segment + 1]
                                tp4[start:segment + 1] = tp4_segment
                            
                            out_wt_final[addr] = tp4
                            addr += 1

    return out_wt_final

###############################################
def CSB_Write(a, b):   # should be replaced!!!#
    return a + b       # should be replaced!!!#
                       # should be replaced!!!#
def CSB_Read(a, b):    # should be replaced!!!#
    return a + b       # should be replaced!!!#
###############################################

def RunConvSingleTime(CHin, Hin, Win, CHout,
                        Kx, Ky, Sx, Sy,
                        pad_x, pad_y, relu_en, L0_DW, L1_DW,
                        feature_in_base, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                        wt_base_addr, wt_num_div_Tout, wt_num_div_Tin, wt_scale, conv_out_scale,
                        feature_out_base, feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                        out_width, out_height, dat_buf_num, dma_dat_reuse, dma_wt_reuse):
    
    shift_value = 0
    shift_sign = 0
    
    if CHin % (Tin_L0 * Max_Sparsity) == 0:
        CH_in_res_Tin_div_Tin16_minus1 = Max_Sparsity - 1
    else:
        CH_in_res_Tin_div_Tin16_minus1 = ((CHin % (Tin_L0 * Max_Sparsity)) + Tin_L0 - 1) // Tin_L0 - 1
    
    if CHin % (Tin_L0 * Max_Sparsity) == 0:
        CH_in_res_Tin_div_SparsityNum_Tin16_minus1 = Sparsity // (16 / Max_Sparsity) - 1
    else:
        CH_in_res_Tin_div_SparsityNum_Tin16_minus1 = ((CHin % (Tin_L0 * Max_Sparsity)) + (Sparsity_num * Tin_L0) - 1) // (Sparsity_num * Tin_L0) - 1
    
    if CHin % (Tin_L0 * Max_Sparsity) == 0:
        CH_in_res_Tin_div_SparsityNum_mul2_minus1 = Sparsity - 1
    else:
        CH_in_res_Tin_div_SparsityNum_mul2_minus1 = (CH_in_res_Tin_div_Tin16_minus1 + Sparsity_num - 1) // Sparsity_num * 2 - 1
    
    if CHin % Tin_L0 == 0:
        CH_in_res_Tin_res_16_div_Tout_minus1 = Tin_L0 // Tout - 1
    else:
        CH_in_res_Tin_res_16_div_Tout_minus1 = (((CHin % (Tin_L0 * Max_Sparsity)) % Tin_L0) + Tout - 1) // Tout - 1
    
    Tin_factor = MAX_DAT_DW // DAT_DW_L0
    sparsity = Sparsity

    shift_sign = 0  # right shift
    shift_value = (feature_in_scale + wt_scale) - conv_out_scale
    if shift_value < 0:
        shift_value = conv_out_scale - (feature_in_scale + wt_scale)
        shift_sign = 1  # left shift

    CSB_Write(2, Tin_factor)
    CSB_Write(3, dat_buf_num)
    CSB_Write(4, Win)
    CSB_Write(5, Hin)
    CSB_Write(6, Win * Hin)
    CSB_Write(7, (CHin + Tout - 1) // Tout)
    CSB_Write(8, (CHin + (Tin_L0 * Max_Sparsity) - 1) // (Tin_L0 * Max_Sparsity))
    CSB_Write(31, CH_in_res_Tin_res_16_div_Tout_minus1)
    CSB_Write(41, CH_in_res_Tin_div_Tin16_minus1)
    CSB_Write(42, CH_in_res_Tin_div_SparsityNum_Tin16_minus1)
    CSB_Write(43, CH_in_res_Tin_div_SparsityNum_mul2_minus1)
    CSB_Write(9, pad_y)
    CSB_Write(10, pad_x)
    CSB_Write(11, Sx)
    CSB_Write(12, Sy)
    CSB_Write(13, Kx)
    CSB_Write(14, Ky)
    CSB_Write(15, out_width)
    CSB_Write(16, out_width * out_height)
    CSB_Write(17, out_height)
    CSB_Write(18, CHout)
    CSB_Write(19, (CHout + Tout - 1) // Tout)
    CSB_Write(33, ((out_width * out_height + Tout - 1) // Tout))
    CSB_Write(34, feature_out_base)
    CSB_Write(35, feature_out_surface_stride)
    CSB_Write(36, feature_out_line_stride)
    CSB_Write(38, relu_en)
    CSB_Write(39, (L1_DW - 1))
    CSB_Write(40, sparsity)

    CSB_Write(20, feature_in_base)
    CSB_Write(21, feature_in_surface_stride)
    CSB_Write(22, feature_in_line_stride)
    CSB_Write(23, (wt_num_div_Tout // Tin_factor))
    CSB_Write(32, wt_num_div_Tin)
    CSB_Write(24, wt_base_addr)
    CSB_Write(25, (dma_wt_reuse << 1) | dma_dat_reuse)

    CSB_Write(26, (shift_sign << (log2_scale - 2)) | shift_value)

    # Kick off the run
    CSB_Write(0, 0b011111)

    rdata = CSB_Read(37)
    while rdata != 1:
        rdata = CSB_Read(37)


def RunConv(CHin, Hin, Win, CHout, Kx, Ky, Sx, Sy, pad_x, pad_y, relu_en, L0_DW, L1_DW,
            feature_in_base, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
            wt_base_addr, wt_size_in_bytes, wt_num_div_Tin, wt_scale, conv_out_scale,
            feature_out_base, feature_out_surface_stride, feature_out_line_stride, feature_out_scale):

    out_width = ((Win + 2 * pad_x - Kx) // Sx + 1)
    out_height = ((Hin + 2 * pad_y - Ky) // Sy + 1)
    overlap = Ky - Sy
    dat_num_per_row = Win * slice_of_CHin_L0
    dat_banks_min = (dat_num_per_row * Ky + BRAM_DEPTH - 1) // BRAM_DEPTH
    wt_banks_min = (Kx * Ky * Tout * slice_of_CHin_L0 + BRAM_DEPTH - 1) // BRAM_DEPTH

    if (dat_banks_min + wt_banks_min) > BRAM_NUM:
        print("Resource limitation exceeded")
        return

    mininum_bw = 0
    best_dat_banks = 0
    best_method = 0

    for dat_buf_num in range(dat_banks_min, BRAM_NUM - wt_banks_min + 1):
        wt_banks = BRAM_NUM - dat_buf_num
        out_ch_slice = (BRAM_DEPTH * wt_banks // (Kx * Ky * Tout * slice_of_CHin_L0)) * Tout

        if out_ch_slice >= CHout:
            out_ch_slice = CHout
            CHout_Split_Times = 1
        else:
            CHout_Split_Times = (CHout + out_ch_slice - 1) // out_ch_slice

        out_ch_slice_last = CHout % out_ch_slice if CHout % out_ch_slice != 0 else out_ch_slice

        out_height_first = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row + pad_y - Ky) // Sy + 1)
        in_height_first = (out_height_first - 1) * Sy + Ky - pad_y

        out_height_middle = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row - Ky) // Sy + 1)
        in_height_middle = (out_height_middle - 1) * Sy + Ky

        if out_height_first >= out_height:
            out_height_first = out_height
            in_height_first = Hin

        Hout_Split_Times = (out_height - out_height_first) // out_height_middle + 1 if (out_height - out_height_first) % out_height_middle == 0 else (out_height - out_height_first) // out_height_middle + 2
        out_height_last = (out_height - out_height_first) % out_height_middle if (out_height - out_height_first) % out_height_middle != 0 else out_height_middle

        in_height_last = Hin - in_height_first + overlap - (Hout_Split_Times - 2) * (in_height_middle - overlap)
        total_bw_if_reuse_wt = (dat_num_per_row * Hin + dat_num_per_row * overlap * (Hout_Split_Times - 1)) * CHout_Split_Times + Kx * Ky * CHout * slice_of_CHin_L0
        total_bw_if_reuse_dat = Hout_Split_Times * Kx * Ky * CHout * slice_of_CHin_L0 + dat_num_per_row * Hin + dat_num_per_row * overlap * (Hout_Split_Times - 1)

        if mininum_bw == 0 or total_bw_if_reuse_wt < mininum_bw:
            best_dat_banks = dat_buf_num
            mininum_bw = total_bw_if_reuse_wt
            best_method = 0

        if mininum_bw == 0 or total_bw_if_reuse_dat < mininum_bw:
            best_dat_banks = dat_buf_num
            mininum_bw = total_bw_if_reuse_dat
            best_method = 1

    dat_buf_num = best_dat_banks

    wt_banks = BRAM_NUM - dat_buf_num
    out_ch_slice = (BRAM_DEPTH * wt_banks // (Kx * Ky * Tout * slice_of_CHin_L0)) * Tout

    if out_ch_slice >= CHout:
        out_ch_slice = CHout
        CHout_Split_Times = 1
    else:
        CHout_Split_Times = (CHout + out_ch_slice - 1) // out_ch_slice

    out_ch_slice_last = CHout % out_ch_slice if CHout % out_ch_slice != 0 else out_ch_slice

    out_height_first = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row + pad_y - Ky) // Sy + 1)
    in_height_first = (out_height_first - 1) * Sy + Ky - pad_y

    out_height_middle = (((BRAM_DEPTH * dat_buf_num) // dat_num_per_row - Ky) // Sy + 1)
    in_height_middle = (out_height_middle - 1) * Sy + Ky

    if out_height_first >= out_height:
        out_height_first = out_height
        in_height_first = Hin

    Hout_Split_Times = (out_height - out_height_first) // out_height_middle + 1 if (out_height - out_height_first) % out_height_middle == 0 else (out_height - out_height_first) // out_height_middle + 2
    out_height_last = (out_height - out_height_first) % out_height_middle if (out_height - out_height_first) % out_height_middle != 0 else out_height_middle

    in_height_last = Hin - in_height_first + overlap - (Hout_Split_Times - 2) * (in_height_middle - overlap)

    if best_method == 0:
        for n in range(CHout_Split_Times):
            for k in range(Hout_Split_Times):
                CH_in_single = CHin
                CH_out_single = out_ch_slice if n != CHout_Split_Times - 1 else out_ch_slice_last

                if k == 0:
                    line_offset_in = 0
                    line_offset_out = 0
                    pad_y_single = pad_y
                    dma_wt_reuse_single = 0
                else:
                    line_offset_in = (in_height_first - overlap) + (k - 1) * (in_height_middle - overlap)
                    line_offset_out = out_height_first + (k - 1) * out_height_middle
                    pad_y_single = 0
                    dma_wt_reuse_single = 1

                if k == 0:
                    in_height_single = in_height_first
                    out_height_single = out_height_first
                elif k == Hout_Split_Times - 1:
                    in_height_single = in_height_last
                    out_height_single = out_height_last
                else:
                    in_height_single = in_height_middle
                    out_height_single = out_height_middle

                RunConvSingleTime(CH_in_single, in_height_single, Win, CH_out_single,
                                    Kx, Ky, Sx, Sy, pad_x, pad_y_single, relu_en, L0_DW, L1_DW,
                                    feature_in_base + feature_in_line_stride * line_offset_in, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                                    wt_base_addr + wt_size_in_bytes // CHout * out_ch_slice * n, wt_size_in_bytes // CHout * CH_out_single * 8 // (WT_DW_L0 * Tout), wt_num_div_Tin // CHout * CH_out_single, wt_scale, conv_out_scale,
                                    feature_out_base + feature_out_line_stride * line_offset_out + feature_out_surface_stride * n * (out_ch_slice // Tout), feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                                    out_width, out_height_single, best_dat_banks, 0, dma_wt_reuse_single)

    else:
        for k in range(Hout_Split_Times):
            for n in range(CHout_Split_Times):
                CH_in_single = CHin
                CH_out_single = out_ch_slice if n != CHout_Split_Times - 1 else out_ch_slice_last
                dma_dat_reuse_single = 0 if n == 0 else 1

                if k == 0:
                    line_offset_in, line_offset_out, pad_y_single = 0, 0, pad_y
                    in_height_single, out_height_single = in_height_first, out_height_first
                else:
                    line_offset_in = (in_height_first - overlap) + (k - 1) * (in_height_middle - overlap)
                    line_offset_out = out_height_first + (k - 1) * out_height_middle
                    pad_y_single = 0
                    if k == Hout_Split_Times - 1:
                        in_height_single, out_height_single = in_height_last, out_height_last
                    else:
                        in_height_single, out_height_single = in_height_middle, out_height_middle

                RunConvSingleTime(CH_in_single, in_height_single, Win, CH_out_single,
                                    Kx, Ky, Sx, Sy, pad_x, pad_y_single, relu_en, L0_DW, L1_DW,
                                    feature_in_base + feature_in_line_stride * line_offset_in, feature_in_surface_stride, feature_in_line_stride, feature_in_scale,
                                    wt_base_addr + wt_size_in_bytes // CHout * out_ch_slice * n, wt_size_in_bytes // CHout * CH_out_single * 8 // (WT_DW_L0 * Tout), wt_num_div_Tin // CHout * CH_out_single, wt_scale, conv_out_scale,
                                    feature_out_base + feature_out_line_stride * line_offset_out + feature_out_surface_stride * n * (out_ch_slice // Tout), feature_out_surface_stride, feature_out_line_stride, feature_out_scale,
                                    out_width, out_height_single, best_dat_banks, dma_dat_reuse_single, 0)
       