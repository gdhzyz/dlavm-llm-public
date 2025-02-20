FPGA_DDR_BASE_ADDRESS = 0x200000000
FPGA_DDR_SIZE         = 0x100000000
MIN_BLOCK_SIZE        = 32 * (8 // 8)


class Block:

    def __init__(self, available, blocksize, board_addr):
        self.available = available
        self.blocksize = blocksize
        self.board_addr = board_addr


MEM = [Block(True, FPGA_DDR_SIZE, 0)]


def FPGA_Malloc(numbytes):
    global MEM
    num = len(MEM)
    ret_addr = 0
    for index in range(num):
        block = MEM[index]
        if block.available and block.blocksize >= numbytes:
            ret_addr = MEM[index].board_addr
            new_size = MEM[index].blocksize - numbytes
            if new_size > 0:
                MEM.insert(index+1, Block(True, new_size, ret_addr + numbytes))
                MEM[index].available = False
                MEM[index].blocksize = numbytes
            break
    return ret_addr
