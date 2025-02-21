#ifndef __RPC_SERVER_H__
#define __RPC_SERVER_H__
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <WinSock2.h> // windows平台网络库头文件
#include <WS2tcpip.h>
#pragma comment(lib, "ws2_32.lib") // 库文件
#include <iostream>
#include <string>
#include <map>
#include "./HBM_LN/HBM_DDR_ln.h"
#include "./xdma_lib/xdma_rw.h"

typedef void (*Module)(HANDLE, int, int, int);

uint16_t embeddings [65024*4096];

void read_bin(const char* bin_file, int size) {
    FILE *fp1 = fopen(bin_file, "rb");
    if (fp1==NULL) {
        printf("Can't open file: %s\n",bin_file);
    }
    fread(embeddings, sizeof(uint16_t), size, fp1);
    fclose(fp1);
}

void read_bin_split(const char* prefix_path, int numb) {
    int block = 65024 / numb;
    char bin_file[100];
    for (int i = 0; i < numb; i++) {
        sprintf(bin_file, "%s/Embedding_%02d-of-%02d.bin", prefix_path, i+1, numb);
        FILE *fp1 = fopen(bin_file, "rb");
        if (fp1==NULL) {
            printf("Can't open file: %s\n",bin_file);
        }
        fread(&embeddings[block*4096*i], sizeof(uint16_t), block*4096, fp1);
        fclose(fp1);
    }
}

void set_inputs(uint16_t* data, int index, int offset) {
    memcpy(data + (offset*4096), &embeddings[4096*index], 4096*sizeof(uint16_t));
}

void General_Map_Feature_Data_demo(struct FPGA_HBM_LN_cfg cfg, int Height, int Width, int CH, uint16_t *in, int *mem[]) {
    for(int i=0;i<Height;i++) {
        for(int j=0;j<Width;j++) {
            for(int k=0;k<CH;k=k+Tout) {
                int tmp[MAX_DAT_DW*Tout/32*2] = {0};
                for(int kk=k;kk<k+Tout;kk++) {
                    if(kk<CH) 
                       tmp[kk-k]=int(in[i*cfg.CHin*cfg.Win+j*cfg.CHin+kk]);
                    else
                       tmp[kk-k]=0;
                }
                for(int m=0;m<MAX_DAT_DW*Tout/32;m++) {
                    mem[m][Height*Width*(k/Tout)+i*Width+j]=(tmp[2*m+1]<<MAX_DAT_DW)+tmp[2*m];
                }
            }
        }
    }
}

void DAT_IN_TRANS_FUNCTION_demo(struct FPGA_HBM_LN_cfg cfg, uint16_t *dat_in, int *HBM_DDR[]) {
    int *dat_in_mem[MAX_DAT_DW*Tout*Tb/32];
    for (int i = 0; i<MAX_DAT_DW*Tout*Tb/32; i++) {
        dat_in_mem[i] = (int*)malloc(sizeof(int)*cfg.Hin*cfg.Win*cfg.CHin_div_Tout);
        if (dat_in_mem[i] == NULL){printf("fail to malloc dat_in_mem \n");}
    }

    int *tp_dat_in_mem[Tb][MAX_DAT_DW*Tout/32];
    for(int j=0;j<MAX_DAT_DW*Tout/32;j++) {
        for(int i=0;i<Tb;i++) {
            tp_dat_in_mem[i][j] = (int*)malloc(sizeof(int)*cfg.Win*cfg.Hin*cfg.CHin_div_Tout);
            if (tp_dat_in_mem[i][j] == NULL){printf("fail to malloc tp_dat_in_mem \n");}
        }
    }

    // dat_in
    for(int b=0;b<Tb;b++)
        General_Map_Feature_Data_demo(cfg, cfg.Hin, cfg.Win, cfg.CHin, dat_in, tp_dat_in_mem[b]);

    for(int i=0;i<cfg.Win*cfg.Hin*cfg.CHin_div_Tout;i++) {
        for(int b=0;b<Tb;b++) {
            for(int j=0;j<MAX_DAT_DW*Tout/32;j++) {
                dat_in_mem[MAX_DAT_DW*Tout/32*b+j][i]=tp_dat_in_mem[b][j][i];
            }
        }
    }

    for(int i=0;i<cfg.Win*cfg.Hin*cfg.CHin_div_Tout;i++) {
        for(int j=0;j<Tb*AXI_DAT_WIDTH/32;j++) {
            HBM_DDR[0][i*AXI_DAT_WIDTH/32+j] = dat_in_mem[j][i];
        }
    }

    // Free malloc
    for(int i=0;i<MAX_DAT_DW*Tout*Tb/32;i++) {
        free(dat_in_mem[i]);
        dat_in_mem[i] = NULL;
    }
    for(int i=0;i<MAX_DAT_DW*Tout/32;i++) {
        free(tp_dat_in_mem[0][i]);
        tp_dat_in_mem[0][i] = NULL;
    }
}

void send_inputs(HANDLE h2cx_device, BYTE* write_data, uint64_t dat_in_addr, DWORD bin_data_size) {
    LARGE_INTEGER addr;
    addr.QuadPart = dat_in_addr;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else {
        DWORD data_size;
        if (!WriteFile(h2cx_device, write_data, bin_data_size, &data_size, NULL)){
            fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
            CloseHandle(h2cx_device);
        }
    }
}

void map_inputs(HANDLE h2cx_device, struct FPGA_HBM_LN_cfg cfg, uint16_t* data, uint64_t dat_in_addr) {
    DWORD bin_data_size = sizeof(int)*cfg.Win*cfg.Hin*cfg.CHin_div_Tout*Tb*AXI_DAT_WIDTH/32;
    BYTE* write_data = allocate_buffer(bin_data_size, 0);
    DAT_IN_TRANS_FUNCTION_demo(cfg, data, (int**)&write_data);

    if (!write_data) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", bin_data_size, GetLastError());
        CloseHandle(h2cx_device);
    }
    if (write_data == NULL) {
        printf("Error! No valid data given!\n");
        CloseHandle(h2cx_device);
    }
    send_inputs(h2cx_device, write_data, dat_in_addr, bin_data_size);
    if (write_data)	_aligned_free(write_data);
}

uint16_t get_output(HANDLE c2hx_device, uint64_t data_out) {
    DWORD bin_data_size = 128;
    BYTE* read_data = allocate_buffer(128, 0);
    if (!read_data) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", bin_data_size, GetLastError());
         CloseHandle(c2hx_device);
    }
    memset(read_data, 0, bin_data_size);

    LARGE_INTEGER addr;
    addr.QuadPart = data_out;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(c2hx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(c2hx_device);
    }
    else{
        DWORD size;
        // read from device into allocated buffer
        if (!ReadFile(c2hx_device, read_data, bin_data_size, &size, NULL)){
            fprintf(stderr, "ReadFile from device %s failed with Win32 error code: %ld\n", "c2hx_device", GetLastError());
            CloseHandle(c2hx_device);
        }
    }
    uint16_t next_id = ((uint16_t*)read_data)[0];
    printf("FPGA result: %d\n", next_id);
    _aligned_free(read_data);
    return next_id;
}

class RPCClient {
typedef uint16_t (*exe_t)(RPCClient*, std::string);
  public:
    static Module mod;
    RPCClient(int client, HANDLE accel, HANDLE c2hx, HANDLE h2cx, uint64_t data_in, uint64_t data_out) {
        this->client = client;
        this->accel = accel;
        this->c2hx = c2hx;
        this->h2cx = h2cx;
        this->data_in = data_in;
        this->data_out = data_out;
        this->last_token = 0;
        command["rkvc"] = run_model_kvcache;
        command["rkvs"] = run_model_kvcache_show;
        memset(r_buffer,0,sizeof(r_buffer));
        memset(s_buffer,0,sizeof(s_buffer));
    }
    ~RPCClient() { }

    void wrap_recv() {
        memset(r_buffer,0,sizeof(r_buffer));
        if (recv(client, r_buffer, sizeof(r_buffer), 0)<=0) {
            perror("receive"); 
            exit(-1);
        }
    }

    void wrap_send() {
        if (send(client, s_buffer, sizeof(s_buffer), 0)<=0) {
            perror("send"); 
            exit(-1);
        }
        memset(s_buffer,0,sizeof(s_buffer));
    }

    int main_loop() {
        wrap_recv();
        char* message = r_buffer;
        std::cout << message << std::endl;
        if (isEqual("command", message)) {
            message += 7;
            std::string name;
            std::string attrs;
            name.assign(message, 4);
            message += 4;
            if (isEqual("attrs", message)) {
                message += 5;
                int size = atoi(std::string().assign(message, 4).c_str());
                message += 4;
                attrs.assign(message, size);
                message += size;
            } else {
                attrs = "";
            }
            exe_t func;
            if ((func = command[name]) != nullptr) {
                uint16_t ret_state = func(this, attrs);
                memcpy(s_buffer, &ret_state, sizeof(uint16_t));
                std::cout << "response:" << ret_state << std::endl;
            } else {
                uint16_t ret_state = -1;
                memcpy(s_buffer, &ret_state, sizeof(uint16_t));
                sprintf(s_buffer+sizeof(uint16_t), "not found command function, please retry!");
            }
        } else {
            if (isEqual("quit", message)) {
                return 0; 
            }
            uint16_t ret_state = -2;
            memcpy(s_buffer, &ret_state, sizeof(uint16_t));
            sprintf(s_buffer+sizeof(uint16_t), "not rpc command, please retry!");
        }
        wrap_send();
        return 1; 
    }

    static uint16_t run_model_kvcache(RPCClient* client, std::string attrs) {
        char* attrs_ptr = const_cast<char*>(attrs.c_str());
        uint16_t token = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t kvcache = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t memory = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t* input_ids = (uint16_t*)(attrs_ptr);
        uint16_t* data = (uint16_t*)malloc(2*(token*4096));
        uint16_t index = 0;
        uint16_t ids[512];
        for (int i = 0; i < token; i++) {
            set_inputs(data, input_ids[i], i);
        }
        struct FPGA_HBM_LN_cfg EMBEDDED_INPUT_CFG = GetFPGA_HBM_LN_cfg(
            /*Height*/ token, /*Width_in*/ 4096, /*RMS_Norm*/ 0, /*Hin*/ 1, /*RELU_EN*/ 0,
            /*DAT_IN_BASE_ADDR*/  client->data_in,
            /*LN_WT_BASE_ADDR*/   NULL,
            /*DAT_OUT_BASE_ADDR*/ NULL
        );
        map_inputs(client->h2cx, EMBEDDED_INPUT_CFG, data, client->data_in);
        free(data);
        data = NULL;
        token += client->last_token;
        printf("%d, %d, %d\n", token, kvcache, input_ids[0]);
        mod(client->accel, token, 0, client->last_token);
        ids[index] = get_output(client->c2hx, client->data_out);
        DWORD bin_data_size = 4096*2;
        BYTE* kvcache_data = allocate_buffer(bin_data_size, 0);
        while (ids[index] != 2 && token < 1024) {
            set_inputs((uint16_t*)kvcache_data, ids[index], 0);
            send_inputs(client->h2cx, kvcache_data, client->data_in, bin_data_size);
            index ++;
            token ++;
            mod(client->accel, token, 1, client->last_token);
            ids[index] = get_output(client->c2hx, client->data_out);
        }
        _aligned_free(kvcache_data);
        kvcache_data = NULL;
        index ++;
        if (memory)
            client->last_token = token + 1;
        else
            client->last_token = 0;
        memcpy(client->s_buffer, &index, sizeof(uint16_t));
        memcpy(client->s_buffer+2, ids, sizeof(uint16_t)*index);
        client->wrap_send();
        return 0;
    }

    static uint16_t run_model_kvcache_show(RPCClient* client, std::string attrs) {
        int send_numb = 32;
        char* attrs_ptr = const_cast<char*>(attrs.c_str());
        uint16_t token = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t kvcache = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t memory = *((uint16_t*)attrs_ptr); attrs_ptr += 2;
        uint16_t* input_ids = (uint16_t*)(attrs_ptr);
        uint16_t* data = (uint16_t*)malloc(2*(token*4096));
        uint16_t index = 0;
        uint16_t ids[128];
        for (int i = 0; i < token; i++) {
            set_inputs(data, input_ids[i], i);
        }
        struct FPGA_HBM_LN_cfg EMBEDDED_INPUT_CFG = GetFPGA_HBM_LN_cfg(
            /*Height*/ token, /*Width_in*/ 4096, /*RMS_Norm*/ 0, /*Hin*/ 1, /*RELU_EN*/ 0,
            /*DAT_IN_BASE_ADDR*/  client->data_in,
            /*LN_WT_BASE_ADDR*/   NULL,
            /*DAT_OUT_BASE_ADDR*/ NULL
        );
        map_inputs(client->h2cx, EMBEDDED_INPUT_CFG, data, client->data_in);
        free(data);
        data = NULL;
        token += client->last_token;
        printf("%d, %d, %d\n", token, kvcache, input_ids[0]);
        mod(client->accel, token, 0, client->last_token);
        ids[index] = get_output(client->c2hx, client->data_out);
        DWORD bin_data_size = 4096*2;
        BYTE* kvcache_data = allocate_buffer(bin_data_size, 0);
        while (ids[index] != 2 && token < 128) {
            set_inputs((uint16_t*)kvcache_data, ids[index], 0);
            send_inputs(client->h2cx, kvcache_data, client->data_in, bin_data_size);
            index ++;
            token ++;
            if (index == send_numb) {
                memcpy(client->s_buffer, &index, sizeof(uint16_t));
                memcpy(client->s_buffer+2, ids, sizeof(uint16_t)*index);
                client->wrap_send();
                index = 0;
            }
            mod(client->accel, token, 1, client->last_token);
            ids[index] = get_output(client->c2hx, client->data_out);
        }
        _aligned_free(kvcache_data);
        kvcache_data = NULL;
        index ++;
        if (memory)
            client->last_token = token + 1;
        else
            client->last_token = 0;
        memcpy(client->s_buffer, &index, sizeof(uint16_t));
        memcpy(client->s_buffer+2, ids, sizeof(uint16_t)*index);
        client->wrap_send();
        return 0;
    }
  private:
    bool isEqual(const std::string& target, char* source) {
        for (int i = 0; i < target.size(); i++) {
            if (target[i] != source[i]) {
                return false;
            }
        }
        return true;
    }
    char r_buffer[1024];
    char s_buffer[1024];
    int client;
    HANDLE accel;
    HANDLE c2hx;
    HANDLE h2cx;
    uint64_t data_in;
    uint64_t data_out;
    int last_token;
    std::map<std::string, exe_t> command;
};

bool init_Socket() {
    WSADATA wsadata;
    if (0 != WSAStartup(MAKEWORD(2, 2), &wsadata)) {
        perror("WSAStartup");
        return false;
    }
    return true;
}

#endif
