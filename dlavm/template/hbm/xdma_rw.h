#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <strsafe.h>

#include <Windows.h>
#include <SetupAPI.h>
#include <INITGUID.H>
#include <WinIoCtl.h>

#include "xdma_public.h"

#pragma comment(lib, "setupapi.lib")

#define verbose_msg_TRUE 1
#define cnn_base_addr 0x20000
#define ddr_base_addr 0x200000000
#define NUM_OF_RW_CH 4
// #define BYPASS
#define AXIDMA

struct bin_inf
{
    uint64_t bin_data_to_FPGA_bar ;
    DWORD bin_data_size ;
    char * bin_data_file;
};

struct bin_inf* get_bin_inf(uint64_t bin_data_to_FPGA_bar, DWORD bin_data_size, char * bin_data_file);

int verbose_msg(const char* const fmt, ...);
BYTE* allocate_buffer(size_t size, size_t alignment);
int get_devices(GUID guid, char* devpath, size_t len_devpath);
int generate_device_path(char* user_device_path, char** c2hx_device_path, char** h2cx_device_path, char* bypass_device_path);

//c2h, client to host - read; 
//h2c host to client - write
int open_device(HANDLE *user_device, HANDLE *bypass_device, HANDLE *c2hx_device, HANDLE *h2cx_device);

void set_user_device_base_addr(HANDLE user_device, uint64_t base_address);
void user_device_write_brust(HANDLE user_device, DWORD data_size,  BYTE* write_data);

//write and read
void user_device_write(HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* write_data);
void user_device_read (HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* read_data);

void h2cx_device_write_bin(HANDLE h2cx_device, struct bin_inf* write_bin_inf);
void c2hx_device_read_bin (HANDLE c2hx_device, struct bin_inf* read_bin_inf );

void h2cx_device_write(HANDLE h2cx_device, uint64_t base_address, DWORD data_size,  BYTE* write_date);
void c2hx_device_read (HANDLE c2hx_device, uint64_t base_address, DWORD data_size,  BYTE* read_data );

void CSB_Write(HANDLE user_device, int addr, uint32_t data);
int CSB_Read(HANDLE user_device, int addr);

#include "xdma_rw.cpp"