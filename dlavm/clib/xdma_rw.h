#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
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

int verbose_msg(const char* const fmt, ...);
BYTE* allocate_buffer(size_t size, size_t alignment);
int get_devices(GUID guid, char* devpath, size_t len_devpath);
int generate_device_path(char* user_device_path, char** c2hx_device_path, char** h2cx_device_path, char* bypass_device_path);

//write and read
void user_device_write(HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* write_data);
void user_device_read (HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* read_data);

//c2h, client to host - read; 
//h2c host to client - write
extern "C" __declspec(dllexport) HANDLE* open_device();
extern "C" __declspec(dllexport) void CSB_Write(HANDLE user_device, int addr, uint32_t data);
extern "C" __declspec(dllexport) int CSB_Read(HANDLE user_device, int addr);

extern "C" __declspec(dllexport) uint32_t init(HANDLE h2cx_device, uint64_t base_addr, uint32_t size);
extern "C" __declspec(dllexport) uint32_t DDR_Update(HANDLE h2cx_device, uint64_t address, uint32_t value);
extern "C" __declspec(dllexport) uint8_t* DDR_Read(HANDLE c2hx_device, uint64_t address, uint32_t data_size);
extern "C" __declspec(dllexport) uint32_t DDR_Write(HANDLE h2cx_device, uint8_t* data, uint64_t address, uint32_t byte_size);