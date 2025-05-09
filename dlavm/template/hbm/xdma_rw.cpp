
#include "xdma_rw.h"

int verbose_msg(const char* const fmt, ...) {
    int ret = 0;
    va_list args;
    if (verbose_msg_TRUE) {
        va_start(args, fmt);
        ret = vprintf(fmt, args);
        va_end(args);
    }
    return ret;
}

struct bin_inf* get_bin_inf(uint64_t bin_data_to_FPGA_bar, DWORD bin_data_size, char * bin_data_file){
    struct bin_inf* ret = (struct bin_inf*)malloc(sizeof(struct bin_inf));
    if (ret == NULL)
	{
		printf("failed to malloc structure\n");
		return NULL;
	}else{
        ret->bin_data_file=bin_data_file;
        ret->bin_data_size=bin_data_size;
        ret->bin_data_to_FPGA_bar=bin_data_to_FPGA_bar;
    }

    return ret;
}

BYTE* allocate_buffer(size_t size, size_t alignment) {

    if(size == 0) {
        size = 4;
    }

    if (alignment == 0) {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        alignment = sys_info.dwPageSize;
    }
    //printf("Allocating host-side buffer of size %llu, aligned to %llu bytes\n", size, alignment);
    return (BYTE*)_aligned_malloc(size, alignment);

}

int get_devices(GUID guid, char* devpath, size_t len_devpath) {
     
    HDEVINFO device_info = SetupDiGetClassDevs((LPGUID)&guid, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (device_info == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "GetDevices INVALID_HANDLE_VALUE\n");
        exit(-1);
    }

    SP_DEVICE_INTERFACE_DATA device_interface;
    device_interface.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);

    // enumerate through devices
    DWORD index;
    for (index = 0; SetupDiEnumDeviceInterfaces(device_info, NULL, &guid, index, &device_interface); ++index) {

        // get required buffer size
        ULONG detailLength = 0;
        if (!SetupDiGetDeviceInterfaceDetail(device_info, &device_interface, NULL, 0, &detailLength, NULL) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            fprintf(stderr, "SetupDiGetDeviceInterfaceDetail - get length failed\n");
            break;
        }

        // allocate space for device interface detail
        PSP_DEVICE_INTERFACE_DETAIL_DATA dev_detail = (PSP_DEVICE_INTERFACE_DETAIL_DATA)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, detailLength);
        if (!dev_detail) {
            fprintf(stderr, "HeapAlloc failed\n");
            break;
        }
        dev_detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        // get device interface detail
        if (!SetupDiGetDeviceInterfaceDetail(device_info, &device_interface, dev_detail, detailLength, NULL, NULL)) {
            fprintf(stderr, "SetupDiGetDeviceInterfaceDetail - get detail failed\n");
            HeapFree(GetProcessHeap(), 0, dev_detail);
            break;
        }

        StringCchCopy(devpath, len_devpath, dev_detail->DevicePath);
        HeapFree(GetProcessHeap(), 0, dev_detail);
    }

    SetupDiDestroyDeviceInfoList(device_info);

    return index;
}


int generate_device_path(char* user_device_path, char** c2hx_device_path, char** h2cx_device_path, char* bypass_device_path){
    // get device path from GUID
    char device_base_path[MAX_PATH + 1] = "";
    DWORD num_devices = get_devices(GUID_DEVINTERFACE_XDMA, device_base_path, sizeof(device_base_path));
    verbose_msg("Devices found: %d\n", num_devices);
    if (num_devices < 1) {
        printf("Error with num_devices < 1");
        return -1;
    }
    // extend device path to include target device node (xdma_control, xdma_user etc) 
    verbose_msg("Device base path: %s\n", device_base_path);
    char device_path[MAX_PATH + 1] = "";
    strcpy_s(device_path, sizeof(device_path), device_base_path);
    strcat_s(device_path, sizeof(device_path), "\\");
    verbose_msg("Device path: %s\n", device_path);

    //user device path generate
    strcpy_s(user_device_path, sizeof(char)*(MAX_PATH + 1), device_path);
    strcat_s(user_device_path, sizeof(char)*(MAX_PATH + 1), "user");

    //c2hx device path generate
    for(int i=0; i<4; i++){
        char c2h[10];
        sprintf(c2h, "c2h_%1d", i);
        strcpy_s(c2hx_device_path[i], sizeof(char)*(MAX_PATH + 1), device_path);
        strcat_s(c2hx_device_path[i], sizeof(char)*(MAX_PATH + 1), c2h);
    }

    //h2cx device path generate
    for(int i=0; i<4; i++){
        char h2c[10];
        sprintf(h2c, "h2c_%1d", i);
        strcpy_s(h2cx_device_path[i], sizeof(char)*(MAX_PATH + 1), device_path);
        strcat_s(h2cx_device_path[i], sizeof(char)*(MAX_PATH + 1), h2c);
    }

    //bypass device path generate
    strcpy_s(bypass_device_path, sizeof(char)*(MAX_PATH + 1), device_path);
    strcat_s(bypass_device_path, sizeof(char)*(MAX_PATH + 1), "bypass");

    verbose_msg("user_device_path path: %s\n", user_device_path);
    for(int i=0; i<4; i++){
        verbose_msg("c2h_%1d_device_path path: %s\n", i, c2hx_device_path[i]);
    }
    for(int i=0; i<4; i++){
        verbose_msg("h2c_%1d_device_path path: %s\n", i, h2cx_device_path[i]);
    }
    verbose_msg("bypass_device_path path: %s\n", bypass_device_path);

    return 0;

}

int open_device(HANDLE *user_device, HANDLE *bypass_device, HANDLE *c2hx_device, HANDLE *h2cx_device){
    // generate_device_path for user, c2h_x, h2c_x, bypass
    char *user_device_path = (char*)malloc(sizeof(char)*(MAX_PATH + 1));
    if (user_device_path == NULL){printf("malloc user_device_path error \n");return 0;}

    char *c2hx_device_path[NUM_OF_RW_CH];
    for(int i=0;i<NUM_OF_RW_CH;i++)
    {
        c2hx_device_path[i]=(char*)malloc(sizeof(char)*(MAX_PATH + 1));
        if (c2hx_device_path[i] == NULL){perror("main");return 0;}
    }

    char *h2cx_device_path[NUM_OF_RW_CH];
    for(int i=0;i<NUM_OF_RW_CH;i++)
    {
        h2cx_device_path[i]=(char*)malloc(sizeof(char)*(MAX_PATH + 1));
        if (h2cx_device_path[i] == NULL){perror("main");return 0;}
    }

    char *bypass_device_path = (char*)malloc(sizeof(char)*(MAX_PATH + 1));
    if (bypass_device_path == NULL){perror("main");return 0;}

    generate_device_path(user_device_path, c2hx_device_path, h2cx_device_path, bypass_device_path);
    
    //open device file
#ifdef AXIDMA
    user_device[0] = CreateFile(user_device_path, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (user_device[0] == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error opening device, win32 error code: %ld\n", GetLastError());
        return 0;
    }
#endif
    for(int i=0; i<NUM_OF_RW_CH; i++){
        c2hx_device[i] = CreateFile(c2hx_device_path[i], GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (c2hx_device[i] == INVALID_HANDLE_VALUE){
            fprintf(stderr, "Error opening device, win32 error code: %ld\n", GetLastError());
            return 0;
        }
    }

    for(int i=0; i<NUM_OF_RW_CH; i++){
        h2cx_device[i] = CreateFile(h2cx_device_path[i], GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h2cx_device[i] == INVALID_HANDLE_VALUE){
            fprintf(stderr, "Error opening device, win32 error code: %ld\n", GetLastError());
            return 0;
        }
    }
#ifdef BYPASS
    bypass_device[0] = CreateFile(bypass_device_path, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (bypass_device[0] == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error opening bypass_device_path device, win32 error code: %ld\n", GetLastError());
        return 0;
    }
#endif

    free(user_device_path);
    user_device_path = NULL;
    
    free(bypass_device_path);
    bypass_device_path = NULL;
    
    for(int i=0;i<NUM_OF_RW_CH;i++)
    {
        free(c2hx_device_path[i]);
        c2hx_device_path[i] = NULL;
    }
    
    for(int i=0;i<NUM_OF_RW_CH;i++)
    {
        free(h2cx_device_path[i]);
        h2cx_device_path[i] = NULL;
    }
    return 0;
}

void set_user_device_base_addr(HANDLE user_device, uint64_t base_address){
    LARGE_INTEGER addr;
    addr.QuadPart = base_address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(user_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(user_device);
    }
}

void user_device_write_brust(HANDLE user_device, DWORD data_size,  BYTE* write_data){
    // set file pointer to offset of target address within PCIe BAR
    if (!WriteFile(user_device, write_data, data_size, &data_size, NULL)){
        fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
        CloseHandle(user_device);
    }
}

void user_device_write(HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* write_data){
    LARGE_INTEGER addr;
    addr.QuadPart =cnn_base_addr + base_address;
    //int out=write_data[0]+(write_data[1]<<8)+(write_data[2]<<16)+(write_data[3]<<24);
    //printf("wiitre address: %llx, write data = %d \n",  addr.QuadPart, out);
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(user_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(user_device);
    }
    else{
            if (!WriteFile(user_device, write_data, data_size, &data_size, NULL)){
                fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
                CloseHandle(user_device);
            }
    }
}

void user_device_read(HANDLE user_device, uint64_t base_address, DWORD data_size,  BYTE* read_data){
    LARGE_INTEGER addr;
    addr.QuadPart =cnn_base_addr + base_address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(user_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(user_device);
    }
    else{
            if (!ReadFile(user_device, read_data, data_size, &data_size, NULL)){
                fprintf(stderr, "ReadFile from device %s failed with Win32 error code: %ld\n", "user_device_path", GetLastError());
                CloseHandle(user_device);
            }
    }
}

void h2cx_device_write_bin(HANDLE h2cx_device, struct bin_inf* write_bin_inf){

    
    // FILE* inputFile = fopen(write_bin_file,"rb");
    // if(inputFile==NULL)
    // {
    // 	printf("Can't open file: %s\n",write_bin_file);
    // }

    FILE* inputFile;
    if (fopen_s(&inputFile, write_bin_inf->bin_data_file, "rb") != 0) {
        fprintf(stderr, "Could not open file <%s>\n", write_bin_inf->bin_data_file);
        CloseHandle(h2cx_device);
    }

    /* determine file size */
    if (write_bin_inf->bin_data_size == 0) {
        fseek(inputFile, 0, SEEK_END);
        fpos_t fpos;
        fgetpos(inputFile, &fpos);
        fseek(inputFile, 0, SEEK_SET);
        write_bin_inf->bin_data_size = (DWORD)fpos;
    }

    BYTE* write_date = allocate_buffer(write_bin_inf->bin_data_size, 0);

    if (!write_date) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", write_bin_inf->bin_data_size, GetLastError());
        CloseHandle(h2cx_device);
    }
    write_bin_inf->bin_data_size=fread(write_date, sizeof(BYTE), write_bin_inf->bin_data_size, inputFile);

    fclose(inputFile);
    //verbose_msg("%ld bytes read from file %s\n", data_size, write_bin_file);

    if (write_date == NULL) {
        printf("Error! No valid data given!\n");
        CloseHandle(h2cx_device);
    }

    LARGE_INTEGER addr;
    addr.QuadPart = write_bin_inf->bin_data_to_FPGA_bar;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else{
            if (!WriteFile(h2cx_device, write_date, write_bin_inf->bin_data_size, &(write_bin_inf->bin_data_size), NULL)){
                fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
                CloseHandle(h2cx_device);
            }
    }



    if (write_date)	_aligned_free(write_date);
    // free(write_date);
    // write_date = NULL;
}

void c2hx_device_read_bin (HANDLE c2hx_device, struct bin_inf* read_bin_inf ){

    BYTE* read_data = allocate_buffer(read_bin_inf->bin_data_size, 0);
    if (!read_data) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", read_bin_inf->bin_data_size, GetLastError());
         CloseHandle(c2hx_device);
    }
    memset(read_data, 0, read_bin_inf->bin_data_size);

    
    LARGE_INTEGER addr;
    addr.QuadPart = read_bin_inf->bin_data_to_FPGA_bar;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(c2hx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
         CloseHandle(c2hx_device);
    }
    else{
    
    // read from device into allocated buffer
            if (!ReadFile(c2hx_device, read_data, read_bin_inf->bin_data_size, &(read_bin_inf->bin_data_size), NULL)){
                fprintf(stderr, "ReadFile from device %s failed with Win32 error code: %ld\n", "c2hx_device", GetLastError());
                 CloseHandle(c2hx_device);
            }
    }

    FILE* output;
    if (fopen_s(&output, read_bin_inf->bin_data_file, "wb") != 0) {
        fprintf(stderr, "Could not open file <%s>\n", read_bin_inf->bin_data_file);
        CloseHandle(c2hx_device);
    }

    fwrite(read_data, sizeof(BYTE), read_bin_inf->bin_data_size, output);
    fclose(output);
    //verbose_msg("%ld bytes write to file %s\n", data_size, read_bin_file);

    if (read_data) _aligned_free(read_data);
}



void h2cx_device_write(HANDLE h2cx_device, uint64_t base_address, DWORD data_size,  BYTE* write_date){

    LARGE_INTEGER addr;
    addr.QuadPart = base_address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else{
            if (!WriteFile(h2cx_device, write_date, data_size, &data_size, NULL)){
                fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","h2cx_device", GetLastError());
                CloseHandle(h2cx_device);
            }
    }
}

void c2hx_device_read (HANDLE c2hx_device, uint64_t base_address, DWORD data_size,  BYTE* read_data ){
    LARGE_INTEGER addr;
    addr.QuadPart = base_address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(c2hx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
         CloseHandle(c2hx_device);
    }
    else{
    
    // read from device into allocated buffer
            if (!ReadFile(c2hx_device, read_data, data_size, &data_size, NULL)){
                fprintf(stderr, "ReadFile from device %s failed with Win32 error code: %ld\n", "user_device_path", GetLastError());
                 CloseHandle(c2hx_device);
            }
    }
}


void CSB_Write(HANDLE user_device, int addr, uint32_t data)
{
	DWORD data_size=4;
	// BYTE  write_data[4];
	// write_data[0]=data&0xff;
	// write_data[1]=(data>>8)&0xff;
	// write_data[2]=(data>>16)&0xff;
	// write_data[3]=(data>>24)&0xff;
    
	// BYTE  write_data_memcpy[4];
	// memcpy(write_data_memcpy, &data, sizeof(int));
	//printf("write_data memcpy =%x ; write_data= %x \n", write_data_memcpy[0], write_data[0]);

    BYTE* write_data_memcpy = (BYTE*) &data;
	//user_device_write_brust(user_device, data_size,  write_data_memcpy);
    user_device_write(user_device, uint64_t(addr<<2), data_size,  write_data_memcpy);
    //  WriteFile(user_device, write_data_memcpy, data_size, &data_size, NULL);
    // // if (!WriteFile(user_device, write_data_memcpy, data_size, &data_size, NULL)){
    // //     fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
    // //     CloseHandle(user_device);
    // // }
}

int CSB_Read(HANDLE user_device, int addr)
{
	DWORD data_size=4;
	BYTE  read_data[4];
	user_device_read (user_device, uint64_t(addr<<2), data_size,  read_data);
	//int out=*read_data;
	int out=read_data[0]+(read_data[1]<<8)+(read_data[2]<<16)+(read_data[3]<<24);
	return out;
}