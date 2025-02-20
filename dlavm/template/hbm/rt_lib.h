uint32_t init(HANDLE h2cx_device, uint64_t base_addr, uint32_t size) {
    DWORD real_size;
    BYTE* write_data = allocate_buffer(size, 0);
    memset(write_data, 0, size);
    LARGE_INTEGER addr;
    addr.QuadPart = base_addr;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else{
        if (!WriteFile(h2cx_device, write_data, size, &real_size, NULL)){
            fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
            CloseHandle(h2cx_device);
            return 0;
        }
    }
    if (write_data)	_aligned_free(write_data);
    return real_size;
}


uint32_t DDR_Write_bin(HANDLE h2cx_device, char* file_path, uint64_t address, uint32_t byte_size) {
    DWORD real_size;
    FILE* inputFile;
    if (fopen_s(&inputFile, file_path, "rb") != 0) {
        fprintf(stderr, "Could not open file <%s>\n", file_path);
        CloseHandle(h2cx_device);
    }

    BYTE* write_data = allocate_buffer(byte_size, 0);
    if (!write_data) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", byte_size, GetLastError());
        CloseHandle(h2cx_device);
    }
    fread(write_data, sizeof(BYTE), byte_size, inputFile);
    fclose(inputFile);

    if (write_data == NULL) {
        printf("Error! No valid data given!\n");
        CloseHandle(h2cx_device);
    }

    LARGE_INTEGER addr;
    addr.QuadPart = address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else {
        if (!WriteFile(h2cx_device, write_data, byte_size, &real_size, NULL)) {
            fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
            CloseHandle(h2cx_device);
        }
    }
    if (write_data)	_aligned_free(write_data);
    return real_size;
}


uint32_t DDR_Update(HANDLE h2cx_device, uint64_t address, uint32_t value) {
    DWORD real_size;
    uint32_t size = 4;
    BYTE* write_data = allocate_buffer(size, 0);
    ((uint32_t*)write_data)[0] = value;
    LARGE_INTEGER addr;
    addr.QuadPart = address;
    // set file pointer to offset of target address within PCIe BAR
    if (INVALID_SET_FILE_POINTER == SetFilePointerEx(h2cx_device, addr, NULL, FILE_BEGIN)) {
        fprintf(stderr, "Error setting file pointer, win32 error code: %ld\n", GetLastError());
        CloseHandle(h2cx_device);
    }
    else{
        if (!WriteFile(h2cx_device, write_data, size, &real_size, NULL)){
            fprintf(stderr, "WriteFile to device %s failed with Win32 error code: %d\n","user_device_path", GetLastError());
            CloseHandle(h2cx_device);
            return 0;
        }
    }
    if (write_data)	_aligned_free(write_data);
    return real_size;
}


uint8_t* DDR_Read(HANDLE c2hx_device, uint64_t address, uint32_t data_size) {
    DWORD bin_data_size = data_size;
    BYTE* read_data = allocate_buffer(data_size, 0);
    if (!read_data) {
        fprintf(stderr, "Error allocating %ld bytes of memory, error code: %ld\n", bin_data_size, GetLastError());
         CloseHandle(c2hx_device);
    }
    memset(read_data, 0, bin_data_size);
    
    LARGE_INTEGER addr;
    addr.QuadPart = address;
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
    return (uint8_t*)read_data;
}