uint64_t runtime0 =   6227968; //A   6227968 storage size:  16777216 B-> ADDR_A storage size: 122,159,104 B-> 1024*1024*2*32=67,108,864 B 
uint64_t runtime1 =  73336832; //B  23005184 storage size:  56098816 B-> ADDR_B storage size: 122,159,104 B-> 1024*1024*2*32=67,108,864 B
uint64_t runtime2 = 140445696; //C  79104000 storage size: 268435456 B-> ADDR_C storage size: 122,159,104 B-> 1024*1024*2*32=67,108,864 B
uint64_t runtime3 = 207554560; //D 347539456 storage size: 268435456 B-> ADDR_D storage size: 122,159,104 B-> 1024*1024*2*32=67,108,864 B
uint64_t runtime4 = 274663424; //E 615974912 storage size:   1048576 B-> ADDR_E storage size: 122,159,104 B-> 1024*1024*2*32=67,108,864 B
uint64_t runtime5 = 341772288; //F

uint64_t ADDR_A   = runtime0;
uint64_t ADDR_B   = runtime1;
uint64_t ADDR_C   = runtime2;
uint64_t ADDR_D   = runtime3;
uint64_t ADDR_E   = runtime4;
uint64_t ADDR_F   = runtime5;

void step1 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.layer_norm accel operator node, storage data in runtime1 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, 532480);
  CSB_Write(device, 195, ADDR_A);
  CSB_Write(device, 196, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_B);
  CSB_Write(device, 199, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, 128);
  CSB_Write(device, 202, 1);
  CSB_Write(device, 203, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 204, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 205, 155648);
  CSB_Write(device, 206, 8);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 32);
#ifdef PRINT_STEP
printf("start: step1!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.layer_norm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step2 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 1024);
    CSB_Write(device, 8, 1024);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_B + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 0);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_C + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 3);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 548864);
    CSB_Write(device, 27, 0);
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 30, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 31, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 32, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 33, 799);
#ifdef PRINT_STEP
printf("start: step2!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step3 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.pos_emb accel operator node, storage data in runtime3 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, (kvcache ? ((token - 1) * 64) : (last_token * 64)));
  CSB_Write(device, 195, ADDR_C);
  CSB_Write(device, 196, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_D);
  CSB_Write(device, 199, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, 4);
  CSB_Write(device, 202, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 203, last_token);
  CSB_Write(device, 204, 32);
  CSB_Write(device, 205, 262144);
  CSB_Write(device, 206, 131072);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 4);
#ifdef PRINT_STEP
printf("start: step3!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.pos_emb run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step4 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 256);
    CSB_Write(device, 8, 256);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_B + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 270336);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_C + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 0);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 565248);
    CSB_Write(device, 27, 0);
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 30, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 31, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 32, (64 * (kvcache ? 1 : (token - last_token))));
    CSB_Write(device, 33, 799);
#ifdef PRINT_STEP
printf("start: step4!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step5 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.pos_emb accel operator node, storage data in runtime4 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, (kvcache ? ((token - 1) * 64) : (last_token * 64)));
  CSB_Write(device, 195, ADDR_C);
  CSB_Write(device, 196, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_E);
  CSB_Write(device, 199, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, 4);
  CSB_Write(device, 202, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 203, last_token);
  CSB_Write(device, 204, 2);
  CSB_Write(device, 205, 262144);
  CSB_Write(device, 206, 131072);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 4);
#ifdef PRINT_STEP
printf("start: step5!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.pos_emb run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step6 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.dat2hbm accel operator node, storage data in hbm_cache0 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 195, ADDR_E);
  CSB_Write(device, 196, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, 96343040);
  CSB_Write(device, 199, 524288);
  CSB_Write(device, 200, 131072);
  CSB_Write(device, 201, 28);
  CSB_Write(device, 202, last_token);
  CSB_Write(device, 203, (token - last_token));
  CSB_Write(device, 204, 2);
  CSB_Write(device, 205, 4);
  CSB_Write(device, 206, 1);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 64);
#ifdef PRINT_STEP
printf("start: step6!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.dat2hbm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step7 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.trp_mvm accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = ((((seq + 128) - 1) / 128) - 1);
  int out_w_slice_last = (seq - (((((seq + 128) - 1) / 128) - 1) * 128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 194, 96343040);
    CSB_Write(device, 195, (ADDR_D + ((out_w * 128) * 64)));
    CSB_Write(device, 196, ((4 * seq) * 64));
    CSB_Write(device, 197, (seq * 64));
    CSB_Write(device, 198, (ADDR_C + ((out_w * 128) * 64)));
    CSB_Write(device, 199, ((((((seq + last_token) + 32) - 1) / 32) * seq) * 64));
    CSB_Write(device, 200, (seq * 64));
    CSB_Write(device, 201, 524288);
    CSB_Write(device, 202, 131072);
    CSB_Write(device, 203, 128);
    CSB_Write(device, 204, token);
    CSB_Write(device, 205, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 206, 29584705);
    CSB_Write(device, 207, 991234);
    CSB_Write(device, 208, 0);
    CSB_Write(device, 209, 2);
#ifdef PRINT_STEP
printf("start: step7!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.trp_mvm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step8 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.softmax accel operator node, storage data in runtime3 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, (1 - kvcache));
  CSB_Write(device, 195, ADDR_C);
  CSB_Write(device, 196, ((64 * (kvcache ? 1 : (token - last_token))) * (((token + 32) - 1) / 32)));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_D);
  CSB_Write(device, 199, ((64 * (kvcache ? 1 : (token - last_token))) * (((token + 32) - 1) / 32)));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, (((token + 32) - 1) / 32));
  CSB_Write(device, 202, 32);
  CSB_Write(device, 203, token);
  CSB_Write(device, 204, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 205, (token - last_token));
  CSB_Write(device, 206, last_token);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 8);
#ifdef PRINT_STEP
printf("start: step8!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.softmax run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step9 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn accel operator node, storage data in runtime4 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 256);
    CSB_Write(device, 8, 256);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_B + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 287232);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_E + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 0);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 566272);
    CSB_Write(device, 27, 0);
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (seq * 64));
    CSB_Write(device, 30, (seq * 64));
    CSB_Write(device, 31, (seq * 64));
    CSB_Write(device, 32, (seq * 64));
    CSB_Write(device, 33, 799);
#ifdef PRINT_STEP
printf("start: step9!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step10 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.dat2hbm accel operator node, storage data in hbm_cache1 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 195, ADDR_E);
  CSB_Write(device, 196, ((64 * (kvcache ? 1 : (token - last_token))) * 4));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, 96375808);
  CSB_Write(device, 199, 524288);
  CSB_Write(device, 200, 131072);
  CSB_Write(device, 201, 28);
  CSB_Write(device, 202, last_token);
  CSB_Write(device, 203, (token - last_token));
  CSB_Write(device, 204, 2);
  CSB_Write(device, 205, 4);
  CSB_Write(device, 206, 0);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 64);
#ifdef PRINT_STEP
printf("start: step10!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.dat2hbm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step11 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.f2w_mvm accel operator node, storage data in runtime1 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int out_w_per_slice = (4194304 / (((((((seq + last_token) + 32) - 1) / 32) * 32) * 16) * 16));
  int wout_split_times_minus1 = ((((seq + out_w_per_slice) - 1) / out_w_per_slice) - 1);
  int out_w_slice_last = (seq - (((((seq + out_w_per_slice) - 1) / out_w_per_slice) - 1) * out_w_per_slice));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 194, 96375808);
    CSB_Write(device, 195, (ADDR_D + ((out_w * out_w_per_slice) * 64)));
    CSB_Write(device, 196, ((((((seq + last_token) + 32) - 1) / 32) * seq) * 64));
    CSB_Write(device, 197, (seq * 64));
    CSB_Write(device, 198, (ADDR_B + ((out_w * out_w_per_slice) * 64)));
    CSB_Write(device, 199, ((4 * seq) * 64));
    CSB_Write(device, 200, (seq * 64));
    CSB_Write(device, 201, 524288);
    CSB_Write(device, 202, 131072);
    CSB_Write(device, 203, 128);
    CSB_Write(device, 204, token);
    CSB_Write(device, 205, ((out_w < wout_split_times_minus1) ? out_w_per_slice : out_w_slice_last));
    CSB_Write(device, 206, 29614080);
    CSB_Write(device, 207, 991234);
    CSB_Write(device, 208, 0);
    CSB_Write(device, 209, 1);
#ifdef PRINT_STEP
printf("start: step11!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.f2w_mvm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step12 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn_res accel operator node, storage data in runtime3 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 1024);
    CSB_Write(device, 8, 1024);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_B + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 304128);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_D + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 3);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 567296);
    CSB_Write(device, 27, ADDR_A + ((out_w * 128) * 64));
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (seq * 64));
    CSB_Write(device, 30, (seq * 64));
    CSB_Write(device, 31, (seq * 64));
    CSB_Write(device, 32, (seq * 64));
    CSB_Write(device, 33, 1823);
#ifdef PRINT_STEP
printf("start: step12!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn_res run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step13 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.layer_norm accel operator node, storage data in runtime0 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, 583680);
  CSB_Write(device, 195, ADDR_D);
  CSB_Write(device, 196, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_A);
  CSB_Write(device, 199, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, 128);
  CSB_Write(device, 202, 1);
  CSB_Write(device, 203, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 204, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 205, 155648);
  CSB_Write(device, 206, 8);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 32);
#ifdef PRINT_STEP
printf("start: step13!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.layer_norm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step14 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 1024);
    CSB_Write(device, 8, 384);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_A + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 1478400);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_C + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 13);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 654848);
    CSB_Write(device, 27, 0);
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (seq * 64));
    CSB_Write(device, 30, (seq * 64));
    CSB_Write(device, 31, (seq * 64));
    CSB_Write(device, 32, (seq * 64));
    CSB_Write(device, 33, 799);
#ifdef PRINT_STEP
printf("start: step14!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step15 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.activate accel operator node, storage data in runtime1 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, 524288);
  CSB_Write(device, 195, ADDR_C);
  CSB_Write(device, 196, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 197, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 198, ADDR_B);
  CSB_Write(device, 199, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 200, (64 * (kvcache ? 1 : (token - last_token))));
  CSB_Write(device, 201, 428);
  CSB_Write(device, 202, 1);
  CSB_Write(device, 203, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 204, (kvcache ? 1 : (token - last_token)));
  CSB_Write(device, 205, 428);
  CSB_Write(device, 206, 13696);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 16);
#ifdef PRINT_STEP
printf("start: step15!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.activate run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step16 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn_res accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = (seq+128-1)/128-1;
  int out_w_slice_last = (seq-(wout_split_times_minus1*128));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 4096);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 1024);
    CSB_Write(device, 8, 384);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 128 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_A + ((out_w * 128) * 64)));
    CSB_Write(device, 11, 574464);
    CSB_Write(device, 12, 16896);
    CSB_Write(device, 13, (ADDR_C + ((out_w * 128) * 64)));
    CSB_Write(device, 14, 13);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 2);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 2048);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 600064);
    CSB_Write(device, 27, ADDR_B + ((out_w * 128) * 64));
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (seq * 64));
    CSB_Write(device, 30, (seq * 64));
    CSB_Write(device, 31, (seq * 64));
    CSB_Write(device, 32, (seq * 64));
    CSB_Write(device, 33, 1823);
#ifdef PRINT_STEP
printf("start: step16!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn_res run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step17 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.mvm_bn_res accel operator node, storage data in runtime0 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  int seq=token-last_token;
  int wout_split_times_minus1 = ((((seq + 38) - 1) / 38) - 1);
  int out_w_slice_last = (seq - (((((seq + 38) - 1) / 38) - 1) * 38));
  for (int out_w = 0; out_w < (wout_split_times_minus1 + 1); out_w += 1) {
    CSB_Write(device, 2, 13696);
    CSB_Write(device, 3, ((out_w < wout_split_times_minus1) ? 38 : out_w_slice_last));
    CSB_Write(device, 4, 1);
    CSB_Write(device, 5, ((out_w < wout_split_times_minus1) ? 38 : out_w_slice_last));
    CSB_Write(device, 6, 1);
    CSB_Write(device, 7, 288);
    CSB_Write(device, 8, 64);
    CSB_Write(device, 9, ((out_w < wout_split_times_minus1) ? 38 : out_w_slice_last));
    CSB_Write(device, 10, (ADDR_C + ((out_w * 38) * 64)));
    CSB_Write(device, 11, 2382336);
    CSB_Write(device, 12, 56576);
    CSB_Write(device, 13, (ADDR_A + ((out_w * 38) * 64)));
    CSB_Write(device, 14, 14);
    CSB_Write(device, 15, 28);
    CSB_Write(device, 16, 0);
    CSB_Write(device, 17, 0);
    CSB_Write(device, 18, 0);
    CSB_Write(device, 19, 0);
    CSB_Write(device, 20, 0);
    CSB_Write(device, 21, 0);
    CSB_Write(device, 22, 5769216);
    CSB_Write(device, 23, 3670144);
    CSB_Write(device, 24, 1408);
    CSB_Write(device, 25, 0);
    CSB_Write(device, 26, 709632);
    CSB_Write(device, 27, ADDR_D + ((out_w * 38) * 64));
    CSB_Write(device, 28, 0);
    CSB_Write(device, 29, (seq * 64));
    CSB_Write(device, 30, (seq * 64));
    CSB_Write(device, 31, (seq * 64));
    CSB_Write(device, 32, (seq * 64));
    CSB_Write(device, 33, 1823);
#ifdef PRINT_STEP
printf("start: step17!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
  }
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn_res run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step477 (HANDLE& device, int kvcache, int token, int last_token) {
// accel.hbm.layer_norm accel operator node, storage data in runtime1 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 194, 5951488);
  CSB_Write(device, 195, (ADDR_A + (((token - last_token) - 1) * 64)) );
  CSB_Write(device, 196, (64 * (token - last_token)));
  CSB_Write(device, 197, (64 * (token - last_token)));
  CSB_Write(device, 198, ADDR_E);
  CSB_Write(device, 199, 64);
  CSB_Write(device, 200, 64);
  CSB_Write(device, 201, 128);
  CSB_Write(device, 202, 1);
  CSB_Write(device, 203, 1);
  CSB_Write(device, 204, 1);
  CSB_Write(device, 205, 155648);
  CSB_Write(device, 206, 8);
  CSB_Write(device, 207, 0);
  CSB_Write(device, 208, 0);
  CSB_Write(device, 209, 32);
#ifdef PRINT_STEP
printf("start: step477!\n");
#endif
  while(CSB_Read(device, 193) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.layer_norm run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void step478 (HANDLE& device) {
// accel.hbm.mvm_bn accel operator node, storage data in runtime2 with 0 offset
#ifdef REGS_DEBUG
LARGE_INTEGER start_run;
LARGE_INTEGER stop_run;
LARGE_INTEGER freq;
double time_sec0;
QueryPerformanceFrequency(&freq);
QueryPerformanceCounter(&start_run);
for (int i = 0; i < 1000; i=i+1) {
#endif
  CSB_Write(device, 2, 4096);
  CSB_Write(device, 3, 1);
  CSB_Write(device, 4, 1);
  CSB_Write(device, 5, 1);
  CSB_Write(device, 6, 1);
  CSB_Write(device, 7, 1024);
  CSB_Write(device, 8, 512);
  CSB_Write(device, 9, 1);
  CSB_Write(device, 10, ADDR_E);
  CSB_Write(device, 11, 92051456);
  CSB_Write(device, 12, 16896);
  CSB_Write(device, 13, 79104000);
  CSB_Write(device, 14, 63);
  CSB_Write(device, 15, 28);
  CSB_Write(device, 16, 0);
  CSB_Write(device, 17, 615974912);
  CSB_Write(device, 18, 0);
  CSB_Write(device, 19, 0);
  CSB_Write(device, 20, 0);
  CSB_Write(device, 21, 0);
  CSB_Write(device, 22, 5769216);
  CSB_Write(device, 23, 3670144);
  CSB_Write(device, 24, 2048);
  CSB_Write(device, 25, 0);
  CSB_Write(device, 26, 5967872);
  CSB_Write(device, 27, 0);
  CSB_Write(device, 28, 0);
  CSB_Write(device, 29, 64);
  CSB_Write(device, 30, 64);
  CSB_Write(device, 31, 64);
  CSB_Write(device, 32, 64);
  CSB_Write(device, 33, 2847);
#ifdef PRINT_STEP
printf("start: step478!\n");
#endif
  while(CSB_Read(device, 1) != 1) {}
#ifdef REGS_DEBUG
}
QueryPerformanceCounter(&stop_run);
time_sec0 = (unsigned long long)(stop_run.QuadPart - start_run.QuadPart) / (double)freq.QuadPart;
printf("accel.hbm.mvm_bn run time     = %fs(1000 times), %fs(1 times) \n",time_sec0, time_sec0/1000);
#endif
}

void glm0912_2048_lite_wt2hbm_0924_1901(HANDLE& device, int token, int kvcache, int last_token) {
  step1(device, kvcache, token, last_token);
  step2(device, kvcache, token, last_token);
  step3(device, kvcache, token, last_token);
  step4(device, kvcache, token, last_token);
  step5(device, kvcache, token, last_token);
  step6(device, kvcache, token, last_token);
  step7(device, kvcache, token, last_token);
  step8(device, kvcache, token, last_token);
  step9(device, kvcache, token, last_token);
  step10(device, kvcache, token, last_token);
  step11(device, kvcache, token, last_token);
  step12(device, kvcache, token, last_token);
  step13(device, kvcache, token, last_token);
  step14(device, kvcache, token, last_token);
  step15(device, kvcache, token, last_token);
  step16(device, kvcache, token, last_token);
  step17(device, kvcache, token, last_token);
  step477(device, kvcache, token, last_token);
  step478(device);
}
