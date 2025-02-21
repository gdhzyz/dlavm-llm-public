#include <iostream>
#include <vector>
#include <set>
#include <utility>

typedef std::vector<std::pair<int, int>> HANDLE;

void CSB_Write(HANDLE& dev, int addr, int data) {
  dev.push_back(std::make_pair(addr, data));
}

int CSB_Read(HANDLE& dev, int addr) {
  return 1;
}

namespace target {
#include "chatglm_2048_0221_0425.h"
}
namespace target1 {
#include "qwen2_2048_0221_0738.h"
}
namespace golden {
#include "glm1128_MAXTOKEN2048_wt2hbm_lite.h"
}
namespace golden1 {
#include "qwen2_2048_lite_wt2hbm_for_1128_bitstream_run_new_addr_new_cfg_final.h"
}

std::set<int> ignore = {194, 195, 198, 10, 11, 13, 17, 26, 27};

int chatglm(int token, int last_token){
  std::cout << "ChatGLM Start!" << std::endl;
  HANDLE target;
  HANDLE golden;
  int seq, kvcache;
  // ignore address
  seq = token - last_token;
  kvcache = (seq == 1) ? 1 : 0;
  target::chatglm_2048_0221_0425(target, seq, last_token);
  golden::glm0912_2048_lite_wt2hbm_0924_1901(golden, token, kvcache, last_token);
  if (target.size() != golden.size()) {
    std::cerr << "Error! Address size not match!" << target.size() << " " << golden.size() << std::endl;
  }
    
  for (uint64_t i = 0; i < target.size(); i++) {
    int addr_1 = target[i].first, data_1 = target[i].second;
    int addr_2 = golden[i].first, data_2 = golden[i].second;
    if (addr_1 != addr_2) {
      std::cerr << "Error! Address not match!" << addr_2 << ", " << data_2 << " : " << addr_1 << ", " << data_1 << std::endl;
      return -1;
    }
    if (ignore.find(addr_1) != ignore.end())
      continue;
    if (data_1 != data_2) {
      std::cerr << "Error! " << i << " " << addr_1 << ": golden=" << data_2 << ", target=" << data_1 << std::endl;
    }
  }
  std::cout << "ChatGLM Finish! Checked " << target.size() << std::endl;
  return 0;
}

int qwen2(int token, int last_token){
  std::cout << "Qwen2 Start!" << std::endl;
  HANDLE target;
  HANDLE golden;
  int seq, kvcache;
  // ignore address
  seq = token - last_token;
  kvcache = (seq == 1) ? 1 : 0;
  target1::qwen2_2048_0221_0738(target, seq, last_token);
  golden1::qwen2_2048_lite_wt2hbm_0927_1041(golden, token, kvcache, last_token);
  if (target.size() != golden.size()) {
    std::cerr << "Error! Address size not match!" << target.size() << " " << golden.size() << std::endl;
  }
    
  for (uint64_t i = 0; i < target.size(); i++) {
    int addr_1 = target[i].first, data_1 = target[i].second;
    int addr_2 = golden[i].first, data_2 = golden[i].second;
    if (addr_1 != addr_2) {
      std::cerr << "Error! Address not match!" << addr_2 << ", " << data_2 << " : " << addr_1 << ", " << data_1 << std::endl;
      return -1;
    }
    if (ignore.find(addr_1) != ignore.end())
      continue;
    if (data_1 != data_2) {
      std::cerr << "Error! " << i << " " << addr_1 << ": golden=" << data_2 << ", target=" << data_1 << std::endl;
    }
  }
  std::cout << "Qwen2 Finish! Checked " << target.size() << std::endl;
  return 0;
}

int main(int argc, char** argv){
  int token = atoi(argv[1]), last_token = atoi(argv[2]);
  chatglm(token, last_token);
  qwen2(token, last_token);
  return 0;
}
