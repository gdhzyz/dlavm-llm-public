## Inst. Compare

### 2025-02-21

此部分主要进行了指令的实际对比，对比方法为复写CSB_Write、CSB_Read以及HANDLE，使得能够通过HANDLE记录每次进入到CSB_Write的值，最终进行对比。
对比指标为除地址相关指令外全部相同。此次对比的对象为ChatGLM模型和Qwen2模型的一个Block以及Output Layer。

xxx_2048_<date>.h为编译器生成，使用的device为hbm_accel.HBM1128（也即当前加速器release版本），其余两个为上板测试通过的指令。

然而，在实际对比中，出现了一些情况：

1. ChatGLM指令全部通过

2. Qwen2指令有关MVM算子，7，8，14有不同

 2.1 考虑不为错误，为out_ch分割方式不同, 7, 8, 14分别表示单次out_ch大小，8为last out_ch大小，14为分割次数

3. 由2，单独根据testbench为目标进行对比，编译器指令通过


因此，考虑产生上述不同的原因为手动编写指令时由于人为因素并未完全按照最优方式生成。

考虑编译器基础指令生成无误，可以使用。

#### Usage

```
g++ main.cc -I. test
./test 19 2
```

test必须接受两个命令行输入，分别为token和last_token
