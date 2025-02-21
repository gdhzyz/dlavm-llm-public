## Usage

当前仅维护main.py及其相关功能，所以在此仅介绍main.py的一些指令。

查看所有参数：

```shell
python3 main.py --help
```

具体内容可参考main.py，其中，py命令不可用。

通常来说，所需关注的参数主要为：model, wt2hbm, debug, aux, lite, prototxt, prefix, save, maxtoken

+ model: 模型名，当前已完成qwen2，llama，chatglm

+ wt2hbm: 权重加载方式，默认为False，即pcie2hbm

+ debug: 将模型的block size改为1，以加快编译速度，减少生成的内容

+ aux: 硬件计算方式，默认为False, 通过regs进行

+ lite: 优化方法，默认为False，开启后将删除所有的for循环，通常配合aux模式使用

+ prototxt: 生成prototxt模型，默认为False，可以通过netron进行可视化操作，查看硬件算子图

+ prefix: 编译器中权重的地址，默认为Block_write_data

+ save: 模型生成的路径，默认为../output

+ maxtoken: kvcache最大的空间，默认为所使用加速器的MAXTOKEN

举例，常规编译qwen2:

```shell
python3 main.py --model qwen2
```

aux模式编译llama并进行lite优化:

```shell
python3 main.py --model llama --aux --lite
```

debug模式下生成chatglm的可视化模型:

```shell
python3 main.py --model chatglm --debug --prototxt
```
