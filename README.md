# DLAVM-LLM

DeepLearning Accelerator LLM Compiler，本项目为大模型编译器，参考TVM开发。

## Usage

项目基于Python开发，无复杂依赖，只需编译一个动态库即可（非必要，且仅Linux下需要），操作如下：

```shell
cd dlavm/clib
make
```

项目源码以python-package的形式开发，命名为dlavm。由于暂未完成，并未采用setup.py等方式加载，需要添加环境变量或者使用软链接将dlavm放置在script中方便跳转查看。

项目入口在script文件夹下，主要有两个main入口，分别对应两种代码生成方式，具体情况如下：

1. main.py

此入口为最新版本的代码生成方案，其特点为每个算子由lower IR编写tasks进行编译生成，在代码中体现为使用dlavm.backend进行编译生成。

然而，此项目并未经过严格测试，生成情况需要测试过后才能确定，并且特定的debug方案以及可视化暂未支持，待添加。

文档可参照doc/DLAVM_Compiler.md进行学习和开发。

2. codegen_main.py

此入口为之前代码生成方案，由于其生成方式比较死板，难以进行for循环等具有Region的代码的生成，所以暂时被弃用。

由于此不支持for代码的生成，所以其驱动代码（tasks）或许较旧，不一定可靠，但是其额外功能较为全面，可以生成prototxt模型结构图，以供参照。

文档可参照doc/Old_Compiler.md

---

