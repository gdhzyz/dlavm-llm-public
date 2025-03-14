# Testbench Simulator

本项目为testbench中加速器指令的仿真生成模块，可接入编译器中作为指令生成的后端进行代码生成任务。

## Usage

本项目的使用较为简单，对于新的testbench模块（需具有宏定义头文件，以.vh后缀结尾），可以通过脚本自动进行分类、复制并处理为无需rtl即可仿真的testbench模块，并可打印加速器的驱动指令，以进行手动编写或接入编译器中。

实际操作过程以及处理内容如下：

```shell
source /tools/Xilinx/Vivado/2023.2/settings64.sh
make TESTBENCH=<path to testbench>
```

上述命令即可完成对testbench文件中所有内容的处理，首先复制所有的tasks.vh到workspace_xxxx(日期/名字)，并修改basic_tasks.vh中的AXI_Write/Read函数为所需；
复制所有的testbench.sv到temp目录下。其次，将testbench.sv中所有有关rtl实例化的模块去除，仅保留tasks中的驱动函数的调用；
最后，删除无用的temp文件夹，并将vivado.Makefile通过软链接加载到workspace中，使其可以通过make命令对testbench进行编译。

以上，此项目的准备工作即完成。然而，由于tasks中的API变化的原因，需要即时同步函数名到bin/process文件中。

其次，可以进入workspace中。使用make命令进行编译测试：

```shell
make TOP_MODULE=<testbench_HBM_MVM> run
```

## 编译器接入

通常，此项目目录不会直接使用，而是加载进编译器中进行自动化调用，可参考"dlavm/driver/nn/ohbm/testbench.py"

首先，需要将testbench中的宏定义名与编译器中的参数进行绑定，即可通过python中已完善的代码自动加载workspace并运行获得结果。

其次，为了能使python能够了解哪个为需要加载的workspace，需要在python的main函数中写入workspace的加载路径，如下所示：

```python
from dlavm.driver import config
config.tb_sim_path = "/home/shenao/dlavm-llm-public/tbsim/workspace_2025_0226"

```

以上参考script/only_hbm/main_test_nn.py即可。
