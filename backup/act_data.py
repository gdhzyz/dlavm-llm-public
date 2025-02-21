import numpy as np
import pwlf
import matplotlib.pyplot as plt
import struct
import time
import pickle
import os

def Real2FP16(data_real):
    hexa_all = np.zeros((0,))
    float_all = np.zeros((0,), dtype=np.float16)
    
    for data_real in data_real:
        if(abs(data_real) <= 65504):
            data_real = float(data_real)
            hexa = struct.unpack('H',struct.pack('e',data_real))[0]
            hexa = hex(hexa)
            hexa = hexa[2:]
        elif(data_real < 0):
            hexa = 'fbff'
        else: 
            hexa = '7bff'
        y = struct.pack("H",int(hexa,16))
        float_i = np.frombuffer(y, dtype =np.float16)[0]
        hexa_all = np.append(hexa_all, hexa)
        float_all = np.append(float_all, float_i)
    return hexa_all, float_all


def get_act_data(name, func, x_min, x_max, hfile_path='approx_pwlf_act.h', pyfile_path='approx_pwlf_act.py'):
    print(f"输出变量名：{name}，激活函数名：{func.__name__}，X min：{x_min}，X max：{x_max}， 保存.h文件名：{hfile_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start = time.perf_counter()
    points_num = 200 #拟合所需要的点数
    segCnt = 15 #分段个数

    # x and y
    x_array = np.linspace(x_min, x_max, points_num)
    y_array = func(x_array)
    # y_array = gelu_1(x_array) 

    # initialize piecewise linear fit with your x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x_array, y_array)

    # fit the data
    res = my_pwlf.fit(segCnt) #拟合

    # predict for the determined points
    xHat = np.linspace(min(x_array), max(x_array), num=100)
    yHat = my_pwlf.predict(xHat) #预测

    # get information of prediction
    slopes = my_pwlf.calc_slopes()
    slopes_fp16, slopes_fp16_r = Real2FP16(slopes) #k

    b=my_pwlf.intercepts
    b_fp16, b_fp16_r = Real2FP16(b) #b

    bp = res
    bp_fp16, bp_fp16_r = Real2FP16(bp) #break points

    se = my_pwlf.standard_errors()
    pre_var = np.sum(my_pwlf.prediction_variance(xHat)**2)/len(xHat) #MSE
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)
    # save
    with open(os.path.join(save_path, name + '.pkl'), 'wb') as f:
        pickle.dump(my_pwlf, f, pickle.HIGHEST_PROTOCOL)

    ############################print--------------change output file name
    plt.figure()
    plt.plot(x_array, y_array, 'o')
    plt.plot(xHat, yHat, '-')
    plt.plot(bp, func(bp), 'o', color = 'r')
    plt.savefig(os.path.join(save_path, f"approx_pwlf_{name}.png"))
    slopes_fp16_r = np.pad(slopes_fp16_r, (1, 0), 'constant', constant_values=0)
    b_fp16_r = np.pad(b_fp16_r, (1, 0), 'constant', constant_values=0)
    with open(os.path.join(save_path, f'approx_pwlf_{name}.bin'),'wb') as file0:
        source = bp_fp16_r.tobytes() + slopes_fp16_r.tobytes() + b_fp16_r.tobytes()
        print(bp_fp16_r.shape, bp_fp16_r.dtype)
        print(slopes_fp16_r.shape, slopes_fp16_r.dtype)
        print(b_fp16_r.shape, b_fp16_r.dtype)
        file0.write(source)
    with open(os.path.join(save_path, f'approx_pwlf_{name}.txt'),'w') as file0:
        print('MSE = %g' %pre_var, file=file0) #MSE
        print('-----------------Region-----------------', file=file0)
        # print(bp.tolist()[::-1], file=file0)
        # print('{16\'h' + ', 16\'h'.join(bp_fp16.tolist()) + '}', file=file0)
        print(bp_fp16_r.tolist(), file=file0)

        print('-----------------k-----------------', file=file0)
        # print(slopes.tolist()[::-1], file=file0)
        # print('{16\'h' + ', 16\'h'.join(slopes_fp16.tolist()) + '}', file=file0)
        print(slopes_fp16_r.tolist(), file=file0)

        print('-----------------b-----------------', file=file0)
        # print(b.tolist()[::-1], file=file0)
        # print('{16\'h' + ', 16\'h'.join(b_fp16.tolist()) + '}', file=file0)
        print(b_fp16_r.tolist(), file=file0)
        print('----------------cpp----------------', file=file0)
    with open(hfile_path, 'a') as file0:
        print(f"// activation: {func.__name__}, x_min: {x_min}, x_max: {x_max}", file=file0)
        print(f'unsigned short {name}_wt[16] = ' + '{0x' + ', 0x'.join(slopes_fp16.tolist()) + '};', file=file0)
        print(f'unsigned short {name}_bias[16] = ' + '{0x' + ', 0x'.join(b_fp16.tolist()) + '};', file=file0)
        print(f'unsigned short {name}_x_region[16] = ' + '{0x' + ', 0x'.join(bp_fp16.tolist()) + '};', file=file0)
    with open(pyfile_path, 'a') as file0:
        print(f"# activation: {func.__name__}, x_min: {x_min}, x_max: {x_max}", file=file0)
        print(f'{name}_wt = ', slopes.tolist(), file=file0)
        print(f'{name}_bias = ', b.tolist(), file=file0)
        print(f'{name}_x_region = ', bp.tolist()[1:], file=file0)


def gelu(x):
    return 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)


if __name__ == "__main__":
    # 保存txt和png的路径, .h文件保存路径在func_args中设置
    save_path = "saved"
    func_args = [
#       [   str_act_name,   activation,     x_min,  x_max, hfile_path='approx_pwlf_act.h', pyfile_path='approx_pwlf_act.py']
        [          "exp",       np.exp,        -5,      0],
        #[         "gelu",         gelu,        -4,      4],
        #[         "tanh",      np.tanh,        -4,      4],
        #[      "sigmoid",      sigmoid,        -4,      4],
        #[         "silu",        swish,        -12,      10],
    ]

    for args in func_args:
        get_act_data(*args)
