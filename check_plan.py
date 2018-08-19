import time
import math
import copy
import csv
import xlrd
import numpy as np
from functools import wraps


def log(info=None):
    print("\n[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]\033[1;35m log \033[0m: ", info, "\n")


def func_timer(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 开始执行...]".format(name=func.__name__))
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 执行完毕，耗时: {time:.2f}s]".format(name=func.__name__, time=t1 - t0))
        return result

    return function_timer


def sig(x):
    return 1 if x >= 0 else 0


def fan(x):  # 0-1 取反，矩阵就是用全1的减去就是反
    return 0 if x else 1


def read_data(path):
    data = list()
    wb = xlrd.open_workbook(path)
    sheet = wb.sheets()[0]

    for i in range(NUM):
        row = i + 2
        name = str(sheet.cell_value(row, 0))
        lng = float(sheet.cell_value(row, 1))
        lat = float(sheet.cell_value(row, 2))
        kind_name = str(sheet.cell_value(row, 3))
        kind = 1 if kind_name == "Butterfly Site" else 0
        data.append({"lat": lat, "lng": lng, "site_name": name,
                     "kind": kind, "preset": False})  # random.randint(0, 1)
    return data


def read_csv(path):
    posi = []
    csv_file = csv.reader(open(path, 'r'))
    for row in csv_file:
        posi.append(row)
    return posi


def distance(lng1, lat1, lng2, lat2):
    lat1 = (math.pi / 180.0) * lat1
    lat2 = (math.pi / 180.0) * lat2
    lng1 = (math.pi / 180.0) * lng1
    lng2 = (math.pi / 180.0) * lng2
    temp = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
    shit = np.where(temp > 1.0)  # 计算结果异常的点，由于精度损失可能会大于1
    temp[shit] = 1.0
    dis = np.arccos(temp) * 6378.0
    dis[(np.arange(0, NUM), np.arange(0, NUM))] = 0.0  # 由于精度损失，要设置对角线上全部为0
    return dis


@func_timer
def check_constraint():
    # 约束1所用辅助量
    cdr = CONNECT * DR  # 两矩阵对应位置元素相乘
    crr = CONNECT * RR
    cdr_col_sum = np.sum(cdr, axis=0)  # axis为0，每一列元素相加，将矩阵压缩为一行
    crr_col_sum = np.sum(crr, axis=0)
    # 约束1 choose[i] == 1 时所用
    lk_1 = crr_col_sum + ONES  # 此处应用了numpy数组的广播
    kj_1 = np.dot(lk_1, crr) + ONES
    ji_1 = np.dot(kj_1, cdr)
    # 约束1 choose[i] == 0 时所用
    cdr_row_sum = np.sum(cdr, axis=1)  # axis为1，每一行元素相加，将矩阵压缩为一列
    jk_2 = np.sum(CONNECT * (DR + RR * cdr_row_sum), axis=1)
    ij_2 = np.sum(CONNECT * (DR + RR * jk_2), axis=1)
    # 约束2所用辅助量
    c_dis = CONNECT * DIS
    cd_dr = c_dis * DR
    cd_dd = c_dis * DD
    cd_rr = c_dis * RR

    for i in range(0, NUM):  # 约束1 建站
        if CHOOSE[i]:
            if data[i]["kind"]:
                cons1 = sig(8 - cdr_col_sum[i]) * sig(12 - ji_1[i])
            else:
                cons1 = sig(4 - cdr_col_sum[i]) * sig(6 - ji_1[i])
        else:
            cons1 = ij_2[i]
        if cons1:
            continue
        else:
            info = "***cons1 failed. i=" + str(i)
            log(info)

    # 约束2 距离
    if len(np.where(cd_dr > 20.0)[0]):
        info = "***cd_dr failed"
        log(info)
    elif len(np.where(cd_dd > 50.0)[0]):
        info = "***cd_dd failed"
        log(info)
    elif len(np.where(cd_rr > 10.0)[0]):
        info = "***cd_rr failed"
        log(info)
    else:
        pass

    info = "Check Success!"
    log(info)


def convert2csu():
    global DIS
    lat1 = np.zeros([NUM, NUM], dtype=float)
    lat2 = np.zeros([NUM, NUM], dtype=float)
    lng1 = np.zeros([NUM, NUM], dtype=float)
    lng2 = np.zeros([NUM, NUM], dtype=float)

    for i in range(0, NUM):
        lat1[i] = data[i]["lat"]
        lat2[:, i] = data[i]["lat"]
        lng1[i] = data[i]["lng"]
        lng2[:, i] = data[i]["lng"]

    DIS = distance(lng1, lat1, lng2, lat2)

    for i in range(NUM):
        if _posi[i][1] == '1':
            CHOOSE[i] = 1
        for j in range(NUM):
            if _graph[i + 1][j + 1] == '1':
                CONNECT[i, j] = 1

    for i in range(0, NUM):
        dis0_50 = list(map(lambda x: int(x), set(np.where(DIS[i] <= 50)[0])))
        for j in dis0_50:
            DD[i][j] = CHOOSE[i] * CHOOSE[j]
        dis0_10 = list(map(lambda x: int(x), set(np.where(DIS[i] <= 10)[0])))
        for j in dis0_10:
            RR[i][j] = fan(CHOOSE[i]) * fan(CHOOSE[j])
        dis0_20 = list(map(lambda x: int(x), set(np.where(DIS[i] <= 20)[0])))
        for j in dis0_20:
            DR[i][j] = fan(DD[i][j]) * fan(RR[i][j])

    return


if __name__ == '__main__':
    NUM = 1000
    data = read_data("B题测试数据-更新.xlsx")
    ONES = np.ones(NUM, int)
    CHOOSE = np.zeros(NUM, int)
    CONNECT = np.zeros([NUM, NUM], int)
    DD = np.zeros([NUM, NUM], int)
    DR = np.zeros([NUM, NUM], int)
    RR = np.zeros([NUM, NUM], int)
    DIS = np.zeros([NUM, NUM], float)
    _posi = read_csv('Posi.csv')
    _graph = read_csv('Graph.csv')

    convert2csu()
    check_constraint()

'''
三个文件名可以输入相对路径或者绝对路径，
程序只能检测无线回传的约束，对于微波连接暂无检测
'''
