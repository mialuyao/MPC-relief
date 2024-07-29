import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.transforms import ScaledTranslation

# region 
# # Define the piecewise function
# def piecewise_function(x, t_eq, t_diff):
#     if x < t_eq:
#         return 0
#     elif x > t_diff:
#         return 1
#     else:
#         return (x - t_eq) / (t_diff - t_eq)

# # Parameters
# t_eq = 2
# t_diff = 5

# # Create x values
# x_values = np.linspace(0, 7, 400)
# y_values = np.array([piecewise_function(x, t_eq, t_diff) for x in x_values])

# # Plot the function with axis arrows and specific annotations, ensuring arrows on both axes and adjusting line lengths
# fig = plt.figure()
# #使用axisartist.Subplot方法创建一个绘图区对象ax
# ax = axisartist.Subplot(fig, 111)
# #将绘图区对象添加到画布中
# fig.add_axes(ax)
# # fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(x_values, y_values, label='Piecewise Function', color = 'black', linewidth=2)  # Increase line thickness
# ax.axvline(t_eq, color='red', linestyle='--')
# ax.axvline(t_diff, color='blue', linestyle='--')
# # ax.hlines(1, -0.1, 7.5, colors='green', linestyle='--')
# # ax.hlines(0, -0.1, 7.5, colors='black')

# # Annotate the axes
# ax.set_xticks([t_eq, t_diff])
# ax.set_xticklabels(['$t_{eq}$', '$t_{diff}$'])
# ax.set_yticks([0, 1])
# # ax.set_yticklabels(['0', '1'])

# # Remove the top and right spines
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.spines['left'].set_position('zero')
# # ax.spines['bottom'].set_position('zero')
# ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
# ax.axis["left"].set_axisline_style("-|>", size = 1.5)
# #通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
# ax.axis["top"].set_visible(False)
# ax.axis["right"].set_visible(False)

# ax.set_xlabel(r'$d$', fontsize=12)
# ax.set_ylabel(r'$diff$', fontsize=12)

# # Drawing axis arrows by extending spines
# # arrow_fmt = dict(markersize=6, color='black', linewidth=2)
# # ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, **arrow_fmt)
# # ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, **arrow_fmt)

# # # Adding axis labels
# # ax.text(7.6, -0.05, 'x', ha='center', va='center', fontsize=12, color='black')
# # ax.text(-0.1, 1.25, 'y', ha='center', va='center', fontsize=12, color='black')

# # Add parameter names
# # ax.text(3.5, -0.1, 'd', ha='center', va='center', fontsize=12, color='black')
# # ax.text(-0.3, 0.5, 'diff', ha='center', va='center', fontsize=12, color='black')
# # ax.set_title('Piecewise Function Plot with Axis Annotations and Arrows')
# # ax.legend()
# ax.grid(True)

# plt.show()

# endregion



# region 
# x<0, y=0; x>1, y=1; x\in [0,1], y=x
# 定义分段函数
# def linear_sigmod(x):
#     if x < 0:
#         return 0
#     elif x > 1:
#         return 1
#     else:
#         return x

# # Define the piecewise function
# def ramp_function(x, t_eq, t_diff):
#     if x < t_eq:
#         return 0
#     elif x > t_diff:
#         return 1
#     else:
#         return (x - t_eq) / (t_diff - t_eq)

# # 生成数据
# x_ls = np.linspace(-1, 4, 400)
# y_ls = np.array([linear_sigmod(xi) for xi in x_ls])

# # Parameters
# t_eq = 1.5
# t_diff = 3

# # Create x values
# x_ramp = np.linspace(-1, 4, 400)
# y_ramp = np.array([ramp_function(x, t_eq, t_diff) for x in x_ramp])

# # 绘图
# plt.figure(figsize=(7, 3))
# plt.plot(x_ls, y_ls, label='Linear Sigmoid function', color='#376795', linewidth=2.5)  # 增加线条粗细
# plt.plot(x_ramp, y_ramp, label='Ramp function', color='#E76254', linewidth=2)  # 增加线条粗细


# # Mark t_eq and t_diff on the x-axis
# plt.axvline(t_eq, color='gray', linestyle='--', linewidth=1)
# plt.axvline(t_diff, color='gray', linestyle='-.', linewidth=1)
# # plt.text(t_eq, -0.1, r'$t_{eq}$', horizontalalignment='center', verticalalignment='center')
# # plt.text(t_diff, -0.1, r'$t_{diff}$', horizontalalignment='center', verticalalignment='center')

# plt.xlabel(r"$x$", loc='center', labelpad=13)
# plt.ylabel(r"$y$", loc='center', labelpad=13)
# # 设置刻度值
# plt.xticks([ 0, 1, t_eq, t_diff], [ 0, 1, r'$t_{eq}$', r'$t_{diff}$'], fontsize=13)
# plt.yticks([0, 1], [0, 1], fontsize=13)

# # # 设置刻度值
# # plt.xticks([-2, -1, 0, 1, 2], [ -2, -1, 0, 1, 2])
# # plt.yticks([0, 1], [0, 1])  # 不显示刻度2，只显示到-1

# #添加虚线
# plt.axhline(1, color='gray', linestyle='--', linewidth=1)
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.axvline(1, color='gray', linestyle='--', linewidth=1)
# plt.axvline(0, color='gray', linestyle='--', linewidth=1)
# # 设置轴范围
# # plt.xlim(-1.5, 2.5)
# # plt.ylim(-0.5, 1.5)

# # plt.title('Piecewise Linear Function')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()
# # endregion

# def hardtanh(x, a, b):
#     return np.where(x < a, a, np.where(x > b, b, x))

# # 自定义两个不同的 (a, b) 对
# params = [(1, 3), (-2, 2)]

# # x 轴取值范围
# x = np.linspace(-5, 5, 400)

# # 创建图形
# plt.figure(figsize=(10, 6))

# # 绘制第一组参数的hardtanh图像
# a, b = params[0]
# y1 = hardtanh(x, a, b)
# plt.plot(x, y1, label=f'Hardtanh ({a}, b={b})')

# # 绘制第二组参数的hardtanh图像
# a, b = params[1]
# y2 = hardtanh(x, a, b)
# plt.plot(x, y2, label=f'hardtanh ({a}, b={b})')

# # 图形装饰
# # plt.title('HardTanh Function')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.axhline(0, color='black',linewidth=0.5)
# # plt.axvline(0, color='black',linewidth=0.5)
# plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.legend()
# plt.show()
# endregion



# 定义linear_sigmod函数
def linear_sigmod(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

# 定义ramp_function函数
def ramp_function(x, t_eq, t_diff):
    if x < t_eq:
        return 0
    elif x > t_diff:
        return 1
    else:
        return (x - t_eq) / (t_diff - t_eq)
    
def N_ramp(x, min_value, max_value):
    if x < min_value:
        return 0
    elif x > max_value:
        return 1
    else:
        return x

# 定义hardtanh函数
def hardtanh(x, a, b):
    return np.where(x < a, a, np.where(x > b, b, x))

# min_value = [1, -1]
# max_value = [3, 2]
# # Define the range of x values
# x_ramp_1 = np.linspace(min_value[0] - 1, max_value[0] + 1, 400)
# y_ramp_1 = vectorized_L_Sigmoid(x_ramp_1, min_value[0], max_value[0])

# x_ramp_2 = np.linspace(min_value[1] - 1, max_value[1] + 1, 400)
# y_ramp_2 = vectorized_L_Sigmoid(x_ramp_2, min_value[1], max_value[1])

plt.figure(figsize=(7,4))

x = np.linspace(-4, 5, 400)
# 生成linear_sigmod数据
y_ls = np.array([linear_sigmod(i) for i in x])

# 生成ramp_function数据
y_ramp_1 = np.array([N_ramp(i, 1, 3) for i in x])
y_ramp_2 = np.array([N_ramp(i, -1, 2) for i in x])
# 创建图形

# 绘制第一组参数的hardtanh图像
y1 = hardtanh(x, 1, 3)
y2 = hardtanh(x, -1, 2)

plt.plot(x, y1, label=f'HardTanh (1, 3)', color='#1E466E', linewidth=2.2)
# 绘制第二组参数的hardtanh图像
plt.plot(x, y2, label=f'HardTanh (-1, 2)', color='#528FAD', linewidth=2.5, linestyle='-.')
# 绘制linear_sigmod图像
plt.plot(x, y_ls, label='N-ramp (0, 1)', color='#FFD06F', linewidth=2.5)
# 绘制linear_sigmod图像
plt.plot(x, y_ramp_1, label='N-ramp (1, 3)', color='#F7AA58', linewidth=1.8, linestyle='-.')
# 绘制ramp_function图像
plt.plot(x, y_ramp_2, label='N-ramp (-1, 2)', color='#E76254', linewidth=2, linestyle="dotted")

# 设置x轴刻度和网格线
plt.xticks(np.arange(-4, 6, 1))  # 设置x轴刻度
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 在每个刻度位置添加垂直虚线
for xc in np.arange(-4, 6, 1):
    plt.axvline(x=xc, color='gray', linestyle='--', linewidth=0.5)


# 图形装饰
# plt.title('Comparison of Different Piecewise Functions')
plt.xlabel('x')
plt.ylabel('y')
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
