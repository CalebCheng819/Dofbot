"""
机器人学课程 Dofbot 机械臂基于改进DH参数法的正 / 逆运动学建模
"""
#运行前先在终端指定编码方式，以避免中文乱码（仅 Windows）
#$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
#$env:PYTHONUTF8 = 1
#python .\AU3307\Dofbot_2025\dh_kine_student.py 2>&1 | Out-File -Encoding utf8 assignment1_2.txt
# --------------------- 0. 控制台编码设置（仅 Windows） ---------------------
import sys, os
if os.name == "nt":  # Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------- 1. 导入常用库 ---------------------
import roboticstoolbox as rtb   # 机器人专用工具箱
import numpy as np              # 矩阵运算
import math                     # 数学函数库
from matplotlib import pyplot as plt          # 2D/3D 可视化

# --------------------- 2. 常量定义 ---------------------
pi = 3.1415926          # 自己指定 π，方便后续打印保留 7 位小数
# 连杆长度（单位：m，与实物一致）
l1 = 0.1045             # 连杆1长度（基座→关节2）
l2 = 0.08285            # 连杆2长度（关节2→关节3）
l3 = 0.08285            # 连杆3长度（关节3→关节4）
l4 = 0.12842            # 连杆4长度（关节4→末端）

# ==============================================
# 用改进 DH 法建立机器人模型Demo
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
DH_demo = rtb.DHRobot(
    [
        rtb.RevoluteMDH(),                  # 连杆1
        rtb.RevoluteMDH(d=l1),                  # 连杆1            
        rtb.RevoluteMDH(d=l2),                  # 连杆2
        rtb.RevoluteMDH(a=l3, alpha=pi),        # 连杆3
        rtb.RevoluteMDH(d=l4),                  # 连杆4

    ],
    name="DH_demo"       # 给机器人起个名字，打印时更直观
)

# 打印标准 DH 参数表（alpha、a、d、theta、offset）
print("========== DH_demo机器人 DH 参数 ==========")
print(DH_demo)

# --------------------- 零位验证 ---------------------
fkine_input0 = [0, 0, 0, 0, 0]          # 全部关节置 0
fkine_result0 = DH_demo.fkine(fkine_input0)
print("\n零位正解齐次变换矩阵:")
print(fkine_result0)
DH_demo.plot(q=fkine_input0, block=True) # 3D 可视化（阻塞模式）

# ==============================================
# 仿真任务0、 用改进 DH 法建立Dofbot机器人模型
# ==============================================
# RevoluteMDH(a, alpha, d, offset)
# 默认 theta 为关节变量，因此只写常数项即可
dofbot = rtb.DHRobot(
    [
                         
        rtb.RevoluteMDH(d=l1),
        rtb.RevoluteMDH(alpha=-pi/2, offset=-pi/2),
        rtb.RevoluteMDH(a=l2),
        rtb.RevoluteMDH(a=l3, offset=pi/2),
        rtb.RevoluteMDH(alpha=pi/2, d=l4)

    ],
    name="Dofbot"
)

# 打印标准 DH 参数表（alpha、a、d、theta、offset）
print("========== Dofbot机器人 DH 参数 ==========")
print(dofbot)

# --------------------- 4. Part0 零位验证 ---------------------
fkine_input0 = [0, 0, 0, 0, 0]          # 全部关节置 0
fkine_result0 = dofbot.fkine(fkine_input0)
print("\n零位正解齐次变换矩阵:")
print(fkine_result0)
dofbot.plot(q=fkine_input0, block=True) # 3D 可视化（阻塞模式）

# ==============================================
# 仿真任务1、 正运动学 —— 给出DH模型在以下 4 组关节角下的正运动学解
# ==============================================
# poses = [
#     [0., pi/3, pi/4, pi/5, 0.],            # demo
#     [pi/2, pi/5, pi/5, pi/5, pi],          # 1
#     [pi/3, pi/4, -pi/3, -pi/4, pi/2],      # 2
#     [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]     # 3
# ]

# -------- 1.1 demo  pose ----------
q_demo = [0., pi/3, pi/4, pi/5, 0.]
T_demo = dofbot.fkine(q_demo)#给定一组关节角 q_demo，计算 DOFBOT 末端执行器在基坐标系下的位姿矩阵 

print("\n========== Part1-0 (demo) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=True)

# -------- 1.2 pose 1 ----------
q_demo = [pi/2, pi/5, pi/5, pi/5, pi]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-1 (pose 1) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=True)

# -------- 1.3 pose 2 ----------
q_demo = [pi/3, pi/4, -pi/3, -pi/4, pi/2]
T_demo = dofbot.fkine(q_demo)
print("\n========== Part1-2 (pose 2) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=True)

# -------- 1.4 pose 3 ----------
q_demo = [-pi/2, pi/3, -2*pi/3, pi/3, pi/3]
T_demo = dofbot.fkine(q_demo)   
print("\n========== Part1-3 (pose 3) 正解 ==========")
print(T_demo)
dofbot.plot(q=q_demo, block=True)

# ==============================================
# 仿真任务2、 逆运动学 —— 给出DH模型在以下 4 组笛卡尔空间姿态下的逆运动学解
# ==============================================
# targets = [
#     # demo
#     np.array([
#         [-1., 0., 0., 0.1],
#         [ 0., 1., 0., 0. ],
#         [ 0., 0.,-1.,-0.1],
#         [ 0., 0., 0., 1. ]
#     ]),
#     # 1
#     np.array([
#         [1., 0., 0., 0.1],
#         [0., 1., 0., 0. ],
#         [0., 0., 1., 0.1],
#         [0., 0., 0., 1. ]
#     ]),
#     # 2
#     np.array([
#         [cos(pi/3), 0.,-sin(pi/3), 0.2],
#         [0.,        1., 0.,        0. ],
#         [sin(pi/3), 0., cos(pi/3), 0.2],
#         [0.,        0., 0.,        1. ]
#     ]),
#     # 3
#     np.array([
#         [-0.866, -0.25,  -0.433, -0.03704],
#         [ 0.5,   -0.433, -0.75,  -0.06415],
#         [ 0.,    -0.866,  0.5,    0.3073 ],
#         [ 0.,     0.,     0.,     1.     ]
#     ])
# ]

# -------- 2.1 demo 目标 ----------
T_des_demo = np.array([
    [-1., 0., 0., 0.1],
    [ 0., 1., 0., 0. ],
    [ 0., 0.,-1.,-0.1],
    [ 0., 0., 0., 1. ]
])
q_ik_demo = dofbot.ik_LM(T_des_demo)[0]   # 取返回元组第 0 个元素
print("\n========== Part2-0 (demo) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_demo))
dofbot.plot(q=q_ik_demo, block=True)

# -------- 2.2 目标 1 ----------
Target_pos1 = np.array([
    [1., 0., 0., 0.1],
    [0., 1., 0., 0. ],
    [0., 0., 1., 0.1],
    [0., 0., 0., 1. ]
])
q_ik_target1 = dofbot.ik_LM(Target_pos1)[0]
print("\n========== Part2-1 (target 1) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_target1))
dofbot.plot(q=q_ik_target1, block=True)

# -------- 2.3 目标 2 ----------
Target_pos2 = np.array([
    [np.cos(pi/3), 0.,-np.sin(pi/3), 0.2],
    [0.,        1., 0.,        0. ],
    [np.sin(pi/3), 0., np.cos(pi/3), 0.2],
    [0.,        0., 0.,        1. ]
])
q_ik_target2 = dofbot.ik_LM(Target_pos2)[0]
print("\n========== Part2-2 (target 2) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_target2))
dofbot.plot(q=q_ik_target2, block=True)

# -------- 2.4 目标 3 ----------

Target_pos3 = np.array([
    [-0.866, -0.25,  -0.433, -0.03704],
    [ 0.5,   -0.433, -0.75,  -0.06415],
    [ 0.,    -0.866,  0.5,    0.3073 ],
    [ 0.,     0.,     0.,     1.     ]
])
q_ik_target3 = dofbot.ik_LM(Target_pos3)[0]
print("\n========== Part2-3 (target 3) 逆解 ==========")
print("关节角（rad）：", np.array(q_ik_target3))
dofbot.plot(q=q_ik_target3, block=True)

# ==============================================
# 仿真任务3、 工作空间可视化（≥500 点）
#     关节限位（°）→ 弧度
#     J1: [-180, 180]  J2~J5: [0, 180]
# ==============================================
#    题目给的是角度范围，对“关节变量 q”限幅；MDH 的实际角度是 theta = q + offset
# -----------------------------
deg = np.deg2rad
dofbot[0].qlim = [deg(-180), deg(180)]
dofbot[1].qlim = [deg(0),    deg(180)]
dofbot[2].qlim = [deg(0),    deg(180)]
dofbot[3].qlim = [deg(0),    deg(180)]
dofbot[4].qlim = [deg(0),    deg(180)]

N = 8000  # 采样点数量（≥500，图里更“饱满”）

low = np.array([lnk.qlim[0] for lnk in dofbot])
high = np.array([lnk.qlim[1] for lnk in dofbot])
q_samples = low + (high - low) * np.random.rand(N, 5)

XYZ = np.zeros((N, 3))
# 逐点 fkine（兼容所有 rtb 版本
for i, q in enumerate(q_samples):
    T = dofbot.fkine(q)
    XYZ[i] = T.t.A1 if hasattr(T.t, "A1") else np.array(T.t)

X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

# -----------------------------
# 5) 3D 可视化
# -----------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s=2, alpha=0.6)
ax.set_title(f"Dofbot Workspace (N={N})")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# 让坐标轴比例一致，避免视觉压扁
max_range = 0.5 * max(X.ptp(), Y.ptp(), Z.ptp())
mid_x, mid_y, mid_z = (X.min()+X.max())/2, (Y.min()+Y.max())/2, (Z.min()+Z.max())/2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()