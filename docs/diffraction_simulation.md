# 衍射仿真与 SLM 相位优化方案

## 1. 傅里叶光学模型

SLM 面复振幅写为

\[
U_{\rm SLM}(x,y)=A(x,y)\exp[i\phi(x,y)].
\]

SLM 位于理想薄透镜瞳面时，后焦平面复振幅正比于其二维傅里叶变换：

\[
U_f(x_f,y_f)\propto
\mathcal{F}\{U_{\rm SLM}\}
\bigg|_{f_x=x_f/(\lambda f),\,f_y=y_f/(\lambda f)}.
\]

代码使用正交归一化二维 FFT，并在整个规格计算和传播链中统一采用傍轴焦平面坐标

\[
x_f=\lambda f f_x,\qquad y_f=\lambda f f_y.
\]

转换为物理坐标。目标位置被映射到最近 FFT 像素，并单独记录该映射误差。

默认入射振幅是 Gaussian：

\[
A(x,y)=\exp[-(x^2+y^2)/w_{\rm in}^2].
\]

若 YAML 未显式给出 `input_beam_radius_mm`，程序根据目标焦斑半径自动计算 (w_{\rm in})。`input_profile: uniform` 可切换为平顶入射。

## 2. 可选非理想因素

`simulation` 配置节提供以下开关：

- `finite_aperture`：使用物镜 NA 对 SLM 面振幅施加圆形孔径；
- `quantize_phase`：把连续相位量化到 `slm.phase_levels` 个等级；
- `apply_fill_factor`：在焦平面乘以像元有效孔径对应的二维 sinc 包络；`fill_factor` 按厂商常见的面积比例解释，方形像元线性开口为 \(p\sqrt{F}\)；
- `roi_radius_spot_radii`：以目标焦斑半径的倍数定义积分区域，并自动限制为小于半个阵列间距，避免相邻 ROI 重叠；
- `measurement_oversampling`：仅在最终测量时对 SLM 场做零填充 FFT。参考值 2 把焦平面采样间隔缩小一半，不增加 WGS 每轮网格。

若要查看纯理想结果，可以关闭相位量化和像元孔径包络。参考硬件配置默认同时开启二者，而且 WGS 每轮前向传播都会应用相位量化和 sinc 包络。

## 3. 加权 Gerchberg–Saxton 算法

相位优化使用固定随机种子的 WGS：

1. 在 441 个目标位置赋予相同振幅和随机初相位，逆 FFT 得到初始 SLM 相位。
2. 用固定入射振幅和当前相位执行正向 FFT。
3. 在每个目标点周围的 ROI 内直接积分功率 (P_j)，计算

   \[
   CV=\sigma(P_j)/\overline{P}.
   \]
4. 在第 \(j\) 个完整 ROI 内按下式缩放复振幅：

   \[
   U'_f\big|_{ROI_j}=U_f\big|_{ROI_j}
   \left(\frac{\overline{P}}{P_j}\right)^{\alpha/2}
   \]

   其中 `weight_exponent` 为 \(\alpha\)。平方根把功率修正转换为场振幅修正。
5. 目标 ROI 外复场保持自由；逆 FFT 后只保留相位，重新施加已知入射振幅。
6. 同时计算 CV、最小/最大功率比和最大相对偏差。三项连续达到设定容差，或达到最大迭代次数时停止。
7. 用三项归一化违规度的最大值

   \[
   S=\max\left(
   \frac{CV}{CV_{\max}},
   \frac{R_{\min}}{P_{\min}/P_{\max}},
   \frac{D}{D_{\max}}
   \right)
   \]

   选择已实际评价过的最佳相位，其中 \(D=\max_i|P_i/\overline P-1|\)。三项全部通过等价于 \(S\le1\)。

旧实现只约束 441 个中心像素并按 CV 选优，少数离群光斑可被总体 CV 掩盖。当前实现缩放完整 ROI，并按最差归一化指标选优；目标区外仍保持自由，避免强制全平面零背景造成不必要的效率损失。

## 4. 结果评价

每个目标点的圆形 ROI 内计算：

- 背景扣除后的积分功率和峰值强度；
- 强度质心及其相对目标坐标的位置偏差；
- 对

  \[
  \ln I(r)=C-2r^2/w^2
  \]

  做线性拟合得到 (1/e^2) 半径；采样点不足时回退到二阶矩估计；
- 441 点功率 CV；
- 最小/最大功率比 \(P_{\min}/P_{\max}\)；
- 最大相对功率偏差 \(\max_i|P_i/\overline P-1|\)；
- 焦斑半径的最小值、5% 分位、中值、均值、95% 分位、最大值及相对目标误差；
- 目标 ROI 总功率占总焦平面功率的数值衍射效率。

参考配置要求 `CV≤5%`、`Pmin/Pmax≥0.90` 且最大相对偏差不超过 10%。最终测量使用零填充把离散傅里叶场插值到更密的频率网格：若过采样因子为 \(q\)，采样间隔由 \(\Delta x_f\) 变为 \(\Delta x_f/q\)。零填充不增加新的光学信息，但能显著降低质心和曲线拟合对 FFT 网格吸附的敏感性。4096²、2 倍过采样需要执行 8192² FFT，内存随 \(q^2\) 增长。

## 5. 运行方式与输出

只计算选型要求：

```powershell
python -m slm_splitter size examples/reference_21x21.yaml
```

优化相位：

```powershell
python -m slm_splitter optimize examples/reference_21x21.yaml
```

使用已有相位进行仿真：

```powershell
python -m slm_splitter simulate examples/reference_21x21.yaml --phase outputs/strict_reference_21x21/slm_phase.npy
```

依次执行选型、优化和仿真：

```powershell
python -m slm_splitter run examples/reference_21x21.yaml
```

4096² 完整计算会使用数百 MB 至数 GB 峰值内存。可先运行 `examples/quick_demo.yaml` 验证软件和绘图流程：

```powershell
python -m slm_splitter run examples/quick_demo.yaml
```

输出目录包含：

- `slm_requirements.json/.md`：解析计算和候选器件检查；
- `slm_phase.npy`：弧度制浮点相位；
- `slm_phase_8bit.png`：0–255 线性相位图；
- `optimization_metrics.json`、`convergence.png`：优化历史；
- `simulation_metrics.json`、`spot_metrics.csv`：总指标和逐光斑指标；
- `focal_plane_linear.png`、`focal_plane_log.png`、`central_cross_section.png`、`central_spot.png`：焦平面结果。

## 6. 模型边界

- 当前模型是单色、标量、相干、理想薄透镜模型，不同时优化多个波长。
- 未包含实测像差、SLM 表面形貌、零级泄漏、偏振耦合、器件相位 LUT 和物镜离轴像差。
- 8 位 PNG 不能直接等同于真实 SLM 灰度码；必须经目标波长下的相位标定表转换。
- 仿真通过表示离散数学模型满足指标，不替代 SLM 标定、物镜视场测试和实验焦平面测量。

## 7. 修正后的实现准确性审计

傅里叶传播部分遵循标量、单色、傍轴薄透镜模型 [D1, D2]；WGS 的基本交替投影框架来自 Gerchberg–Saxton/Fienup，针对多焦点阵列的权重均衡可参见 Di Leonardo 等人的工作 [D3–D5]。修正后的实现完成了以下一致性工作，同时保留必要边界：

1. **填充率已进入优化循环。** 95% 按面积比例处理，线性像元开口为 \(p\sqrt{0.95}\)；sinc 包络和 256 级量化均进入每轮 WGS [D6]。
2. **坐标模型已统一。** 传播、Nyquist 视场和像元上限均使用 \(x_f=\lambda f f_x\)。该关系是傍轴模型；接近 NA=0.5 时仍需改用一致的 Debye、角谱或矢量衍射模型。
3. **二维位置检查已修正。** 规格计算采用 \(\frac12\sqrt{\Delta x^2+\Delta y^2}\)；参考案例原生网格数值上界约 0.712 µm，2 倍过采样后的最终质心最大误差为 0.805 µm，仍小于 2 µm 指标。
4. **最终测量已使用零填充。** 参考配置的测量采样由 1.0066 µm 降至 0.5033 µm。优化仍在原生 4096² 网格进行，避免每轮使用 8192² FFT；零填充只用于最终测量，不改变相位解。
5. **物镜包络检查已补充。** 参考 99% Gaussian 包络要求最低物镜 NA=0.25696，配置已从 0.25 调整为 0.30。
6. **效率仍是数值 ROI 效率。** `numerical_roi_efficiency` 为背景扣除后目标 ROI 数值功率与仿真平面总功率之比；新参考值为 78.37%。它不包含 SLM 反射率、偏振损耗、零级泄漏、物镜透过率或探测器响应。
7. **等强验收已收紧。** 4096² 新结果为 CV=1.849%、最小/最大比=90.63%、最大相对偏差=5.68%，三项同时通过；旧结果 CV=4.744% 但最小/最大比只有 59.3%，不再视为等强通过。
8. **最佳相位选择已收紧。** 优化器按三项中最差的归一化指标选择实际评价过的相位，而不是只选择最低 CV。
9. **1 µm 半径得到更细网格复核。** 2 倍过采样后的中值 \(1/e^2\) 半径为 1.078 µm，5%–95% 分位为 1.011–1.225 µm，明显比只报告原生网格中值 1.168 µm 更完整。少数拟合离群值仍保留在 JSON/CSV 中；由于设计尚未规定半径允许误差，程序报告误差但不擅自判定半径通过。

综合判断：**修正后的 Fourier/WGS 代码在其标量傍轴、像素化相位模型内自洽并通过目标指标；最终硬件验收仍必须加入具体 SLM 的实测 LUT、像素串扰和波前像差。**

## 8. 参考资料

- **[D1]** J. W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H. Freeman, 2017, ISBN 978-1-319-11916-4。用于薄透镜傅里叶变换、Fraunhofer 衍射和采样理论。
- **[D2]** D. G. Voelz, *Computational Fourier Optics: A MATLAB Tutorial*, SPIE Press, 2011。[SPIE 图书页面](https://spie.org/Publications/Book/866274)。用于离散傅里叶光学与数值传播实现。
- **[D3]** R. W. Gerchberg and W. O. Saxton, “A practical algorithm for the determination of phase from image and diffraction plane pictures,” *Optik* 35, 237–246 (1972)。Gerchberg–Saxton 原始算法。
- **[D4]** J. R. Fienup, “Phase retrieval algorithms: a comparison,” *Applied Optics* 21, 2758–2769 (1982)。[DOI: 10.1364/AO.21.002758](https://doi.org/10.1364/AO.21.002758)。用于交替投影与相位恢复算法比较。
- **[D5]** R. Di Leonardo, F. Ianni, and G. Ruocco, “Computer generation of optimal holograms for optical trap arrays,” *Optics Express* 15, 1913–1922 (2007)。[DOI: 10.1364/OE.15.001913](https://doi.org/10.1364/OE.15.001913)。用于多焦点阵列的加权全息图优化。
- **[D6]** V. Arrizón, G. Ruiz, R. Carrada, and L. A. González, “Pixelated phase computer holograms for the accurate encoding of scalar complex fields,” *JOSA A* 24, 3500–3507 (2007)。[DOI: 10.1364/JOSAA.24.003500](https://doi.org/10.1364/JOSAA.24.003500)。用于像素化相位调制与离散全息编码。
