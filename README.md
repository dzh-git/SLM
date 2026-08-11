# SLM 21×21 等强焦斑阵列工具

本项目计算相位型空间光调制器（SLM）的规格约束，使用理想薄透镜傅里叶模型仿真焦平面，并通过基于光斑 ROI 的加权 Gerchberg–Saxton（WGS）算法生成 21×21 等强焦斑阵列的相位图。最终测量支持零填充过采样，均匀性同时验收 CV、最小/最大功率比和最大相对偏差。

## 快速开始

```powershell
python -m slm_splitter size examples/reference_21x21.yaml
python -m slm_splitter run examples/quick_demo.yaml
pytest -q
```

真实 1 µm 目标配置为 `examples/reference_21x21.yaml`。它使用 4096×4096 网格，完整优化需要较大内存和较长时间。`examples/quick_demo.yaml` 保留 21×21 阵列，但放宽焦斑尺寸，仅用于快速验证软件流程。

详细计算依据和模型说明见：

- [项目目标、验收口径与代码审查指南](docs/project_scope_and_review_guide.md)（建议审查者先读）
- [SLM 参数选型依据](docs/slm_selection.md)
- [衍射仿真与相位优化方案](docs/diffraction_simulation.md)
