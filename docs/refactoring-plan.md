# MASt3R-SLAM Hydra 重构方案

## 一、目标

将当前基于全局 `config` dict + `config.yaml` 继承体系的配置方式，重构为 **Hydra** 结构化配置（structured configs）方式。每个核心模块归入独立文件夹，暴露一个主类（或一组紧密相关的类），并在 `conf/` 目录下管理各自的 YAML 配置文件。

---

## 二、当前架构 vs 目标架构

### 当前问题

- 全局 `config = {}` dict，通过 `from mast3r_slam.config import config` 在 10+ 个文件中直接读取
- 子进程入口 `run_backend` / `run_visualization` 需要调用 `set_global_config(cfg)` 重新设置全局状态
- `config.yaml` 的 `inherit` 机制是自制的，不支持组合覆盖
- 各模块配置散落在一个大 YAML 的不同 section 中，模块边界不清晰

### 目标架构

```
mast3r_slam/
├── conf/                          # Hydra 配置目录
│   ├── config.yaml                # 顶层组合配置（引用所有子模块配置）
│   ├── model/
│   │   └── mast3r.yaml            # 模型加载 / 推理配置
│   ├── matching/
│   │   └── default.yaml           # 投影匹配参数
│   ├── tracking/
│   │   └── default.yaml           # 帧间追踪参数
│   ├── optimization/
│   │   └── default.yaml           # 全局 BA 参数
│   ├── retrieval/
│   │   └── default.yaml           # 检索数据库参数
│   ├── data/
│   │   ├── default.yaml           # 数据集通用配置
│   │   └── tum.yaml               # 按数据集覆盖
│   └── visualization/
│       └── default.yaml
├── main.py                        # Hydra 入口 @hydra.main
├── mast3r_slam/
│   ├── __init__.py
│   ├── model/                     # MASt3R 模型封装
│   │   ├── __init__.py
│   │   └── wapper.py              # 类: MASt3RWrapper
│   ├── matching/                  # 特征匹配
│   │   ├── __init__.py
│   │   └── matcher.py             # 函数集(无状态)，通过配置注入
│   ├── tracking/                  # 帧追踪器
│   │   ├── __init__.py
│   │   └── frame_tracker.py       # 类: FrameTracker
│   ├── optimization/              # 全局 BA
│   │   ├── __init__.py
│   │   └── factor_graph.py        # 类: FactorGraph
│   ├── retrieval/                 # 检索数据库
│   │   ├── __init__.py
│   │   └── database.py            # 类: RetrievalDatabase
│   ├── data/                      # 数据加载
│   │   ├── __init__.py
│   │   ├── dataset.py             # 类: MonocularDataset + 子类 + load_dataset
│   │   └── intrinsics.py          # 类: Intrinsics
│   ├── frame/                     # 核心数据结构
│   │   ├── __init__.py
│   │   ├── frame.py               # dataclass: Frame, 函数: create_frame
│   │   └── shared.py              # 类: SharedStates, SharedKeyframes
│   ├── visualization/             # 3D 可视化
│   │   ├── __init__.py
│   │   ├── window.py              # 类: Window, 函数: run_visualization
│   │   └── utils.py               # 可视化辅助函数
│   ├── geometry.py                # 纯数学工具（不改动，无 config 依赖）
│   ├── nonlinear_optimizer.py     # 鲁棒代价函数（不改动，无 config 依赖）
│   ├── image.py                   # 图像工具（不改动，无 config 依赖）
│   ├── evaluate.py                # 评估导出（参数化 use_calib）
│   └── tictoc.py                  # 计时工具（不改动）
└── config/                        # 旧配置目录，重构后移除
```

---

## 三、模块拆分方案

### 3.1 model — MASt3R模型封装 `mast3r_slam/model/`

**当前来源:** `mast3r_slam/mast3r_utils.py`

**主类: `MASt3RWrapper`**

```python
class MASt3RWrapper:
    def __init__(self, model_cfg: ModelConfig, device="cuda"):
        # 加载 MASt3R 模型和 retrieval 模型
        self.model = load_mast3r(model_cfg.checkpoint, device)
        self.retriever_model = load_retriever(model_cfg.retrieval_checkpoint, ...)

    # 单帧推理（init / reloc）
    def inference_mono(self, frame) -> tuple[Tensor, Tensor]: ...

    # 非对称推理（tracking）
    def inference_asymmetric(self, frame_i, frame_j) -> tuple[Tensor, ...]: ...

    # 对称推理（global BA）
    def inference_symmetric(self, frame_i, frame_j) -> tuple[Tensor, ...]: ...
    def decode_symmetric_batch(self, feat_i, pos_i, feat_j, pos_j, ...) -> tuple[Tensor, ...]: ...

    # 组合推理+匹配的高层接口
    def match_asymmetric(self, frame_i, frame_j, matcher, idx_i2j_init=None) -> tuple: ...
    def match_symmetric(self, feat_i, pos_i, feat_j, pos_j, matcher) -> tuple: ...
```

**配置 `conf/model/mast3r.yaml`:**

```yaml
# @package _group_
checkpoint: "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
retrieval_checkpoint: "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
codebook_path: "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
img_downsample: 1  # 特征图下采样因子
```

**变更要点:**
- 移除全局 config 读取（`downsample` 改为从构造函数注入 `img_downsample`）
- `load_mast3r` / `load_retriever` 变为类的内部方法
- `resize_img` 保留为独立函数（无 config 依赖）

---

### 3.2 matching — 特征匹配 `mast3r_slam/matching/`

**当前来源:** `mast3r_slam/matching.py`

**无状态模块**（纯函数集），不需要类封装，但通过配置注入参数：

```python
class Matcher:
    """封装匹配逻辑，持有匹配配置"""
    def __init__(self, matching_cfg: MatchingConfig):
        self.cfg = matching_cfg

    def match(self, X11, X21, D11, D21, idx_1_to_2_init=None) -> tuple[Tensor, Tensor]:
        return match_iterative_proj(X11, X21, D11, D21, self.cfg, idx_1_to_2_init)
```

或者保持函数式风格，但将 `MatchingConfig` 作为参数传入：

```python
def match(X11, X21, D11, D21, cfg: MatchingConfig, idx_1_to_2_init=None):
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, cfg, idx_1_to_2_init)
    return idx_1_to_2, valid_match2
```

**配置 `conf/matching/default.yaml`:**

```yaml
# @package _group_
max_iter: 10
lambda_init: 1e-8
convergence_thresh: 1e-6
dist_thresh: 0.1       # 3D 空间距离阈值
radius: 3
dilation_max: 5
```

---

### 3.3 tracking — 帧追踪器 `mast3r_slam/tracking/`

**当前来源:** `mast3r_slam/tracker.py`

**主类: `FrameTracker`**（保持不变，仅改构造函数）

```python
class FrameTracker:
    def __init__(self, model: MASt3RWrapper, keyframes, tracking_cfg: TrackingConfig,
                 use_calib: bool, device="cuda"):
        self.cfg = tracking_cfg      # 不再从全局 config 读取
        self.use_calib = use_calib   # 不再从全局 config 读取
        self.model = model
        self.keyframes = keyframes
        self.device = device
        self.reset_idx_f2k()

    def track(self, frame: Frame): ...
    def get_points_poses(self, ...): ...
    def solve(self, ...): ...
    def opt_pose_ray_dist_sim3(self, ...): ...
    def opt_pose_calib_sim3(self, ...): ...
```

**配置 `conf/tracking/default.yaml`:**

```yaml
# @package _group_
min_match_frac: 0.05
max_iters: 50
C_conf: 0.0
Q_conf: 1.5
rel_error: 1e-3
delta_norm: 1e-3
huber: 1.345
match_frac_thresh: 0.333
sigma_ray: 0.003
sigma_dist: 10.0
sigma_pixel: 1.0
sigma_depth: 10.0          # log-depth
sigma_point: 0.05
pixel_border: -10          # 仅 calib 模式
depth_eps: 1e-6            # 仅 calib 模式
filtering_mode: weighted_pointmap
filtering_score: median
```

---

### 3.4 optimization — 全局BA `mast3r_slam/optimization/`

**当前来源:** `mast3r_slam/global_opt.py`

**主类: `FactorGraph`**（保持不变，仅改构造函数）

```python
class FactorGraph:
    def __init__(self, model: MASt3RWrapper, frames, opt_cfg: OptimizationConfig,
                 K=None, device="cuda"):
        self.cfg = opt_cfg
        self.model = model
        self.frames = frames
        self.device = device
        self.window_size = opt_cfg.window_size
        ...

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False): ...
    def solve_GN_rays(self): ...
    def solve_GN_calib(self): ...
```

**配置 `conf/optimization/default.yaml`:**

```yaml
# @package _group_
pin: 1
window_size: 1000000
C_conf: 0.0
Q_conf: 1.5
min_match_frac: 0.1
pixel_border: -10          # 仅 calib
depth_eps: 1e-6            # 仅 calib
max_iters: 10
sigma_ray: 0.003
sigma_dist: 10.0
sigma_pixel: 1.0
sigma_depth: 10.0
sigma_point: 0.05
delta_norm: 1e-8
use_cuda: true             # 是否使用 CUDA 后端
```

---

### 3.5 retrieval — 检索数据库 `mast3r_slam/retrieval/`

**当前来源:** `mast3r_slam/retrieval_database.py`

**主类: `RetrievalDatabase`**（基本不需要改动）

当前这个类已经是"无 config 依赖"的状态——`k` 和 `min_thresh` 都是作为 `update()` 的参数传入的。在 Hydra 重构中：

```python
class RetrievalDatabase(Retriever):
    def __init__(self, retrieval_cfg: RetrievalConfig, backbone=None, device="cuda"):
        # 可以从 cfg 中获取默认的 k 和 min_thresh
        self.default_k = retrieval_cfg.k
        self.default_min_thresh = retrieval_cfg.min_thresh
        super().__init__(retrieval_cfg.checkpoint, backbone=backbone, device=device)
```

或者保持现有接口不变，由调用方（`main.py`）传入 Hydra 配置中的值。

**配置 `conf/retrieval/default.yaml`:**

```yaml
# @package _group_
k: 3
min_thresh: 0.005
```

---

### 3.6 data — 数据加载 `mast3r_slam/data/`

**当前来源:** `mast3r_slam/dataloader.py`

**需要重构的类:**

```python
class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg: DatasetConfig, dtype=np.float32):
        self.use_calibration = data_cfg.use_calib   # 不再读全局 config
        ...

    def subsample(self, subsample): ...
    def has_calib(self): ...

class Intrinsics:
    @staticmethod
    def from_calib(img_size, W, H, calib, use_calib, center_principle_point,
                   always_undistort=False):
        # 接收 use_calib 和 center_principle_point 而不是读全局 config
        ...

def load_dataset(dataset_path, data_cfg: DatasetConfig):
    # 将 data_cfg 传给各 dataset 子类的构造函数
```

**配置 `conf/data/default.yaml`:**

```yaml
# @package _group_
subsample: 1
img_downsample: 1
center_principle_point: true
use_calib: false  # 由顶层 config 覆盖
```

---

### 3.7 frame — 核心数据结构 `mast3r_slam/frame/`

**当前来源:** `mast3r_slam/frame.py`

**主要成分:**

| 组件 | 类型 | 是否依赖 config | 处理方式 |
|------|------|----------------|----------|
| `Mode` | Enum | 否 | 直接移动 |
| `Frame` | dataclass | 是（`filtering_mode`, `filtering_score`） | 添加配置字段或在构造函数注入 |
| `create_frame()` | 函数 | 是（`img_downsample`） | 将 `img_downsample` 作为参数 |
| `SharedStates` | 类 | 否 | 直接移动 |
| `SharedKeyframes` | 类 | 是（`use_calib`） | 将 `use_calib` 作为构造函数参数 |

**变更要点:**

```python
@dataclasses.dataclass
class Frame:
    ...
    filtering_mode: str = "weighted_pointmap"   # 新增字段
    filtering_score: str = "median"             # 新增字段

    def update_pointmap(self, X, C):
        # 使用 self.filtering_mode 替代 config["tracking"]["filtering_mode"]
        ...

def create_frame(i, img, T_WC, img_size=512, img_downsample=1, device="cuda:0"):
    # img_downsample 从参数获取

class SharedKeyframes:
    def __init__(self, manager, h, w, use_calib=False, buffer=512, ...):
        self.use_calib = use_calib   # 不再读全局 config
```

---

### 3.8 visualization — 3D可视化 `mast3r_slam/visualization/`

**当前来源:** `mast3r_slam/visualization.py` + `mast3r_slam/visualization_utils.py`

**需要重构:**

```python
class Window(Wnd):
    def __init__(self, states, keyframes, main2viz, viz2main,
                 use_calib=False, **kwargs):
        self.use_calib = use_calib   # 不再读全局 config

    def render(self, t, frametime):
        # 使用 self.use_calib

def run_visualization(states, keyframes, main2viz, viz2main,
                      use_calib=False):
    # 不再调用 set_global_config()
    window = Window(states, keyframes, main2viz, viz2main, use_calib=use_calib)
```

**配置 `conf/visualization/default.yaml`:**

```yaml
# @package _group_
# 可视化的额外配置（如颜色映射、点云密度等，待扩展）
point_size: 1.0
max_points: 200000
```

---

### 3.9 不重构的模块

以下模块**无 config 依赖**，保持为独立 `.py` 文件（不移入文件夹）：

| 文件 | 原因 |
|------|------|
| `geometry.py` | 纯数学函数（Sim(3) 运算、投影、Jacobian），零 config 依赖 |
| `nonlinear_optimizer.py` | Huber/Tukey 函数、收敛检查，零 config 依赖 |
| `image.py` | 图像梯度计算工具 |
| `tictoc.py` | 计时工具 |

---

### 3.10 移除的模块

| 文件 | 替代方案 |
|------|----------|
| `mast3r_slam/config.py` | 整个模块被 Hydra structured configs 取代，不再需要 `load_config`、`set_global_config`、全局 `config` dict |

---

## 四、Hydra 配置体系

### 4.1 顶层配置 `conf/config.yaml`

```yaml
# @package _global_
defaults:
  - model: mast3r
  - matching: default
  - tracking: default
  - optimization: default
  - retrieval: default
  - data: default
  - visualization: default

# 顶层标志
use_calib: false
single_thread: false

# 可被命令行覆盖
dataset_path: "datasets/tum/rgbd_dataset_freiburg1_room/"
calib_path: ""
no_viz: false
save_as: "default"
```

### 4.2 预置组合配置

通过 Hydra 的 experiment config 机制，可以组合出不同场景：

**`conf/experiment/tracking_calib.yaml`**（当前 `calib.yaml` 的等效）:
```yaml
# @package _global_
use_calib: true
data:
  subsample: 2
```

**`conf/experiment/eval_calib.yaml`**（当前 `eval_calib.yaml` 的等效）:
```yaml
# @package _global_
use_calib: true
single_thread: true
data:
  subsample: 2
```

**`conf/experiment/eval_no_calib.yaml`**（当前 `eval_no_calib.yaml` 的等效）:
```yaml
# @package _global_
use_calib: false
single_thread: true
```

运行方式：
```bash
# 等价于原来的 --config config/calib.yaml
python main.py +experiment=tracking_calib

# 等价于原来的 --config config/eval_calib.yaml --no-viz
python main.py +experiment=eval_calib no_viz=true

# 自定义 intrinsics
python main.py calib_path=config/intrinsics.yaml
```

---

## 五、`main.py` 重构

### 5.1 入口函数

```python
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. 数据集加载
    dataset = load_dataset(cfg.dataset_path, cfg.data)
    if cfg.calib_path:
        cfg.use_calib = True
        cfg.data.use_calib = True

    # 2. 创建共享数据结构（不再需要 manager，对主进程而言）
    keyframes = SharedKeyframes(...)
    states = SharedStates(...)

    # 3. 模型加载
    model = MASt3RWrapper(cfg.model, device="cuda:0")

    # 4. 模块实例化
    retrieval = RetrievalDatabase(cfg.retrieval, backbone=model.retriever_model)
    matcher_cfg = cfg.matching
    tracker = FrameTracker(model, keyframes, cfg.tracking, cfg.use_calib)
    # factor_graph 在 backend 子进程中创建

    # 5. 启动子进程（传入必要的配置对象，不再传全局 dict）
    backend = mp.Process(
        target=run_backend,
        args=(model, states, keyframes, cfg.optimization, cfg.retrieval,
              cfg.reloc, cfg.use_calib, K)
    )
    ...

    # 6. 主循环不变
    ...
```

### 5.2 子进程入口

```python
def run_backend(model, states, keyframes, opt_cfg, retrieval_cfg, reloc_cfg,
                use_calib, K):
    # 不再调用 set_global_config()
    factor_graph = FactorGraph(model, keyframes, opt_cfg, K=K)
    retrieval = RetrievalDatabase(retrieval_cfg)
    ...

def run_visualization(states, keyframes, main2viz, viz2main, use_calib):
    # 不再调用 set_global_config()
    window = Window(states, keyframes, main2viz, viz2main, use_calib=use_calib)
```

---

## 六、配置依赖矩阵

| 模块 | 依赖的配置 | 是否作为类构造函数参数 | 备注 |
|------|-----------|----------------------|------|
| `MASt3RWrapper` | `model` 全部 | 是 | `img_downsample` 在 `inference_*` 时使用 |
| `Matcher` | `matching` 全部 | 是（或改为函数参数） | 无状态，config 在每次调用时传入 |
| `FrameTracker` | `tracking` 全部 + `use_calib` | 是 | 重构前从全局 `config` 读两处 |
| `FactorGraph` | `optimization` 全部 | 是 | |
| `RetrievalDatabase` | `retrieval` | 可选（调用参数传入） | 当前已无 config 依赖 |
| `MonocularDataset` | `data` + `use_calib` | 是 | |
| `Intrinsics.from_calib` | `use_calib` + `center_principle_point` | 改函数参数 | |
| `Frame` | `filtering_mode` + `filtering_score` | 改为dataclass字段 | |
| `create_frame` | `img_downsample` | 改函数参数 | |
| `SharedKeyframes` | `use_calib` | 改构造函数参数 | |
| `Window` | `use_calib` | 改构造函数参数 | |
| `save_reconstruction` | `use_calib` | 改函数参数 | |
| `run_backend` | `optimization` + `retrieval` + `reloc` + `use_calib` | 作为函数参数 | 不再调用 `set_global_config` |
| `run_visualization` | `use_calib` | 作为函数参数 | 不再调用 `set_global_config` |

---

## 七、实施步骤

### 阶段一：基础设施（0 依赖其他改动）
1. 安装 Hydra：`pip install hydra-core omegaconf`
2. 在 `mast3r_slam/` 同级创建 `conf/` 目录及所有子配置 YAML
3. 创建 `mast3r_slam/config.py` 中的 dataclass 定义（`MatchingConfig`, `TrackingConfig` 等）

### 阶段二：模块独立（逐个模块迁移）
4. 创建 `mast3r_slam/model/` — 将 `mast3r_utils.py` 重构为 `MASt3RWrapper`
5. 创建 `mast3r_slam/matching/` — 将 `matching.py` 参数化
6. 创建 `mast3r_slam/tracking/` — 修改 `FrameTracker` 构造函数
7. 创建 `mast3r_slam/optimization/` — 修改 `FactorGraph` 构造函数
8. 创建 `mast3r_slam/retrieval/` — 移动 `retrieval_database.py`
9. 创建 `mast3r_slam/data/` — 拆分 `dataloader.py`
10. 创建 `mast3r_slam/frame/` — 拆分 `frame.py`，参数化配置依赖
11. 创建 `mast3r_slam/visualization/` — 参数化 `use_calib`

### 阶段三：入口重构
12. 重写 `main.py` 为 Hydra 入口，移除所有 `from mast3r_slam.config import config`
13. 修改所有子进程入口，移除 `set_global_config()`
14. 删除 `mast3r_slam/config.py` 和旧 `config/` 目录

### 阶段四：验证
15. 用 `python main.py --config config/calib.yaml` 的等效 Hydra 命令验证功能
16. 运行各数据集评估脚本

---

## 八、兼容性考虑

| 风险 | 缓解措施 |
|------|----------|
| `Frame.update_pointmap` 的 `filtering_mode`/`filtering_score` 需要运行时传入 | 在 `create_frame` 时从 `TrackingConfig` 获取默认值，或保留为类变量 |
| `SharedKeyframes.__getitem__` 每次读取 `config["use_calib"]` 判断是否设置 `kf.K` | 改为 `self.use_calib` 成员变量 |
| 子进程 `torch.multiprocessing.spawn` 中不能直接传 Hydra cfg 对象（序列化问题） | 将每个子进程需要的配置字段提取为普通 dict 或 dataclass 实例再传递 |
| MASt3R inference 中的 `downsample` 是 `torch.inference_mode` 下的，不能有 Python 控制流开销 | `img_downsample` 在初始化时确定，推理时直接用数字 |
| WSL/windows 分支 | windows 分支同样需要适配 |

### 关于多进程序列化

Hydra 的 `DictConfig` / `OmegaConf` 对象在 `torch.multiprocessing.spawn` 中可能无法直接序列化。解决方案：

```python
# 在主进程中提取需要的配置为普通数据结构
backend_cfg = {
    "optimization": OmegaConf.to_container(cfg.optimization, resolve=True),
    "retrieval": OmegaConf.to_container(cfg.retrieval, resolve=True),
    "reloc": OmegaConf.to_container(cfg.reloc, resolve=True),
    "use_calib": cfg.use_calib,
}
backend = mp.Process(target=run_backend, args=(model, states, keyframes, backend_cfg, K))
```

或者使用 `pickle`-safe 的 dataclass 实例（非 OmegaConf 对象）。

---

## 九、配置组合示例

重构后，以下场景只需组合不同配置组即可：

```bash
# 带标定的实时运行
python main.py use_calib=true data.subsample=2

# 不带标定的实时运行
python main.py use_calib=false

# 单线程评估运行（带标定）
python main.py use_calib=true single_thread=true no_viz=true \
  data.subsample=2 save_as=eval_tum1

# 使用 TUM 数据集特定的覆盖配置
python main.py +data=tum

# 使用自定义 intrinsics
python main.py calib_path=config/intrinsics.yaml
```
