# BA-Transformer源码说明文档

## 1. 项目概述

BA-Transformer (BAT) 是一种基于Transformer架构的医学图像分割模型，创新性地引入了边界感知机制。该模型专为解决医学图像（特别是皮肤病变）分割中的边界精确定位问题而设计。通过在Transformer框架中整合边界注意力门控（Boundary-wise Attention Gate, BAG）机制，BAT能够在保持Transformer对全局信息建模能力的同时，强化对目标边界的感知能力，从而提高分割精度。

该项目实现了完整的BA-Transformer框架，支持ISIC2018、ISIC2016和BUSI256等皮肤病变数据集，包含从数据预处理、模型训练到性能评估的完整流程。

## 2. 项目结构

```
BA-Transformer/
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖包列表
├── .gitignore               # Git忽略文件配置
├── framework.jpg            # 项目框架示意图
├── train.py                 # 训练主程序
├── test.py                  # 测试主程序
├── infer.py                 # 推理脚本
│
├── src/                     # 源码目录
│   ├── transformer.py           # Transformer相关实现
│   ├── BAT_Modules.py           # BAT模块实现
│   ├── losses.py                # 损失函数定义
│   ├── utils.py                 # 工具函数
│   ├── process_point.py         # 点处理相关
│   ├── process_resize.py        # 图像缩放处理
│   ├── process_my_busi256.py    # BUSI256数据处理
│   └── process_my_isic2018.py   # ISIC2018数据处理
│
├── Ours/                    # 自定义模型与模块
│   ├── ASPP.py                  # 空洞空间金字塔池化模块
│   ├── Base_transformer.py      # 基础Transformer实现
│   ├── base.py                  # 基础网络结构
│   ├── cell_DETR.py             # Cell-DETR相关实现
│   ├── non_local.py             # 非局部模块
│   ├── resnet.py                # ResNet骨干网络
│   └── __init__.py              # 包初始化
│
├── dataset/                 # 数据集处理与分割
│   ├── isic2018.py              # ISIC2018数据集处理
│   ├── isic2016.py              # ISIC2016数据集处理
│   ├── busi256.py               # BUSI256数据集处理
│   ├── data_split_isic2018.json # ISIC2018数据集划分
│   ├── data_split_busi256.json  # BUSI256数据集划分
│   └── __init__.py              # 包初始化
├── ISIC2018/                # ISIC2018原始数据
│   ├── Image/                    # 图像数据
│   ├── Label/                    # 标签数据
│   └── Point/                    # 点标注数据
├── BUSI256/                 # BUSI256原始数据
│   ├── Image/                    # 图像数据
│   ├── Label/                    # 标签数据
│   └── Point/                    # 点标注数据
├── logs/                    # 日志与训练结果
│   ├── isic2018/
│   │   └── _1_1_0_e6_loss_0_aug_1/
│   │       └── fold_0/
│   │           ├── parameter.txt     # 参数记录
│   │           └── log/              # 训练日志（TensorBoard等）
│   └── busi256/
│       └── _1_1_0_e6_loss_0_aug_1/
│           └── fold_0/
│               ├── parameter.txt     # 参数记录
│               └── log/              # 训练日志
├── lib/                     # 第三方/外部依赖库
│   ├── non_local/               # 非局部相关模块
│   └── Cell_DETR_master/        # Cell-DETR相关代码
└── .git/                        # Git版本控制文件夹
```

## 3. 核心源码文件详解

### 3.1 src/BAT_Modules.py

该文件实现了BA-Transformer的核心模块，包括边界注意力门控（BAG）机制。

- **CrossAttention**: 实现标准的交叉注意力机制，用于特征融合
- **BoundaryCrossAttention**: 扩展CrossAttention，集成了边界注意力门控
- **MultiHeadAttention**: 多头注意力机制，用于Transformer中的注意力计算
- **BoundaryWiseAttentionGateAtrous2D/1D**: 基于空洞卷积的边界注意力门控，2D和1D版本
- **BoundaryWiseAttentionGate2D/1D**: 基础边界注意力门控实现，2D和1D版本

BAG的核心思想是通过特殊的注意力机制强化对图像边界区域的关注，其主要分为两种实现：基础版和基于空洞卷积的增强版。

```python
class BoundaryWiseAttentionGateAtrous2D(nn.Module):
    def __init__(self, in_channels, hidden_channels = None):
        # 使用多尺度空洞卷积捕获边界信息
        # 空洞率分别为1,2,4,6，覆盖不同感受野
        # ...

    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x  # 特征增强和残差连接
        return x, weight
```

### 3.2 src/transformer.py

该文件实现了标准Transformer和边界感知Transformer（BoundaryAwareTransformer）。

- **Transformer**: 标准Transformer实现，包含编码器和解码器
- **BoundaryAwareTransformer**: 集成了边界感知机制的Transformer
- **TransformerEncoder/BoundaryAwareTransformerEncoder**: 编码器实现
- **TransformerEncoderLayer/BoundaryAwareTransformerEncoderLayer**: 编码器层实现
- **TransformerDecoder/TransformerDecoderLayer**: 解码器及其层实现

BoundaryAwareTransformerEncoderLayer的关键在于它集成了BAG模块，使得Transformer能够更好地感知边界区域：

```python
class BoundaryAwareTransformerEncoderLayer(TransformerEncoderLayer):
    "Add Boundary-wise Attention Gate to Transformer's Encoder"

    def __init__(self,
                 d_model,
                 nhead,
                 BAG_type='2D',
                 Atrous=True,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         normalize_before)
        # 选择BAG类型：1D或2D，空洞或非空洞
        if BAG_type == '1D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous1D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate1D(d_model)
        elif BAG_type == '2D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous2D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate2D(d_model)
        self.BAG_type = BAG_type
```

### 3.3 Ours/Base_transformer.py

该文件定义了BAT的整体模型架构，结合了DeepLabV3、ResNet和边界感知Transformer。

- BAT类

  : 模型的主要类，整合了各个组件

  - 包含backbone（ResNet）
  - 使用Transformer进行特征增强
  - 集成了边界感知机制
  - 输出分割结果和点预测

```python
class BAT(nn.Module):
    def __init__(
            self,
            num_classes,
            num_layers,
            point_pred,
            decoder=False,
            transformer_type_index=0,
            hidden_features=128,
            number_of_query_positions=1,
            segmentation_attention_heads=8):

        super(BAT, self).__init__()

        self.num_classes = num_classes
        self.point_pred = point_pred
        self.transformer_type = "BoundaryAwareTransformer" if transformer_type_index == 0 else "Transformer"
        self.use_decoder = decoder

        # 使用DeepLabV3作为基础模型
        self.deeplab = base(num_classes, num_layers)

        # 特征映射层
        in_channels = 2048 if num_layers == 50 else 512
        self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)

        # 查询位置和位置编码
        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
                                            requires_grad=True)
        self.row_embedding = nn.Parameter(data=torch.randn(100,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)

        # 选择Transformer类型
        self.transformer = [
            Transformer(d_model=hidden_features),
            BoundaryAwareTransformer(d_model=hidden_features)
        ][point_pred]

        # 可选的解码器配置
        if self.use_decoder:
            self.BCA = BoundaryCrossAttention(hidden_features, 8)

        # 特征映射回原始维度
        self.trans_out_conv = nn.Conv2d(in_channels=hidden_features,
                                        out_channels=in_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        bias=True)
```

### 3.4 Ours/ASPP.py

实现了空洞空间金字塔池化（Atrous Spatial Pyramid Pooling）模块，用于多尺度特征提取。

- **ASPP类**: 基础ASPP实现，适用于ResNet18和ResNet34
- **ASPP_Bottleneck类**: 适用于ResNet50及以上的ASPP实现

```python
class ASPP(nn.Module):
    def __init__(self, num_classes, head = True):
        super(ASPP, self).__init__()

        # 1x1卷积分支
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        # 空洞卷积分支，空洞率为6
        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        # 空洞卷积分支，空洞率为12
        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        # 空洞卷积分支，空洞率为18
        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        # 全局平均池化分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        # 合并所有分支
        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        
        # 可选的分类头
        if head:    
            self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.head = head
```

### 3.5 dataset/isic2018.py

ISIC2018数据集的处理类，实现了数据加载、预处理和增强。

- myDataset类

  : PyTorch Dataset实现，处理ISIC2018数据

  - 支持交叉验证（K-Fold）
  - 实现数据增强
  - 加载图像、标签和点标注

```python
class myDataset(data.Dataset):
    def __init__(self, fold, split, aug=False):
        super(myDataset, self).__init__()
        self.split = split
        root_data_dir = './ISIC2018/'

        # 加载图像、标签和点标注路径
        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        
        # 根据交叉验证fold选择训练/验证数据
        indexes = [l[:-4] for l in os.listdir(root_data_dir + 'Image/')]
        valid_indexes = seperable_indexes[fold]
        train_indexes = list(filter(lambda x: x not in valid_indexes, indexes))
        
        # 选择训练集或验证集
        indexes = train_indexes if split == 'train' else valid_indexes
        
        # 构建文件路径
        self.image_paths = [root_data_dir + '/Image/{}.npy'.format(_id) for _id in indexes]
        self.label_paths = [root_data_dir + '/Label/{}.npy'.format(_id) for _id in indexes]
        self.point_paths = [root_data_dir + '/Point/{}.npy'.format(_id) for _id in indexes]
        
        # 数据增强配置
        self.aug = aug
        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
        ])
```

### 3.6 train.py

train.py是整个模型训练的核心脚本，用于设置训练参数、初始化模型、加载数据、定义损失函数和优化策略，并执行训练循环。

#### 命令行参数解析

首先，脚本通过argparse解析命令行参数，包括模型架构、数据集、GPU设置、学习率等训练参数。主要参数包括：

- `--arch`: 模型架构，默认为'BAT'（BA-Transformer）
- `--gpu`: 使用的GPU ID
- `--net_layer`: ResNet骨干网络的层数（18或50）
- `--dataset`: 使用的数据集（isic2018、isic2016或busi256）
- `--fold`: 交叉验证折数
- `--lr_seg`: 分割任务的学习率
- `--n_epochs`: 训练轮数
- `--point_pred`: 是否使用点预测（边界预测）
- `--ppl`: 点预测层数
- `--trans`: 是否使用Transformer

#### 实验设置和日志

然后，脚本设置实验名称和日志路径，创建必要的目录结构，并初始化TensorBoard写入器：

```python
exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
    parse_config.seg_loss) + '_aug_' + str(parse_config.aug) + '/fold_' + str(
        parse_config.fold)

os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
writer = SummaryWriter('logs/{}/log'.format(exp_name))
save_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)
```

#### 数据集和模型加载

接下来，脚本加载指定数据集的训练集和验证集，并初始化模型：

```python
if parse_config.dataset == 'isic2018':
    from dataset.isic2018 import norm01, myDataset
    dataset = myDataset(fold=parse_config.fold, split='train', aug=parse_config.aug)
    dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
# ... 其他数据集类似

train_loader = torch.utils.data.DataLoader(dataset, batch_size=parse_config.bt_size, 
                                          shuffle=True, num_workers=2, 
                                          pin_memory=True, drop_last=True)

if parse_config.arch == 'BAT':
    if parse_config.trans == 1:
        from Ours.Base_transformer import BAT
        model = BAT(1, parse_config.net_layer, parse_config.point_pred, 
                   parse_config.ppl).cuda()
    else:
        from Ours.base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()
```

#### 损失函数和优化器

然后，定义优化器和学习率调度器：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
```

该脚本定义了多种损失函数，包括Dice损失、CE损失、结构损失和Focal损失，并根据配置选择使用的损失函数：

```python
criteon = [focal_loss, ce_loss][parse_config.seg_loss]
```

#### 训练函数（train）

`train(epoch)`函数实现了每个训练轮次的核心流程：

1. 将模型设置为训练模式
2. 遍历数据加载器，获取批次数据
3. 处理点标注（进行下采样匹配特征图尺寸）
4. 进行前向传播获取分割和点预测结果
5. 计算Dice损失和点预测损失
6. 反向传播和参数更新
7. 定期记录训练指标到TensorBoard

关键代码段：

```python
def train(epoch):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        
        # 下采样点标注以匹配特征图尺寸
        point_c4 = nn.functional.max_pool2d(point, kernel_size=(16, 16), stride=(16, 16))
        
        # 前向传播
        if parse_config.point_pred == 1:
            output, point_maps_pre = model(data)
            output = torch.sigmoid(output)
            
            # 计算损失
            loss_dc = dice_loss(output, label)
            point_loss = 0.
            for i in range(len(point_maps_pre)):
                point_loss += criteon(point_maps_pre[i], point_c4)
            point_loss = point_loss / len(point_maps_pre)
            
            loss = loss_dc + point_loss  # 总损失
            
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 验证函数（evaluation）

`evaluation(epoch, loader)`函数实现了验证过程：

1. 将模型设置为评估模式
2. 对验证集中的每个样本进行推理
3. 计算Dice系数和IoU等指标
4. 记录验证指标到TensorBoard
5. 返回平均指标值

#### 训练循环

最后，脚本执行主训练循环：

1. 对每个epoch，调用train和evaluation函数
2. 更新学习率（scheduler.step()）
3. 检查性能并保存最佳模型
4. 实现早停机制

```python
max_dice = 0
max_iou = 0
best_ep = 0
min_loss = 10
min_epoch = 0

for epoch in range(1, EPOCHS + 1):
    # 记录当前学习率
    this_lr = optimizer.state_dict()['param_groups'][0]['lr']
    writer.add_scalar('Learning Rate', this_lr, epoch)
    
    # 训练和验证
    start = time.time()
    train(epoch)
    dice, iou, loss = evaluation(epoch, val_loader)
    scheduler.step()
    
    # 模型保存逻辑
    if iou > max_iou:
        max_iou = iou
        best_ep = epoch
        torch.save(model.state_dict(), save_path)
    else:
        if epoch - best_ep >= parse_config.patience:
            print('Early stopping!')
            break
```

### 3.7 test.py

test.py是模型评估脚本，用于对训练好的模型进行测试并计算定量评估指标。

#### 命令行参数解析

该脚本首先解析命令行参数，包括日志名称、GPU设置、数据集和评估折数等：

```python
parser = argparse.ArgumentParser()
parser.add_argument('--log_name', type=str, default='bat_1_1_0_e6_loss_0_aug_1')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--dataset', type=str, default='isic2016')
parser.add_argument('--arch', type=str, default='BAT')
parser.add_argument('--net_layer', type=int, default=50)
parser.add_argument('--pre', type=int, default=0)
parser.add_argument('--trans', type=int, default=1)
parser.add_argument('--point_pred', type=int, default=1)
parser.add_argument('--ppl', type=int, default=6)
parser.add_argument('--cross', type=int, default=0)
```

#### 数据集和模型加载

然后，脚本加载指定的数据集和预训练模型：

```python
if parse_config.dataset == 'isic2018':
    from dataset.isic2018 import norm01, myDataset
    dataset = myDataset(parse_config.fold, 'valid', aug=False)
# ... 其他数据集类似

if parse_config.arch == 'BAT':
    if parse_config.trans == 1:
        from Ours.Base_transformer import BAT
        model = BAT(1, parse_config.net_layer, parse_config.point_pred, parse_config.ppl).cuda()
    else:
        from Ours.base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()

# 加载预训练权重
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/{}/{}/fold_{}/".format(
    parse_config.dataset, parse_config.log_name, parse_config.fold)
from src.utils import load_model
model = load_model(model, dir_path + 'model/best.pkl')
```

#### 日志配置

脚本配置日志记录，用于保存评估结果：

```python
txt_path = os.path.join(dir_path + 'parameter.txt')
logging.basicConfig(filename=txt_path,
                   level=logging.INFO,
                   format='[%(asctime)s.%(msecs)03d] %(message)s',
                   datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
```

#### 测试函数（test）

`test()`函数实现了模型评估的核心逻辑：

1. 将模型设置为评估模式
2. 遍历测试集，获取批次数据
3. 进行前向传播获取分割结果
4. 二值化预测结果（阈值0.5）
5. 计算评估指标：Dice系数、Jaccard系数（IoU）、Hausdorff 95%距离（HD95）和平均对称表面距离（ASSD）
6. 记录和输出平均指标

关键代码段：

```python
def test():
    model.eval()
    num = 0
    
    dice_value = 0
    jc_value = 0
    hd95_value = 0
    assd_value = 0
    
    # 收集所有预测和标签
    labels = []
    pres = []
    for batch_idx, batch_data in tqdm(enumerate(test_loader)):
        data = batch_data['image'].to(device).float()
        label = batch_data['label'].to(device).float()
        with torch.no_grad():
            # 前向传播
            if parse_config.point_pred == 0:
                output = model(data)
            elif parse_config.point_pred == 1:
                output, _ = model(data)
            output = torch.sigmoid(output)
            output = output.cpu().numpy() > 0.5  # 二值化
        
        label = label.cpu().numpy()
        labels.append(label)
        pres.append(output)
    
    # 合并所有批次结果
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)
    
    # 计算每个样本的指标
    for _id in range(labels.shape[0]):
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id])
        try:
            hd95_ave = hd95(labels[_id], pres[_id])
            assd_ave = assd(labels[_id], pres[_id])
        except RuntimeError:
            num += 1
            hd95_ave = 0
            assd_ave = 0
            
        dice_value += dice_ave
        jc_value += jc_ave
        hd95_value += hd95_ave
        assd_value += assd_ave
    
    # 计算平均指标
    dice_average = dice_value / (labels.shape[0] - num)
    jc_average = jc_value / (labels.shape[0] - num)
    hd95_average = hd95_value / (labels.shape[0] - num)
    assd_average = assd_value / (labels.shape[0] - num)
    
    # 记录指标
    logging.info('Dice value of test dataset  : %f' % (dice_average))
    logging.info('Jc value of test dataset  : %f' % (jc_average))
    logging.info('Hd95 value of test dataset  : %f' % (hd95_average))
    logging.info('Assd value of test dataset  : %f' % (assd_average))
    
    return dice_average
```

最后，脚本调用`test()`函数执行评估：

```python
if __name__ == '__main__':
    test()
```

## 4. 模型参数详解

### 4.1 命令行参数

BA-Transformer的命令行参数分为两类：数据集参数和模型参数。这些参数控制了模型的训练和评估过程。

数据集参数包括：

- `--dataset`: 指定使用的数据集，可选值包括'isic2018'、'isic2016'和'busi256'，分别对应不同的皮肤病变分割数据集。默认为'isic2016'。
- `--fold`: 在交叉验证中使用的折数，默认为'0'。该参数决定了数据如何分割为训练集和验证集。
- `--aug`: 是否使用数据增强，设置为1启用，0禁用。默认为1。数据增强包括高斯噪声、水平/垂直翻转和旋转等，有助于提高模型的泛化能力。
- `--bt_size`: 训练的批量大小，默认为8。根据GPU内存可以适当调整。

模型架构参数包括：

- `--arch`: 模型架构类型，默认为'BAT'，即BA-Transformer。
- `--net_layer`: ResNet骨干网络的层数，可选值为18或50，默认为50。ResNet50提供更强的特征提取能力，但需要更多计算资源。
- `--trans`: 是否使用Transformer，1表示启用，0表示仅使用DeepLabV3。默认为1。
- `--point_pred`: 是否使用边界点预测，1表示启用，0表示禁用。默认为1。启用后，模型将同时进行分割和边界点预测任务。
- `--ppl`: 点预测层数，默认为6。控制BoundaryAwareTransformer中的点预测层数，影响边界感知能力。
- `--cross`: 是否使用跨尺度特征融合，1表示启用，0表示禁用。默认为0。

训练参数包括：

- `--lr_seg`: 分割任务的学习率，默认为1e-4。
- `--n_epochs`: 训练轮数，默认为200。
- `--patience`: 早停耐心值，默认为500。如果验证性能在连续的patience个epoch内没有提升，训练将提前停止。
- `--seg_loss`: 分割损失类型，0表示focal_loss，1表示ce_loss（交叉熵损失）。默认为0。

### 4.2 模型结构参数

BA-Transformer模型（在Ours/Base_transformer.py的BAT类中定义）具有以下核心参数：

- `num_classes`: 分割类别数，通常为1，表示二分类任务（前景/背景）。
- `num_layers`: ResNet骨干网络的层数，可以是18或50。影响特征提取的深度和能力。
- `point_pred`: 控制是否使用边界点预测。当值为1时，模型将同时预测分割掩码和边界点，启用边界感知能力。
- `decoder`: 是否使用额外的解码器。默认为False。启用后可增强特征解码能力，但会增加计算开销。
- `transformer_type_index`: Transformer类型，0表示BoundaryAwareTransformer，1表示标准Transformer。
- `hidden_features`: Transformer的隐藏特征维度，默认为128。较大的值可提供更强的表达能力，但需要更多内存。
- `number_of_query_positions`: 查询位置数量，默认为1。在Transformer中用于生成查询嵌入。
- `segmentation_attention_heads`: 注意力头数量，默认为8。多头注意力允许模型关注不同的特征子空间。

边界注意力门控（BAG）模块具有以下参数：

- `in_channels`: 输入特征通道数，通常等于Transformer的隐藏特征维度。
- `hidden_channels`: BAG模块中的隐藏通道数，默认为`in_channels//2`。
- `BAG_type`: BAG的类型，可以是'1D'或'2D'。'2D'版本在空间维度上操作，提供更细粒度的边界感知。
- `Atrous`: 是否使用空洞卷积，默认为True。空洞卷积提供更大的感受野，有助于捕获多尺度边界信息。

BoundaryAwareTransformer参数包括：

- `point_pred_layers`: 边界点预测层数，默认为6。控制应用BAG的Transformer编码器层数。
- `d_model`: 模型维度，默认为512。控制Transformer内部表示的维度。
- `nhead`: 多头注意力头数，默认为8。
- `num_encoder_layers`: 编码器层数，默认为6。
- `num_decoder_layers`: 解码器层数，默认为2。
- `dim_feedforward`: 前馈网络维度，默认为2048。
- `dropout`: Dropout率，默认为0.1。提供正则化以防止过拟合。

## 5. 模型执行流程

按照ISIC2018数据集的复现过程，模型执行分为三个关键阶段：

### 5.1 数据预处理阶段

```bash
python src/process_my_isic2018.py
```

1. **数据准备**：
   - 读取ISIC2018原始数据集（图像和掩码）
   - 将图像和掩码调整为352×352像素
   - 对掩码进行二值化处理
2. **点标注生成**：
   - 为每个分割掩码检测边界轮廓
   - 根据特定算法选择关键边界点
   - 生成点标注热图
3. **数据集划分**：
   - 创建5折交叉验证数据划分
   - 保存到dataset/data_split_isic2018.json

数据处理完成后，形成预处理后的数据集：

```
ISIC2018/
├── Image/      # 图像 (.npy)
├── Label/      # 掩码 (.npy)
└── Point/      # 点标注 (.npy)
```

### 5.2 模型训练阶段

```bash
python train.py --dataset isic2018 --gpu 0
```

1. **初始化**：
   - 解析命令行参数设置实验配置
   - 创建实验目录和日志记录器
   - 加载训练集和验证集
   - 初始化BA-Transformer模型和优化器
2. **训练循环**：
   - 对每个epoch执行：
     - 训练阶段：前向传播、损失计算（Dice损失+点预测损失）、反向传播
     - 验证阶段：计算Dice系数和IoU
     - 学习率调整和早停检查
     - 保存当前最佳模型
3. **模型保存**：
   - 将最佳模型保存为best.pkl

### 5.3 模型评估阶段

```bash
python test.py --dataset isic2018 --gpu 0 --log_name _1_1_0_e6_loss_0_aug_1 --fold 0
```

1. **准备工作**：
   - 加载验证数据集
   - 初始化模型并加载预训练权重
   - 配置评估日志
2. **评估过程**：
   - 对每个测试样本进行推理
   - 二值化预测结果
   - 计算评估指标：
     - Dice系数（分割重叠度）
     - Jaccard系数（IoU）
     - Hausdorff 95%距离（HD95）
     - 平均对称表面距离（ASSD）
3. **结果记录**：
   - 输出所有指标的平均值
   - 将结果保存到parameter.txt

通过这个流程，可以完成BA-Transformer模型在ISIC2018数据集上的训练和评估，获得分割性能指标。模型通常能达到约0.91的Dice系数和约0.85的Jaccard系数。

