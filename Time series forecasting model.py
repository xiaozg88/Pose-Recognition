import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

# cnn的参数配置
# class Config():
#     data_path = 'motion_augmented_data.csv'
#     timestep = 6  # 时间步长，就是利用多少时间窗口
#     batch_size = 12  # 批次大小
#     feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
#     out_channels = [10, 20, 30]  # 卷积输出通道
#     output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
#     epochs = 200 # 迭代轮数
#     best_loss = float('inf') # 记录损失
#     learning_rate = 0.0002 # 学习率
#     model_name = 'cnn' # 模型名称
#     save_path = './{}.pth'.format(model_name) # 最优模型保存路径
# cnn+lstm的参数配置
class Config():
    data_path = 'motion_augmented_data.csv'
    timestep = 6  # 时间步长，就是利用多少时间窗口
    batch_size = 12  # 批次大小
    feature_size = 2  # 每个步长对应的特征数量，这里只使用2维，每天的风速
    hidden_size = 256  # 隐层大小
    out_channels = 50 # CNN输出通道
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 2  # lstm的层数
    epochs = 200 # 迭代轮数
    best_loss = float('inf') # 记录损失o
    learning_rate = 0.0002 # 学习率
    model_name = 'cnn_lstm' # 模型名称
    save_path = './{}.pth'.format(model_name) # 最优模型保存路径
# cnn+ att的参数配置
# class Config():
#     data_path = 'motion_augmented_data.csv'
#     timestep = 20  # 时间步长，就是利用多少时间窗口
#     batch_size = 16  # 批次大小
#     feature_size = 2  # 每个步长对应的特征数量，这里只使用1维，每天的风速
#     num_heads = 1  # 注意力机制头的数量
#     out_channels = [10, 20, 30]  # 卷积输出通道
#     output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
#     epochs = 150  # 迭代轮数
#     best_loss = 0  # 记录损失
#     learning_rate = 0.00002  # 学习率
#     model_name = 'cnn_attention'  # 模型名称
#     save_path = './{}.pth'.format(model_name)  # 最优模型保存路径
# # lstm+att 的参数配置
# class Config():
#     data_path = 'motion_augmented_data.csv'
#     timestep = 6  # 时间步长，就是利用多少时间窗口
#     batch_size = 12  # 批次大小
#     feature_size = 2  # 每个步长对应的特征数量，这里只使用1维，每天的风速
#     num_heads = 1 # 注意力机制头的数量
#     hidden_size = 256 # lstm隐层维度
#     num_layers = 3 # lstm层数
#     output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
#     epochs = 200 # 迭代轮数
#     best_loss = float('inf')# 记录损失
#     learning_rate = 0.00002 # 学习率
#     model_name = 'lstm_attention' # 模型名称
#     save_path = './{}.pth'.format(model_name) # 最优模型保存路径
# # cnn+lstm+att的参数配置
# class Config():
#     data_path = 'motion_augmented_data.csv'
#     timestep =6  # Time steps
#     batch_size = 12
#     feature_size = 2  # Number of features per time step
#     hidden_size = 256
#     out_channels = 50
#     num_heads = 1
#     output_size = 1
#     num_layers = 2
#     epochs =200
#     learning_rate = 0.0002
#     best_loss = float('inf')
#     best_train_loss_epoch = -1
#     model_name = 'cnn_lstm_attention'
#     save_path = 'attention_LSTM_best_model.pth'
# LSTM配置
# class Config():
#     data_path = 'motion_augmented_data.csv'
#     timestep = 6  # 时间步长，就是利用多少时间窗口
#     batch_size = 12  # 批次大小
#     feature_size = 2  # 每个步长对应的特征数量，这里只使用1维，每天的风速
#     hidden_size = 256  # 隐层大小
#     output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
#     num_layers = 2  # lstm的层数
#     epochs = 200 # 迭代轮数
#     best_loss = 0 # 记录损失
#     learning_rate = 0.0003 # 学习率
#     model_name = 'lstm' # 模型名称
#     save_path = './{}.pth'.format(model_name) # 最优模型保存路径
config = Config()

# 1.加载时间序列数据
# df = pd.read_csv(config.data_path, index_col=0)
df = pd.read_csv(config.data_path, index_col=0, nrows=2000)  # 只加载前2千个数据值

# # 2.将数据进行标准化
# scaler = MinMaxScaler()
# scaler_model = MinMaxScaler()
# data = scaler_model.fit_transform(np.array(df))
# scaler.fit_transform(np.array(df['motion_count']).reshape(-1, 1))
# 数据划分
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep, :feature_size])
        dataY.append(data[index + timestep][0])


    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8* dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]

# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
# x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)   # min_max归一化
x_train, y_train, x_test, y_test = split_data(np.array(df), config.timestep, config.feature_size)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
print(x_train.shape)
print(x_test.shape)

# 5.形成训练、测试数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False)
test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)
print(len(train_loader),len(test_loader))
# 7.定义CNN + LSTM + Attention网络
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, out_channels, num_heads, output_size):
        super(CNN_LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数

        # 卷积层
        self.conv1d = nn.Conv1d(in_channels=feature_size, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # LSTM层
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为 1
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads,
                                               dropout=0.5)

        # 输出层
        self.fc1 = nn.Linear(timestep * hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        x = x.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]

        # 卷积运算
        output = self.conv1d(x)

        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        output = output.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]

        # LSTM运算
        output, (h_0, c_0) = self.lstm(output, (h_0, c_0))  # batch_size, timestep, hidden_size

        # 注意力计算
        attention_output, attn_output_weights = self.attention(output, output, output)

        # 展开
        output = output.flatten(start_dim=1)

        # 全连接层
        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)

        return output
# 7.定义CNN网络
class CNN(nn.Module):
    def __init__(self, feature_size, timestep, out_channels, output_size):
        super(CNN, self).__init__()

        # 定义二维卷积层
        self.conv2d_1 = nn.Conv2d(1, out_channels[0], kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1)
        self.conv2d_4 = nn.Conv2d(out_channels[2], out_channels[2], kernel_size=3, padding=1)
        self.conv2d_5 = nn.Conv2d(out_channels[2], out_channels[2], kernel_size=3, padding=1)

        # 定义输出层
        self.fc1 = nn.Linear(out_channels[2] * timestep * feature_size, 128)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(128, output_size)

        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(dim=1)  # batch_size, channels, timestep, feature_size

        x = self.conv2d_1(x)  # [32, 10, 20, 1]
        x = self.relu(x)

        x = self.conv2d_2(x)  # [32, 20, 20, 1]
        x = self.relu(x)

        x = self.conv2d_3(x)  # [32, 30, 20, 1]
        x = self.relu(x)

        x = self.conv2d_4(x)  # 新增的卷积层
        x = self.relu(x)

        x = self.conv2d_5(x)  # 新增的卷积层
        x = self.relu(x)

        x = x.flatten(start_dim=1)  # [32, 600]

        x = self.fc1(x)  # [32, 128]
        x = self.relu(x)
        x = self.dropout(x)  # 在全连接层之后应用Dropout

        x = self.fc2(x)  # [32, 1]

        return x

# 7.定义LSTM + CNN网络
class LSTM_CNN(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, out_channels, output_size):
        super(LSTM_CNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数

        # LSTM层
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为 1
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

        # 卷积层
        self.conv1d = nn.Conv1d(in_channels=timestep, out_channels=out_channels, kernel_size=3)

        # 输出层
        self.fc1 = nn.Linear(50 * 254, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # batch_size, timestep, hidden_size

        # 卷积运算
        output = self.conv1d(output)

        # 展开
        output = output.flatten(start_dim=1)

        # 全连接层
        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)

        return output

# 7.定义CNN + Attention网络
class CNN_Attention(nn.Module):
    def __init__(self, feature_size, timestep, out_channels, num_heads, output_size):
        super(CNN_Attention, self).__init__()
        self.hidden_size = 14

        # 定义一维卷积层
        self.conv1d_1 = nn.Conv1d(feature_size, out_channels[0], kernel_size=3)
        self.conv1d_2 = nn.Conv1d(out_channels[0], out_channels[1], kernel_size=3)
        self.conv1d_3 = nn.Conv1d(out_channels[1], out_channels[2], kernel_size=3)

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads,
                                               dropout=0.8)

        # 输出层
        self.fc1 = nn.Linear(self.hidden_size * out_channels[2], 256)
        self.fc2 = nn.Linear(256, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        x = x.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]

        x = self.conv1d_1(x)  # [32, 10, 18]
        x = self.relu(x)

        x = self.conv1d_2(x)  # [32, 20, 16]
        x = self.relu(x)

        x = self.conv1d_3(x)  # [32, 30, 14]
        x = self.relu(x)

        # 注意力计算
        attention_output, attn_output_weights = self.attention(x, x, x)
        #         print(attention_output.shape) # [32, 20, 64]
        #         print(attn_output_weights.shape) # [20, 32, 32]

        # 展开
        x = attention_output.flatten(start_dim=1)  # [32, 420]

        # 全连接层
        x = self.fc1(x)  # [32, 256]
        x = self.relu(x)

        x = self.fc2(x)  # [32, output_size]

        return x

# 7.定义LSTM + Attention网络
class LSTM_Attention(nn.Module):
    def __init__(self, feature_size, timestep, hidden_size, num_layers, num_heads, output_size):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                               dropout=0.8)
        # 输出层
        self.fc1 = nn.Linear(hidden_size * timestep, 256)
        self.fc2 = nn.Linear(256, output_size)
        # 激活函数
        self.relu = nn.ReLU()       # 批标准化层



    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # output[32, 20, 64]

        # 注意力计算
        attention_output, attn_output_weights = self.attention(output, output, output)
        #         print(attention_output.shape) # [32, 20, 64]
        #         print(attn_output_weights.shape) # [20, 32, 32]
        # 展开
        output = attention_output.flatten(start_dim=1)  # [32, 1280]

        # 全连接层
        output = self.fc1(output)  # [32, 256]
        output = self.relu(output)

        output = self.fc2(output)  # [32, output_size]

        return output

# 7.定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 我们只需要返回最后一个时间片的数据即可
        return output[:, -1, :]

# 定义LSTM + Attention网络
# best_model = LSTM_Attention(config.feature_size, config.timestep, config.hidden_size, config.num_layers, config.num_heads, config.output_size)
 # 定义LSTM + CNN网络
best_model = LSTM_CNN(config.feature_size, config.timestep, config.hidden_size, config.num_layers, config.out_channels, config.output_size)
 # 定义CNN + Attention网络
# best_model = CNN_Attention(config.feature_size, config.timestep, config.out_channels, config.num_heads, config.output_size)
# 定义CNN网络
# best_model = CNN(config.feature_size, config.timestep, config.out_channels, config.output_size)
# 定义CNN + LSTM + Attention网络
# best_model = CNN_LSTM_Attention(config.feature_size, config.timestep, config.hidden_size, config.num_layers,
#                                  config.out_channels, config.num_heads, config.output_size)
# best_model.load_state_dict(torch.load(config.save_path))
# 定义LSTM网络
# best_model= LSTM(config.feature_size, config.hidden_size, config.num_layers, config.output_size)  # 定义LSTM网络
# 定义优化器、学习率、学习率递减以及损失函数
new_optimizer = torch.optim.AdamW(best_model.parameters(), lr=config.learning_rate)
lr_scheduler = MultiStepLR(new_optimizer, milestones=[40], gamma=0.1)
# 使用不同的损失函数
new_loss_function_mae = nn.L1Loss()  # 设置了平均绝对误差为损失函数
new_loss_function_mse = nn.MSELoss()  # 设置了均方误差为损失函数
# 定义均方根误差损失函数
def RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# 8.模型训练
#  用于保存每一轮的训练与测试误差
# 用于保存每一轮的训练与测试误差
train_losses_mse = []
test_losses_mse = []
train_losses_mae = []
test_losses_mae = []
train_losses_rmse = []
test_losses_rmse = []
# 初始化最优损失以及所在轮次
# 初始化最优损失以及所在轮次
best_mse_train_loss = float('inf')
best_mae_train_loss = float('inf')
best_rmse_train_loss = float('inf')
best_mse_test_loss = float('inf')
best_mae_test_loss = float('inf')
best_rmse_test_loss = float('inf')
best_train_loss_epoch = -1
best_test_loss_epoch = -1


best_train_predictions = None
best_test_predictions = None

for epoch in range(config.epochs):   # 训练轮次
    best_model.train()
    running_loss_mse = 0
    running_loss_mae = 0
    running_loss_rmse = 0

    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        new_optimizer.zero_grad()
        y_train_pred = best_model(x_train)
        loss_mse = new_loss_function_mse(y_train_pred, y_train.reshape(-1, 1))
        loss_mae = new_loss_function_mae(y_train_pred, y_train.reshape(-1, 1))
        loss_rmse = RMSELoss(y_train_pred, y_train.reshape(-1, 1))
        loss_mse.backward(retain_graph=True)
        loss_mae.backward(retain_graph=True)
        loss_rmse.backward()
        new_optimizer.step()
        running_loss_mse += loss_mse.item()
        running_loss_mae += loss_mae.item()
        running_loss_rmse += loss_rmse.item()
        train_bar.desc = "train epoch[{}/{}] mse_loss:{:.3f}, mae_loss:{:.3f}, rmse_loss:{:.3f}".format(epoch + 1,
                                                                                                        Config.epochs,
                                                                                                        running_loss_mse / len(train_loader),
                                                                                                        running_loss_mae / len(train_loader),
                                                                                                        running_loss_rmse / len(train_loader))

    # Calculate and store the training loss for this epoch
    lr_scheduler.step()
    train_loss_mse = running_loss_mse / len(train_loader)   # 计算的是平均损失
    train_loss_mae = running_loss_mae / len(train_loader)
    train_loss_rmse = running_loss_rmse / len(train_loader)
    train_losses_mse.append(train_loss_mse)
    train_losses_mae.append(train_loss_mae)
    train_losses_rmse.append(train_loss_rmse)

    # 检查是否是最佳训练损失，未经过归一化
    best_mse_train_loss = min(train_losses_mse)
    best_mae_train_loss = min(train_losses_mae)
    best_rmse_train_loss = min(train_losses_rmse)
    best_train_loss_epoch_mse = train_losses_mse.index(best_mse_train_loss) + 1
    best_train_loss_epoch_mae = train_losses_mae.index(best_mae_train_loss) + 1
    best_train_loss_epoch_rmse = train_losses_rmse.index(best_rmse_train_loss) + 1

    # 模型验证
    best_model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    test_loss_rmse = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)

        for data in test_bar:
            x_test, y_test = data
            y_test_pred = best_model(x_test)
            loss_mse = new_loss_function_mse(y_test_pred, y_test.reshape(-1, 1))
            loss_mae = new_loss_function_mae(y_test_pred, y_test.reshape(-1, 1))
            loss_rmse = RMSELoss(y_test_pred, y_test.reshape(-1, 1))
            test_loss_mse += loss_mse.item()
            test_loss_mae += loss_mae.item()
            test_loss_rmse += loss_rmse.item()
            test_bar.desc = "test epoch[{}/{}] mse_loss:{:.3f}, mae_loss:{:.3f}, rmse_loss:{:.3f}".format(epoch + 1,
                                                                                                          Config.epochs,
                                                                                                          test_loss_mse / len( test_loader),
                                                                                                          test_loss_mae / len(test_loader),
                                                                                                          test_loss_rmse / len(test_loader))

    # test_losses.append(test_loss)  # 未经过归一化
    test_loss_mse1= test_loss_mse / len(test_loader)
    test_loss_mae1= test_loss_mae / len(test_loader)
    test_loss_rmse1= test_loss_rmse / len(test_loader)
    test_losses_mse.append(test_loss_mse1)  # 计算的是平均损失
    test_losses_mae.append(test_loss_mae1)
    test_losses_rmse.append(test_loss_rmse1)

    # 找到最优测试损失及其所在轮次
    best_mse_test_loss = min(test_losses_mse)
    best_mae_test_loss = min(test_losses_mae)
    best_rmse_test_loss = min(test_losses_rmse)
    best_test_loss_epoch_mse = test_losses_mse.index(best_mse_test_loss) + 1
    best_test_loss_epoch_mae = test_losses_mae.index(best_mae_test_loss) + 1
    best_test_loss_epoch_rmse = test_losses_rmse.index(best_rmse_test_loss) + 1

    # 输出最佳训练损失及其所在轮次
print("==========LSTM+Attention============")
print("Best training MSE loss:", best_mse_train_loss)
print("Epoch of best training MSE loss:", best_train_loss_epoch_mse)
print("Best testing MSE loss:", best_mse_test_loss)
print("Epoch of best testing MSE loss:", best_test_loss_epoch_mse)
print('==========================')
print("Best training MAE loss:", best_mae_train_loss)
print("Epoch of best training MAE loss:", best_train_loss_epoch_mae)
print("Best testing MAE loss:", best_mae_test_loss)
print("Epoch of best testing MAE loss:", best_test_loss_epoch_mae)
print("================================")
print("Best training RMSE loss:", best_rmse_train_loss)
print("Epoch of best training RMSE loss:", best_train_loss_epoch_rmse)
print("Best testing RMSE loss:", best_rmse_test_loss)
print("Epoch of best testing RMSE loss:", best_test_loss_epoch_rmse)

# 9. 绘制结果
plot_size = 200
plt.figure(figsize=(12, 8))
plt.plot((best_model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1), "b", label="Predicted")
plt.plot(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size], "r", label="Actual")
plt.legend(fontsize=20)  # Set the font size for the legend
plt.xticks(fontsize=24)  # Set the font size for the x-axis ticks
plt.yticks(fontsize=24)
plt.xlabel('Date',fontsize=24)
plt.ylabel('Motion Time(m)',fontsize=24)
plt.title("LSTM+Attention Training Data: Predicted vs Actual", fontsize=24)
plt.savefig("LSTM+Attention training_result.png")
plt.show()

y_test_pred = best_model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(y_test_pred.detach().numpy()[: plot_size], "b", label="Predicted")
plt.plot(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size], "r", label="Actual")
plt.legend(fontsize=20)  # Set the font size for the legend
plt.xticks(fontsize=24)  # Set the font size for the x-axis ticks
plt.yticks(fontsize=24)
plt.xlabel('Date',fontsize=24)
plt.ylabel('Motion Time(m)',fontsize=24)
plt.title("LSTM+Attention Testing Data: Predicted vs Actual", fontsize=24)
plt.savefig("LSTM+Attention testing_result.png")
plt.show()

# 绘制mse损失变化图
plt.figure(figsize=(12, 8))
plt.plot(range(1, Config.epochs + 1), train_losses_mse, label='Training MSE Loss', marker='o')
plt.plot(range(1, Config.epochs + 1), test_losses_mse, label='Testing MSE Loss', marker='o')
plt.xlabel('Epoch',fontsize=24)
plt.ylabel('MSE Loss',fontsize=24)
plt.xticks(fontsize=24)  # Set the font size for the x-axis ticks
plt.yticks(fontsize=24)
plt.title('LSTM+Attention Training and Testing MSE Loss vs Epoch', fontsize=24)
plt.legend(fontsize=20)  # Set the font size for the legend
plt.grid(True)
plt.savefig("LSTM+Attention mse_loss_vs_epoch.png")  # Save the loss vs epoch plot
plt.show()
# 绘制mae损失变化图
plt.figure(figsize=(12, 8))
plt.plot(range(1, Config.epochs + 1), train_losses_mae, label='Training MAE Loss', marker='o')
plt.plot(range(1, Config.epochs + 1), test_losses_mae, label='Testing MAE Loss', marker='o')
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('MAE Loss', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title('LSTM+Attention Training and Testing MAE Loss vs Epoch', fontsize=24)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("LSTM+Attention mae_loss_vs_epoch.png")
plt.show()
# 绘制rmse损失变化图
plt.figure(figsize=(12, 8))
plt.plot(range(1, Config.epochs + 1), train_losses_rmse, label='Training RMSE Loss', marker='o')
plt.plot(range(1, Config.epochs + 1), test_losses_rmse, label='Testing RMSE Loss', marker='o')
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('RMSE Loss', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title('LSTM+Attention Training and Testing RMSE Loss vs Epoch', fontsize=24)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("LSTM+Attention rmse_loss_vs_epoch.png")
plt.show()