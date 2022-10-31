clear all;
clc;
tic
%% 数据输入
% 数据输入
f = xlsread('历史流量数据下载.xlsx','SheetJS','B1:B1908');  % 读取excel数据、load函数读取txt、mat文件
figure(1)
plot(f)
%序列的前1908个用于训练，预测未来194个。
dataTrain=f(1:end,:);

%% LSTM预测
%% 标准化数据
%为了获得较好的拟合并防止训练发散，将训练数据标准化为具有零均值和单位方差。
%在预测时，您必须使用与训练数据相同的参数来标准化测试数据。
mu = mean(dataTrain);
sig = std(dataTrain);
 
dataTrainStandardized = (dataTrain - mu) / sig;
%% 准备预测变量和响应
%要预测序列在将来时间步的值，请将响应指定为将值移位了一个时间步的训练序列。
%也就是说，在输入序列的每个时间步，LSTM 网络都学习预测下一个时间步的值。
%预测变量是没有最终时间步的训练序列。
X = dataTrainStandardized(1:end-1);
Y = dataTrainStandardized(2:end);
XTrain = X';
YTrain = Y';
%% 定义 LSTM 网络架构
%创建 LSTM 回归网络。指定 LSTM 层有 10 个隐含单元
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 30;% 如果对预测结果不太满意，可以将其略微增大，过大会过拟合
 
layers = layerGraph();  
TempLayers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, "Name", "lstm")                % LSTM层
    reluLayer("Name","relu")   % 常用学习函数：tanh、sigmoid、relu
    fullyConnectedLayer(50, "Name", "fc1")
    dropoutLayer(0.1)
    fullyConnectedLayer(numResponses, "Name", "fc")          % 全连接层(可以理解为单层BP，作用为整合结果)
    regressionLayer("Name", "regressionoutput")];  
layers = addLayers(layers, TempLayers);

%指定训练选项。
%将求解器设置为 'adam' 并进行 50 轮训练。（求解器不是学习函数）
%要防止梯度爆炸，请将梯度阈值设置为 1。
%指定初始学习率 0.01，在 25 轮训练后通过乘以因子 0.05 来降低学习率。
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',180, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',0, ...
    'Plots','training-progress');%如果不想画迭代图，就把这行删掉（括号注释）
%% 训练 LSTM 网络
%使用 trainNetwork 以指定的训练选项训练 LSTM 网络。
net = trainNetwork(XTrain,YTrain,layers,options);
%% 预测将来时间步
%要预测将来多个时间步的值，请使用 predictAndUpdateState 函数一次预测一个时间步，并在每次预测时更新网络状态。对于每次预测，使用前一次预测作为函数的输入。
%使用与训练数据相同的参数来标准化测试数据。

%要初始化网络状态，请先对训练数据 XTrain 进行预测。
%接下来，使用训练响应的最后一个时间步 YTrain(end) 进行第一次预测。
%循环其余预测并将前一次预测输入到 predictAndUpdateState。
%对于大型数据集合、长序列或大型网络，在 GPU 上进行预测计算通常比在 CPU 上快。
%其他情况下，在 CPU 上进行预测计算通常更快。
%对于单时间步预测，请使用 CPU。
%使用 CPU 进行预测，请将 predictAndUpdateState 的 'ExecutionEnvironment' 选项设置为 'cpu'。
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
 
for i = 2:194
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%使用先前计算的参数对预测去标准化。
YPred_sum = sig*YPred + mu;

%使用预测值绘制训练时序。
figure
plot(dataTrain(1:end))
hold on
idx = 1909:(1909+194);
plot(idx,[f(1908) YPred_sum],'.-')
hold off
xlabel("days")
ylabel("mm")
xlim([0 1909+195])
title("Forecast")
legend(["Observed" "Forecast"])

%将预测值与测试数据进行比较。
figure
plot(YPred_sum,'.-')
hold off
legend(["Forecast"])
ylabel("data")
title("Forecast")
 
toc