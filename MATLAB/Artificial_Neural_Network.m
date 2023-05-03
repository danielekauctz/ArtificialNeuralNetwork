%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Artificial Neural Network (ANN)
%    using feedforwardnet
% 
% Example by Daniele Kauctz Monteiro (2023)
% danielekauctz@hotmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters:
% data: training dataset
% X: input (initial data for the neural network)
% y: output (result for given inputs)
% model: Artificial Neural Network

clear
close
clc

%% TRAINING DATA
data = importdata('data.txt');

X = [data(:,1) data(:,3)];
y = data(:,2);

%% NEURAL NETWORK
hiddenLayerSize = 5;
model = feedforwardnet(hiddenLayerSize);

model.divideParam.trainRatio = 0.8;
model.divideParam.valRatio = 0.1;
model.divideParam.testRatio = 0.1;

% network parameters 
model.trainFcn = 'trainlm';         % Levenberg-Marquardt
model.trainParam.epochs = 1000;     % Maximum limit of the network training
model.trainParam.goal = 1e-8;       % Stopping criterion based on error (mse) goal
model.trainParam.max_fail = 10;     % Maximum validation failures
model.trainParam.min_grad = 1e-8;   % Minimum performance gradient

[model,tr] = train(model, X',y');

%% RESULTS ANALYSIS
expected_y = y(tr.trainInd)';
predicted_y = model(X(tr.trainInd,:)');

% expected_y = y(tr.valInd)';
% predicted_y = model(X(tr.valInd)');

n = length(expected_y);

mean_squared_error = (sum((expected_y-predicted_y).^2))/n;

%% PLOT REGRESSION
figure(1)
plotregression(expected_y,predicted_y)
title('\fontsize{14} \bf Training regression')

%% PLOT PERFORMANCE
figure(2)
plotperform(tr)

%% PREDICTION TEST
X_predict = [0.5758 0.04];          %  true y = 0.5838
y_predict = model(X_predict');