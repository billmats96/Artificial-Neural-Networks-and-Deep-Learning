%% Artificial Neural Networks : Part 1.4 
% Author: Vasileios Matsoukas
%

close all
clear
load('Data_Problem1_regression.mat');

%% Construct new non-linear function

d1 = 8;
d2 = 7;
d3 = 6;
d4 = 6;
d5 = 4;
D = [d1; d2; d3; d4; d5]; 
T = [T1 T2 T3 T4 T5]; 

Tnew = T*D/ sum(D);

%% Build Feedforward Neural Network
%% A) steepest descent.

hidden_neurons = 100;
train_algorithm = 'traingd';
epochs = 1000;
P = con2seq([X1 X2]); T = con2seq(Tnew); % convert the data to a useful format

% Initialize neural network
net = feedforwardnet(hidden_neurons, train_algorithm);
net.trainParam.epochs = epochs;

% Sample without replacement of 3000 datapoints
% Will be used for training, validation, test sets
[sample, idx_sample] = datasample(Tnew, 3000);
idx_sample = sort(idx_sample);
Tnew_new = Tnew(idx_sample);
X1_new = X1(idx_sample);
X2_new = X2(idx_sample);

% Determine division of data in the neural network.
net.divideFcn = 'dividerand';    
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;

% Train the network with the new data
[net, tr_descr] = train(net, [X1_new X2_new]', Tnew_new');   

% Test the network on all data
% a = sim(net, [X1 X2]');
% postregm(a, Tnew');

Y1 = sim(net, [X1_new(cell2mat(tr_descr.testMask) == 1) X2_new(cell2mat(tr_descr.testMask) == 1)]');
% postregm(a, Tnew');

figure(10);
subplot(121);
plot(1:1000,Y1,'bx',1:1000,Tnew_new(cell2mat(tr_descr.testMask) == 1),'r'); % plot the sine function and the output of the networks
title('Gradient Descent');
legend('target','traingd','Location','north');



%% B) levenberg-marquardt
hidden_neurons = 80 ;
train_algorithm = 'trainlm';
epochs = 50 ;
P = con2seq([X1 X2]); T = con2seq(Tnew); % convert the data to a useful format

% Initialize neural network
net = feedforwardnet(hidden_neurons, train_algorithm);
net.trainParam.epochs = epochs;

% Sample without replacement of 3000 datapoints
% Will be used for training, validation, test sets
[sample, idx_sample] = datasample(Tnew, 3000);
idx_sample = sort(idx_sample);
Tnew_new = Tnew(idx_sample);
X1_new = X1(idx_sample);
X2_new = X2(idx_sample);

% Determine division of data in the neural network.
net.divideFcn = 'dividerand'; %dividerand  
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;

% Train the network with the new data
[net, tr_descr] = train(net, [X1_new X2_new]', Tnew_new');   

figure; 
subplot(141);plot(Tnew_new);
subplot(142);plot(Tnew_new(1:1000));
subplot(143);plot(Tnew_new(1001:2000));
subplot(144);plot(Tnew_new(2001:3000));

figure; 
subplot(141);plot(Tnew_new); title('Input Data')
subplot(142);plot(Tnew_new(cell2mat(tr_descr.trainMask) == 1)); title('Training Data')
subplot(143);plot(Tnew_new(cell2mat(tr_descr.valMask) == 1));title('Validation Data')
subplot(144);plot(Tnew_new(cell2mat(tr_descr.testMask) == 1));title('Test Data');



% Test the network on all data
Y2 = sim(net, [X1_new(cell2mat(tr_descr.testMask) == 1) X2_new(cell2mat(tr_descr.testMask) == 1)]');
% postregm(a, Tnew');

figure(10);
subplot(122);
plot(1:1000,Y2,'bx',1:1000,Tnew_new(cell2mat(tr_descr.testMask) == 1),'r'); % plot the sine function and the output of the networks
title('Levenberg-Marquardt');
legend('target','trainlm','Location','north');



