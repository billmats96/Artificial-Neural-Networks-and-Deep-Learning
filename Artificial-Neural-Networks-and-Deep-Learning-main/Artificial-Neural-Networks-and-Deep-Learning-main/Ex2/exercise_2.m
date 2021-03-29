%% Exercise 2 : Recurrent Neural Networks
clc
clear
close all

%% 1st question
T = [1 1; -1 -1; 1 -1]'; 
net = newhop(T); 

plot(T(1,:),T(2,:),'r*')
axis([-1.6 1.6 -1.6 1.6])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');

a = {rands(2,1)};
[y,Pf,Af] = sim(net,{1 50},{},a);

record = [cell2mat(a) cell2mat(y)];
start = cell2mat(a);
hold on
plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))


color = 'rgbmy';
for i=1:50
   a = {1.5*rands(2,1)};
   [y,Pf,Af] = sim(net,{1 50},{},a);
   record=[cell2mat(a) cell2mat(y)];
   start=cell2mat(a);
   plot(start(1,1),start(2,1),'kx',record(1,:),record(2,:),color(rem(i,5)+1))
   drawnow
   for j = 1:50
       H = 0;
       if norm(y{j},1) == 2
           stepzz(i) = j;
           H = 1;
       end 
       if H == 1 
           break;
       end
   end 
end
% average steps for a point to converge is ceil(sum(stepzz))/length(stepzz)
hold off

%% 2nd question 

figure(2);
subplot(121);
T = [1 1; -1 -1; 1 -1]';
plot(T(1,:),T(2,:),'r*')
net = newhop(T);
n=20;
hold on
for i=1:n
    a={2*rands(2,1)};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end

symmetry_points = [ 0 -1 0 1 1.5 0.5 -1.5 -0.5 0 0 0 0 0 -1;...
                    0 1 -1 0 0 0 0 0 -1.25 -0.75 1.25 0.75 1 0];

for i = 1:length(symmetry_points(1,:))
    aa = symmetry_points(:,i);
    [y,Pf,Af] = sim(net,{1 50},{},aa);       % simulation of the network  for 50 timesteps
    record=[aa cell2mat(y)];       % formatting results
    start=aa;                      % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end

legend('real attractor','initial state','time evolution','final points','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
xlabel('a(1)');
ylabel('a(2)');

%% 3rd question 

figure(2);
subplot(122);
T = [1 1 1; -1 -1 1; 1 -1 -1]';
plot3(T(1,:),T(2,:),T(3,:),'r*')
net = newhop(T);
n=10;
hold on
for i=1:n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end
grid on;

symmetry_points_3 = [ 0 1 0 ; ...
                      0 -1 0;...
                      0 1 1];

for i = 1:length(symmetry_points_3(1,:))
    aa = symmetry_points_3(:,i);
    [y,Pf,Af] = sim(net,{1 50},{},aa);       % simulation of the network  for 50 timesteps
    record=[aa cell2mat(y)];       % formatting results
    start=aa;                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r'); % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end


legend('real attractor','initial state','time evolution','final points','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');
xlabel('a(1)');
ylabel('a(2)');
zlabel('a(3)');

%% 4th question

hopdigit_v2(3, 60, 'Reconstruction success with noise level = 3 and 60 iterations' );

hopdigit_v2(6, 600, 'Reconstruction fail with noise level = 6 and 600 iterations');

%% Time series prediction with neural network training 
 
train_data = load('lasertrain.dat');
test_data = load('laserpred.dat');

mu = mean(train_data);
sig = std(train_data);

train_data_scaled = (train_data - mu) / sig;
test_data_scaled = (test_data - mu) / sig;


% Train a MLP (feedforward mode)
hidden_neurons = 5; %10
train_algorithm = 'trainlm';
epochs = 50 ;
lag = 10; %20
[P, T] = getTimeSeriesTrainData(train_data_scaled, lag);

% P = 1:length(train_data); T = train_data'; % convert the data to a useful format

% Initialize neural network
net = feedforwardnet(hidden_neurons, train_algorithm);
net.trainParam.epochs = epochs;
net.trainParam.showWindow=1;

[net, tr_descr] = train(net, P, T);   

y_hat = [];
new_train_data= train_data_scaled;

for i = 1:length(test_data_scaled)           
    
    y_hat(i) = sim(net, new_train_data(end-lag+1:end));
    new_train_data = [new_train_data; y_hat(i)];
end

% add again std + mean
y_hat = sig*y_hat + mu;

% % figure;
% % plot(1:length(test_data),test_data,'bx',1:length(test_data),y_hat,'r');
% % title('Prediction Visualization');
% % legend('target','predicted','Location','north');
% % euclid_dist = norm(y_hat-test_data,2);
% % fprintf('Euclidean distance is %f\n',euclid_dist);
% % mse = mse(y_hat,test_data);
% % fprintf('Mean Square Error is %f\n',mse);
% % rmse = sqrt(mean((y_hat-test_data).^2))
% 
figure;
subplot(2,1,1)
plot(test_data)
hold on
plot(y_hat,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Laser Value")
title("Forecast with RNN")

subplot(2,1,2)
stem(y_hat' - test_data)
xlabel("Time Step")
ylabel("Error")
rmse = sqrt(mean((y_hat'-test_data).^2));
title("RMSE = " + rmse)

%% LSTM Network
train_data = load('lasertrain.dat');
test_data = load('laserpred.dat');

mu = mean(train_data);
sig = std(train_data);

train_data_scaled = (train_data - mu) / sig;
test_data_scaled = (test_data - mu) / sig;

dataTrain = train_data_scaled;
dataTest = test_data_scaled;

XTrain = dataTrain(1:end-1)';
YTrain = dataTrain(2:end)';
numTimeStepsTrain = numel(XTrain);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 50;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% Best so far : Epochs 120, Learn 0.004 Drop at 80, Factor 0.8, Neurns 50
options = trainingOptions('rmsprop', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',120, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.004, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',80, ...
    'LearnRateDropFactor',0.8, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);
XTest = dataTest(1:end)';

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','gpu');
end

YPred = sig*YPred + mu;

YTest = test_data(1:end)';
rmse = sqrt(mean((YPred-YTest).^2));

figure
plot(train_data(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[train_data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time Step")
ylabel("Laser Value")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Laser Value")
title("Forecast with LSTM Network")
 
subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time Step")
ylabel("Error")
title("RMSE = " + rmse)
