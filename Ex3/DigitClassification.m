clear 
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
%rng('default')


% Load the training data into memory
load('digittrain_dataset.mat');

%% Stacked AutoEncoder : Layer 1
hiddenSize1 = 200;
tic;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',120, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

a(1) = toc;
figure;
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

% % Layer 2
% hiddenSize2 = 100;
% tic;
% autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% 
% feat2 = encode(autoenc2,feat1);
% a(2) = toc;

% % Layer 3.1
% 
% hiddenSize3 = 40;
% tic;
% autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.001, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.06, ...
%     'ScaleData', false);
% 
% feat3 = encode(autoenc3,feat2);
% a(3) = toc;
% 
% % Layer 3.2
% 
% hiddenSize4 = 20;
% tic;
% autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.001, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.06, ...
%     'ScaleData', false);
% 
% feat4 = encode(autoenc4,feat3);
% a(4) = toc;
% 
% 
% Layer 3
tic;
softnet = trainSoftmaxLayer(feat1,tTrain,'MaxEpochs',400);
a(2) = toc;

% Deep Net
deepnet = stack(autoenc1,softnet);


% Test deep net
imageWidth = 28;
imageHeight = 28; 
inputSize = imageWidth*imageHeight;
load('digittest_dataset.mat');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
classAcc=100*(1-confusion(tTest,y))


% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
tic;
deepnet = train(deepnet,xTrain,tTrain);
a(3) = toc;
y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
classAcc=100*(1-confusion(tTest,y))
view(deepnet)

%% Compare with normal neural network (1 hidden layers)
% net = patternnet(50);
% tic;
% net=train(net,xTrain,tTrain);
% a(1) = toc;
% y=net(xTest);
% plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% view(net)

%% Compare with normal neural network (2 hidden layers)
% net = patternnet([300 100 40 20]);
% tic;
% net=train(net,xTrain,tTrain);
% a(2) = toc;
% y=net(xTest);
% plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% view(net)