%% Artificial Neural Network Part 1.5
% Training with trainbr

clear
clc
close all

%% Dataset no.2

% Generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks

net2 = feedforwardnet(50,'trainbr');

%training and simulation
net2.trainParam.epochs=1;  % set the number of epochs for the training 

[net2, tr_descr21]=train(net2,p,t);   % train the networks

a21 = sim(net2,p); 
[~, ~, r21] = postregMODIFIED(cell2mat(a21),y);

net2.trainParam.epochs = 14;

[net2, tr_descr22]=train(net2,p,t);   % train the networks

a22=sim(net2,p); 
[~, ~, r22] = postregMODIFIED(cell2mat(a22),y);

net2.trainParam.epochs=985;

[net2, tr_descr23]=train(net2,p,t);   % train the networks

a23=sim(net2,p);

[~, ~, r23] = postregMODIFIED(cell2mat(a23),y);

figure(10);
subplot(121);
plot(x,y,'bx',x,(cell2mat(a23)),'r'); % plot the sine function and the output of the networks
title('Bayesian Regularization Noise-Less Data');
legend('target','trainbr','Location','north');

%% Dataset no. 3

% Generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2) + 0.1*randn(1,length(x));
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks

net3 = feedforwardnet(50,'trainbr');

%training and simulation
net3.trainParam.epochs=1;  % set the number of epochs for the training 

[net3, tr_descr31]=train(net3,p,t);   % train the networks

a31=sim(net3,p); 
[~, ~, r31] = postregMODIFIED(cell2mat(a31),y);

net3.trainParam.epochs=10;

[net3, tr_descr32]=train(net3,p,t);   % train the networks

a32=sim(net3,p); 
[~, ~, r32] = postregMODIFIED(cell2mat(a32),y);

net3.trainParam.epochs=985;

[net3, tr_descr33]=train(net3,p,t);   % train the networks

a33=sim(net3,p);

[~, ~, r33] = postregMODIFIED(cell2mat(a33),y);

figure(10);
subplot(122);
plot(x,y,'bx',x,(cell2mat(a32)),'r'); % plot the sine function and the output of the networks
title('Bayesian Regularization - Noisy Data');
legend('target','trainbr','Location','north');

