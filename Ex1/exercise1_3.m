clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
x=0:0.05:6*pi; y=sin(x.^2) + 0.1*randn(1, length(x));
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks

net1=feedforwardnet(50,'traingd');
net2=feedforwardnet(50,'traingda');
net3=feedforwardnet(50,'traincgf');
net4=feedforwardnet(50,'traincgp');
net5=feedforwardnet(50,'trainbfg');
net6=feedforwardnet(50,'trainlm');

net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

net3.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};

net4.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};

net5.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net5.lw{2,1}=net1.lw{2,1};
net5.b{1}=net1.b{1};
net5.b{2}=net1.b{2};

net6.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net6.lw{2,1}=net1.lw{2,1};
net6.b{1}=net1.b{1};
net6.b{2}=net1.b{2};


%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training 
net2.trainParam.epochs=1;
net3.trainParam.epochs=1;
net4.trainParam.epochs=1;
net5.trainParam.epochs=1;
net6.trainParam.epochs=1;

[net1, tr_descr11]=train(net1,p,t);   % train the networks
[net2, tr_descr21]=train(net2,p,t);
[net3, tr_descr31]=train(net3,p,t);
[net4, tr_descr41]=train(net4,p,t);
[net5, tr_descr51]=train(net5,p,t);
[net6, tr_descr61]=train(net6,p,t);

a11=sim(net1,p); a21=sim(net2,p); a31=sim(net3,p); a41=sim(net4,p); a51=sim(net5,p); a61=sim(net6,p);  % simulate the networks with the input vector p
[~, ~, r11] = postregMODIFIED(cell2mat(a11),y);
[~, ~, r21] = postregMODIFIED(cell2mat(a21),y);
[~, ~, r31] = postregMODIFIED(cell2mat(a31),y);
[~, ~, r41] = postregMODIFIED(cell2mat(a41),y);
[~, ~, r51] = postregMODIFIED(cell2mat(a51),y);
[~, ~, r61] = postregMODIFIED(cell2mat(a61),y);


net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net3.trainParam.epochs=14;
net4.trainParam.epochs=14;
net5.trainParam.epochs=14;
net6.trainParam.epochs=14;

[net1, tr_descr12]=train(net1,p,t);   % train the networks
[net2, tr_descr22]=train(net2,p,t);
[net3, tr_descr32]=train(net3,p,t);
[net4, tr_descr42]=train(net4,p,t);
[net5, tr_descr52]=train(net5,p,t);
[net6, tr_descr62]=train(net6,p,t);
a12=sim(net1,p); a22=sim(net2,p); a32=sim(net3,p); a42=sim(net4,p); a52=sim(net5,p); a62=sim(net6,p);  % simulate the networks with the input vector p

[~, ~, r12] = postregMODIFIED(cell2mat(a12),y);
[~, ~, r22] = postregMODIFIED(cell2mat(a22),y);
[~, ~, r32] = postregMODIFIED(cell2mat(a32),y);
[~, ~, r42] = postregMODIFIED(cell2mat(a42),y);
[~, ~, r52] = postregMODIFIED(cell2mat(a52),y);
[~, ~, r62] = postregMODIFIED(cell2mat(a62),y);

net1.trainParam.epochs=985;
net2.trainParam.epochs=985;
net3.trainParam.epochs=985;
net4.trainParam.epochs=985;
net5.trainParam.epochs=985;
net6.trainParam.epochs=985;

[net1, tr_descr13]=train(net1,p,t);   % train the networks
[net2, tr_descr23]=train(net2,p,t);
[net3, tr_descr33]=train(net3,p,t);
[net4, tr_descr43]=train(net4,p,t);
[net5, tr_descr53]=train(net5,p,t);
[net6, tr_descr63]=train(net6,p,t);

a13=sim(net1,p); a23=sim(net2,p); a33=sim(net3,p); a43=sim(net4,p); a53=sim(net5,p); a63=sim(net6,p);  % simulate the networks with the input vector p
[~, ~, r13] = postregMODIFIED(cell2mat(a13),y);
[~, ~, r23] = postregMODIFIED(cell2mat(a23),y);
[~, ~, r33] = postregMODIFIED(cell2mat(a33),y);
[~, ~, r43] = postregMODIFIED(cell2mat(a43),y);
[~, ~, r53] = postregMODIFIED(cell2mat(a53),y);
[~, ~, r63] = postregMODIFIED(cell2mat(a63),y);
