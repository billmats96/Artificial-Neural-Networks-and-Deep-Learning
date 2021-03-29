%% Artificial Neural Networks and Deep Learning : Exercise 1 Part1
% Author : Vasileios Matsoukas

close all
clear

%% Classification with a 2-Input Perceptron
% A 2-input hard limit neuron is trained to classify 5 input vectors into two
% categories.

%%

X = [ -0.5 -0.5 +0.3 -0.1;  ... % define input points 
      -0.5 +0.5 -0.5 +1.0];
T = [1 0 0 1]; % define tha class attribute of input points (can either be 0 or 1)

figure(1);
plotpv(X,T); % visualize input points with their class attribute

%%
% The perceptron must properly classify the 4 input vectors in X into the two
% categories defined by T.  Perceptrons have HARDLIM neurons.  These neurons are
% capable of separating an input space with a straight line into two categories
% (0 and 1).
%
% Here PERCEPTRON creates a new neural network with a single neuron. The
% network is then configured to the data, so we can examine its
% initial weight and bias values. (Normally the configuration step can be
% skipped as it is automatically done by ADAPT or TRAIN.)

net = perceptron;
net = configure(net,X,T);

%%
% The input vectors are replotted with the neuron's initial attempt at
% classification.
%
% The initial weights are set to zero, so any input gives the same output and
% the classification line does not even appear on the plot.  Fear not... we are
% going to train it!
figure(2);
subplot(121);
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
title('Training the Perceptron - Linearly Non-Separable data')
%%
% Here the input and target data are converted to sequential data (cell
% array where each column indicates a timestep) and copied three times
% to form the series XX and TT.
%
% ADAPT updates the network for each timestep in the series and returns
% a new network object that performs as a better classifier.

XX = repmat(con2seq(X),1,3);
TT = repmat(con2seq(T),1,3);
net = adapt(net,XX,TT);
plotpc(net.IW{1},net.b{1});

%%
% Now SIM is used to classify any other input vector, like [0.7; 1.2]. A plot of
% this new point with the original training set shows how the network performs.
% To distinguish it from the training set, color it red.
figure(2);
subplot(122);
x = [-0.1 0.5 0.6 -0.2 -0.4 -0.6; -0.2 0.8 0.9 -0.7 0.4 -0.4];
    
for i = 2:size(x,2) 
%     y = net(x(:,i));
%     plotpv(x(:,i),y);
%     point = findobj(gca,'type','line');
%     point.Color = 'red';
end

y = net(x);
plotpv(x,y);
% point = findobj(gca,'type','line');
% point.Color = 'red';

%%
% Turn on "hold" so the previous plot is not erased and plot the training set
% and the classification line.
%
% The perceptron correctly classified our new point (in red) as category "zero"
% (represented by a circle) and not a "one" (represented by a plus).

hold on;
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
title('Classification behaviour of the trained NN')
hold off;


