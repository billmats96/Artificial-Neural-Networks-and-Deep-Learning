%% Artificial Neural Networks and Deep Learning 
% Exercise 3 : Deep Feature Learning

close all
clear 

%% Section 1 : PCA 

x = randn(50, 500)';

mean_x = mean(x);

% a)subtract the mean value
x = x - mean_x; 

% b)calculate the covariance matrix of the zero mean dataset
covMat_p = cov(x); 

% c)calculate the eigenvectors and eigenvalues of this covariance matrix
[eigVecs, eigVals] = eig(covMat_p); 

% d) Calculate the quality of reconstruction
information_gain_x = zeros(1, 20);
dim_reduce_x = zeros(1, 20);
reconstruction_error_x = zeros(1,20);

for i = 1:20
    dim_reduce_x(i) = ceil((21-i)/20 * size(x, 2));
    [E, eigenvals] = eigs(covMat_p, dim_reduce_x(i));
    information_gain_x(i) = sum(diag(eigenvals))/sum(diag(eigVals));
    z = E(:,1:dim_reduce_x(i))' * transpose(x);
    x_hat = ( E(:,1:dim_reduce_x(i)) * z)';
    %z = x*E'; % dimensionality reduction
    %xhat = (E * z)' ; % reconstruction
    %xhat = xhat + mean_x ;
    reconstruction_error_x(i) = sqrt(mean(mean((x-x_hat).^2)));
end

figure;
bar(dim_reduce_x', information_gain_x');
xlabel('Number of reduced components')
ylabel('Information Gain')
title('');

choles = load('choles_all');
p = choles.p';
mean_p = mean(p);
p = p - mean_p;
covMat_p = cov(p);
[eigVecs_p, eigVals_p] = eig(covMat_p);

information_gain_choles = zeros(1, 20);
dim_reduce_choles = zeros(1, 20);
reconstruction_error_p = zeros(1,20);
for i = 1:20
    dim_reduce_choles(i) = ceil((21-i)/21 * size(p, 2));
    [E, eigenvals_p] = eigs(covMat_p, dim_reduce_choles(i));
    information_gain_choles(i) = sum(diag(eigenvals_p))/sum(diag(eigVals_p));
    z = E(:,1:dim_reduce_choles(i))' * transpose(p);
    p_hat = ( E(:,1:dim_reduce_choles(i)) * z)';
    %z = E' * p' ;
    %phat = (E * z)' ;
    %phat = phat + mean_p;
    reconstruction_error_p(i) = sqrt(mean(mean((p-p_hat).^2)));
end

figure;
g = suptitle({['PCA Analysis'],' ',' '});
set(g, 'FontSize', 12, 'FontWeight', 'bold')
title('PCA Analysis')
subplot(121)

plot(dim_reduce_x, information_gain_x, 'b-*')
title('Information Gain on Random Data')
ylabel('Information Gain')
xlabel('Number of Reduced Components')

subplot(122)
plot(dim_reduce_choles, information_gain_choles, 'r-*')
title('Information Gain on Highly Corellated Data')
xlabel('Number of Reduced Components')
%check also corr(p) to justify high correlation


% Functions mapstd, processpca
x = x + mean_x;
x = mapstd(x);
maxfrac = 0.01;
[z1, PS] =  processpca(x, maxfrac);
x1_hat = processpca('reverse',z1,PS);


%% Section 1.2 Handwritten Digits and PCA

load -ASCII threes;

colormap('gray')
% i = 500;
% imagesc(reshape(threes(i,:),16,16),[0,1]);

mean_three = mean(threes);
imagesc(reshape(mean_three,16,16),[0,1]);

covMat_three = cov(threes);

[eigVecs_three, eigVals_three] = eig(covMat_three); 
figure;
plot(diag(eigVals_three),'b-*'); 
ylabel('EigenValue Value'); 
xlabel('EigenValue Index');
grid on; 
grid minor


[coeff, score, latent, tsquared, explained, mu] = pca(threes);
z_three = threes * coeff;
z_three_1 = z_three(:,1);
z_three_2 = z_three(:,1:2);
z_three_3 = z_three(:,1:3);
z_three_4 = z_three(:,1:256);

total_rec = score(:,1:256) * coeff(:,1:256)' + repmat(mu, 500, 1);

% z_three_1_rec = z_three_1 * coeff(1,:);
% z_three_2_rec = z_three_2 * coeff(1:2,:);
% z_three_3_rec = z_three_3 * coeff(1:3,:);
% z_three_4_rec = z_three_4 * coeff(1:256,:);

z_three_1_rec = score(:,1) * coeff(:,1)' + repmat(mu, 500, 1);
z_three_2_rec = score(:,1:2) * coeff(:,1:2)' + repmat(mu, 500, 1);
z_three_3_rec = score(:,1:3) * coeff(:,1:3)' + repmat(mu, 500, 1);
z_three_4_rec = score(:,1:4) * coeff(:,1:4)' + repmat(mu, 500, 1);


figure; 
colormap('gray')
subplot(221);
imagesc(reshape(mean(z_three_1_rec),16,16),[0,1]);
title('Reconstruction from 1 PC')

subplot(222);
imagesc(reshape(mean(z_three_2_rec),16,16),[0,1]);
title('Reconstruction from 2 PCs')

subplot(223);
imagesc(reshape(mean(z_three_3_rec),16,16),[0,1]);
title('Reconstruction from 3 PCs')

subplot(224);
imagesc(reshape(mean(z_three_4_rec),16,16),[0,1]);
title('Reconstruction from 4 PCs')

rec_error = zeros(1,50);

for k = 1:50
    rec_error(k) = pca_compress(threes,k);
end

rec_256 = pca_compress(threes, 256);

figure;
subplot(131)
[eigVecs_three, eigVals_three] = eig(covMat_three); 
plot(diag(eigVals_three),'b'); 
ylabel('Amplitude'); 
xlabel('Index');
title('Covariance Matrix Eigenvalues')
grid on; 
grid minor

subplot(132)
plot(rec_error,'r-*') 
xlabel('Number Of PCs used for reconstruction')
ylabel('Reconstruction error')
title('Dataset Reconstruction') 
grid on; 
grid minor
xlim([0.9 50])

a = sort(diag(eigVals_three), 'descend');
eig_sum = cumsum(a);

subplot(133)
plot(eig_sum((1:50)),'b-*') 
xlabel('Number of most important egeinvalues')
ylabel('Cumulative sum of eigenvalues')
title('Dataset Reconstruction') 
grid on; 
grid minor
hold on 
plot(ones(1,50)*eig_sum(end));
legend('Cumulative Sum','Total Sum')
xlim([0.9 50])



