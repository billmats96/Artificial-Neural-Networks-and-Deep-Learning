x1 = (1:1:10)';
x2 = (2:2:20)';

x1 = [2.5 0.5 2.2 1.9 3.1 2.3 2 1 1.5 1.1]';
x2 = [2.4 0.7 2.9 2.2 3.0 2.7 1.6 1.1 1.6 0.9]';
x = [ x1 x2 ] ;

mean_x = mean(x);

x = x - mean(x);

covMat = cov(x);

[eigVecs, eigVals] = eigs(covMat, 2); 

RowFeatureVector = eigVecs(:,1)';
RowZeroMeanData = transpose(x);

z = RowFeatureVector * RowZeroMeanData;

x_hat = (RowFeatureVector' * z)';


coeff = pca(x);
features = x * coeff;

features = features(:,1);

