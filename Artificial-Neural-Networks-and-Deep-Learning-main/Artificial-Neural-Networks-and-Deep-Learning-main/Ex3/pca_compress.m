function rec_error = pca_compress(data, num_of_pcs)

[coeff, score, ~, ~, ~, mu] = pca(data);

rec_data = score(:,1:num_of_pcs) * coeff(:,1:num_of_pcs)' + repmat(mu, size(data,1), 1);

rec_error = sqrt(mean(mean((data-rec_data).^2)));

end