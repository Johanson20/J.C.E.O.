function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

n = size(X, 2);
mu = zeros(1,n);
sigma = zeros(1,n);
X_norm = zeros(size(X));
for i=1:n
    mu(i) = mean(X(:, i));
    sigma(i) = std(X(:, i));
    r = max(X(:,i)) - min(X(:,i));
    X_norm(:, i) = (X(:,i) - mu(i))/r;
end

end
