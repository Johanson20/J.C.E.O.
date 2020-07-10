function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


rawJ = (1/2)*(X*Theta' - Y).^2;
J = sum(sum(rawJ.*R)) + (lambda/2)*(sum(sum(X.^2)) + sum(sum(Theta.^2)));

X_g = (X*Theta' - Y).*R;
X_grad = X_g*Theta + lambda*X;

Thetag = (X*Theta' - Y).*R;
Theta_grad = Thetag'*X + lambda*Theta;


grad = [X_grad(:); Theta_grad(:)];

end
