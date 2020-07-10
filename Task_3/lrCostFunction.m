function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h_x = sigmoid(X*theta);
thet = theta;
thet(1) = 0;
J = sum((1/m)*(-y.*log(h_x) - (1-y).*log(1 - h_x))) + lambda/(2*m)*sum(thet.^2);

grad = (1/m)*(X'*(h_x - y)) + (lambda/m)*thet;

end
