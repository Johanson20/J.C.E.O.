function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

grad = zeros(size(theta));

hypo = X*theta;
J = (1/(2*m))*sum((hypo - y).^2);

grad(1) = sum((1/m)*(hypo - y).*X(:,1));
for i=2:length(theta)
    grad(i) = sum((1/m)*(hypo - y).*X(:,i)) + (lambda/m)*theta(i);
    J = J + (lambda/(2*m))*(theta(i).^2);
end

end
