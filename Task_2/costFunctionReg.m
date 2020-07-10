function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

hypo = sigmoid(X*theta);
thet = theta;
thet(1) = 0;
J = (1/m)*sum(-y.*log(hypo) - (1-y).*log(1-hypo)) + (lambda/(2*m))*sum(thet.^2);
grad = zeros(size(theta));

grad(1) = sum((1/m)*(hypo - y).*X(:,1));
for i=2:length(theta)
    grad(i) = sum((1/m)*(hypo - y).*X(:,i)) + (lambda/m)*theta(i);
end

end
