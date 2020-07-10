function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

hypo = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(hypo) - (1-y).*log(1-hypo));

grad = zeros(size(theta));

for i=1:length(theta)
    grad(i) = sum((1/m)*(hypo - y).*X(:,i));
end

end
