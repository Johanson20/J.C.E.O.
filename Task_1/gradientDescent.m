function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    delta = (alpha/m)*(X*theta - y);
    theta(1) = theta(1) - sum(delta);
    theta(2) = theta(2) - sum(delta.*X(:,2));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
