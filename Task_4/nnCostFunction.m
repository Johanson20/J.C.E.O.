function [J, grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1), X];
h1 = sigmoid(X*Theta1');
h1 = [ones(m, 1), h1];
h2 = sigmoid(h1*Theta2');
yk = y;
r1 = size(Theta1, 1);
r2 = size(Theta2, 1);

for i=1:num_labels
    index = yk == i;
    yk(:) = 0;
    yk(index) = 1;
    J = J + sum(-yk.*log(h2(:, i)) - (1-yk).*log(1-h2(:, i)));
    yk = y;
end
J = J/m + lambda/(2*m)*(sum(Theta1(r1+1:end).^2) + sum(Theta2(r2+1:end).^2));

yi = zeros(m, num_labels);
for i=1:num_labels
index = y == i;
yi(index, i) = 1;
end

delta3 = h2 - yi;
delta2 = delta3 * Theta2 .* (h1.*(1 - h1));
thet1 = zeros(size(Theta1));
thet2 = zeros(size(Theta2));
thet1(:, 2:end) = Theta1(:, 2:end) * (lambda/m);
thet2(:, 2:end) = Theta2(:, 2:end) * (lambda/m);

Theta2_grad = (Theta2_grad + delta3' * h1)/m + thet2;
Theta1_grad = (Theta1_grad + (delta2(:, 2:end))' * X)/m + thet1;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
