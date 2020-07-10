function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1), X];
prod = sigmoid(X*Theta1');
prod = [ones(m, 1), prod];
value = sigmoid(prod*Theta2');

for i=1:num_labels
vv = value(:, i);
index = vv == max(value, [], 2);
p(index) = i;
end

end
