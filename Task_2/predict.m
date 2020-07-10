function p = predict(theta, X)

value = sigmoid(X*theta);
pos = value >= 0.5;
neg = value < 0.5;
value(pos) = true;
value(neg) = false;
p = value;

end
