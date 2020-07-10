function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma = [0.01,0.03,0.1,0.3,1,3,10,30];
val = zeros(1,3);

for i = 1:8
    Ctest = C(i);
    for j = 1:8
        sigmatest = sigma(j);
        model = svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmatest));
        prediction = svmPredict(model, Xval);
        error = mean(double(prediction ~= yval));
        val(end+1,:) = [Ctest, sigmatest, error];
    end
end
val(1,:) = [];

[~, I] = min(val(:,3));
C = val(I,1);
sigma = val(I,2);

end
