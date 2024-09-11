clc;clear all;
train=xlsread('train.xlsx');
train1=train(:,1:end-1)./100;
[a,b]=size(train1);
trainD=[];
for i=1:1:a
    trainD(:,:,:,i)=train1(i,:);
end

targetD=categorical(train(:,end));

%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([973 1 1]) % 22X1X1 refers to number of features per sample
    
    convolution2dLayer([20 1],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([20 1],'Stride',2)
    
     convolution2dLayer([20 1],16,'Padding','same')
     batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([20 1],'Stride',2,'Name', 'pool2')
    
    convolution2dLayer([20 1],32,'Padding','same')
     batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([20 1],'Stride',2,'Name', 'pool3')
    
     fullyConnectedLayer(100)
    fullyConnectedLayer(4) % 2 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',400, ...
    'MiniBatchSize', 64, ...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainD,targetD',layers,options);

%layer = 'pool2';
%featuresTrain = activations(net,trainD,layer,'OutputAs','rows');

predictedLabels = classify(net,trainD);
accuracy = sum(predictedLabels == targetD)/numel(targetD)
%% test
test=xlsread('test.xlsx');
test1=test(:,1:end-1)./100;
[a,b]=size(test1);
testD=[];
for i=1:1:a
    testD(:,:,:,i)=test1(i,:);
end

testYD=categorical(test(:,end));
testLabels = classify(net,testD);
accuracyTest = sum(testLabels == testYD)/numel(testYD)
