clear all;

%% Import Data
im_dir = 'C:\Users\Kim\AP_186\data\training\';

imds = imageDatastore(im_dir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% im_dir = 'C:\Users\Kim\AP_186\data\training\';
% 
% imdsValidation = imageDatastore(im_dir, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');

%% Load and Explore Image Data

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

labelCount = countEachLabel(imds);
% img = readimage(imds,1);
% size(img);



%% Specify Training and Validation Sets
% 
numTrainFiles = 10000;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Augement Images
% imageAugmenter = imageDataAugmenter( ...
%     'RandScale',[0.5 1]);

% imageSize = [28 28 1];
imageSize = [128 128 3];
% augimds = augmentedImageDatastore(imageSize,imds,'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain);
augimdsValidation = augmentedImageDatastore(imageSize,imdsValidation);

%% Define Network Architecture

layers = [
%     imageInputLayer([28 28 1])
    imageInputLayer([imageSize])
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%% Specify Training Options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'ValidationPatience', Inf,...
%     'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

% opts = trainingOptions('sgdm', ...
%     'MaxEpochs',15, ...
%     'Shuffle','every-epoch', ...
%     'Plots','training-progress', ...
%     'Verbose',false, ...
%     'ValidationData',{XValidation,YValidation});

%% Train Network Using Training Data

net = trainNetwork(augimdsTrain,layers,options);
% net = trainNetwork(augimds,layers,opts);

%% Save Fig

% %# neural net, and view it
% jframe = view(net);
% 
% %# create it in a MATLAB figure
% hFig = figure('Menubar','none', 'Position',[100 100 565 166]);
% jpanel = get(jframe,'ContentPane');
% [~,h] = javacomponent(jpanel);
% set(h, 'units','normalized', 'position',[0 0 1 1]);
% 
% %# close java window
% jframe.setVisible(false);
% jframe.dispose();
% 
% %# print to file
% set(hFig, 'PaperPositionMode', 'auto');
% saveas(hFig, 'out.png');
% 
% %# close figure
% close(hFig);

%% Classify Validation Images and Compute Accuracy

YPred = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

%%

YYPred = string(YPred);
YYValidation = string(YValidation);
%% Check
num_images = 10;
col = 5;
row = 2;

% figure;
% % perm = randperm(20000,num_images);
% for i = 1:num_images
%     subplot(row,col,i);
%     imshow(imdsTrain.Files{perm(i)});
%     title(YYPred((i)));
% end

% figure;
% perm = randperm(5000,num_images);
% for i = 1:num_images
%     subplot(row,col,i);
%     imshow(imdsValidation.Files{perm(i)});
%     title(YYValidation((i)));
% end

figure;
% perm = randperm(5000,num_images);
for i = 1:num_images
    subplot(row,col,i);
    imshow(imdsValidation.Files{perm(i)});
%     imshow(imdsValidation.Files{(i)});
    check = YYPred((perm(i)));
    if check == "class_a"
        xx = "cat";
    else
        xx = "dog";
    end
    title(xx);
%     suptitle('Test on the CNN');
end