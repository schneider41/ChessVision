% ***************************************
% Chess Vision - Digital Image Processing
%                                           
% Roy Schneider, Ariel Moshe, Ram Shirazi
% ***************************************

clc
clear
close all
addpath(genpath('.'))

disp("Ariel Moshe - 312549660")
disp("Roy Schneider - 312538150")
disp("Ram Shirazi - 204272058")

isPlot = true; % bolean for plots
url = "http://192.168.1.16:8020/videoView";
cam = ipcam(url);
% cam = webcam(1);

%% Step 1: Calibration
waitfor(helpdlg({'Welcome to ChessVision'; 'Please setup the board and place white pawn on 1A square.'},'Welcome'))
while true
    preview(cam)
    waitfor(helpdlg('Press OK when board setup is ready', 'Calibration'));
    closePreview(cam)
    while true
        I = cam.snapshot;
        tform = calibrateBoard(I, false);
        if ~isempty(tform)
            break
        end
    end
    
    button = questdlg('Continue or calibrate again?', 'Calibration', 'Continue', 'Calibrate', 'Continue');
    if strcmp(button, 'Calibrate')
        close
    else
        if ~isPlot
            close
        end
        break
    end
end

%% Step 2: Run game
reset = true;
while reset
    
    % Pieces classification
    while true
        waitfor(helpdlg('Press Initialize when pieces are placed', 'Initialize'));
        I = cam.snapshot;
    %     [pieceNames] = classifyPieces(I, tform, isPlot);
        pieceNames = defaultBoard();
        fen = board2fen(reshape(pieceNames,8,8));
        ilegalBoard = displayBoard(fen);
        if ilegalBoard
            waitfor(helpdlg('Ilegal board!!! Please try again'));
        else
            break
        end
    end


    button = questdlg('Whos turn?', 'Choose sides', 'White', 'Black', 'White');
    if strcmp(button, 'White')
        whiteTurn = true;
    elseif strcmp(button, 'Black')
        whiteTurn = false;
    end
 
    % Moves tracking
    reset = trackMoves(cam, tform, pieceNames, whiteTurn, isPlot);
end

%% Creating Dataset
%% Capture images
path = fullfile('images');
if ~exist(path, 'dir')
    mkdir(path)
end

button = questdlg('Do you want to take images?','Creating dataset','Yes','No','Yes');
if strcmp(button, 'Yes')
    preview(cam)
    questdlg('Take image?','Creating dataset','Yes','No','Yes');
    while strcmp(button, 'Yes')
        I = cam.snapshot;
        imgNum = dir([path '/*.jpg']);
        imgNum = numel(imgNum) + 1;
        imwrite(I, fullfile(path, [num2str(imgNum) '.jpg']));
        button = questdlg('Take another image?','Creating dataset','Yes','No','Yes');
    end
    closePreview(cam)
end

%% Lable images

path = fullfile('images','noise');
if ~exist(path, 'dir')
    mkdir(path)
end

images = imageDatastore(path,'IncludeSubfolders', false, 'LabelSource', 'foldernames');
for i = 13:length(images.Files)
    disp(' ')
    disp(['image num: ' num2str(i)])
    disp(['image file: ' images.Files{i}])
    
    I = imread(images.Files{i});
    %     I = imcrop(I,[241,21,520,698]);
    figure(11)
    [I, roi] = imcrop(I);
    close
    tform = calibrateBoard(I, true);
    
    figure(12)
    imshow(I)
    
    if isempty(tform)
        I = imread(images.Files{i});
        I = imcrop(I,roiBackup);
        roi = roiBackup;
        tform = tfom_backup;
        
        %     continue;
    end
    
    labelImage(I, tform);
    close
    button = questdlg('Continue to next image?','Labeling Images','Yes','No','Yes');
    if strcmp(button, 'No')
        break
    end
    tfom_backup = tform ;
    roiBackup = roi;
end

%% Arrange images into datasets
for c = 1:Consts.NUMCLASSES
    createDataset(c);
end

%% Create Classifiers
for ratioGroup = 1
    disp(['Clasiffying group ' num2str(ratioGroup)])
    rootFolder = fullfile('data', num2str(ratioGroup));
    categories = Consts.CLASSNAMES;
    trainingSet = imageDatastore(rootFolder,'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    createClassifier(rootFolder);
end

%% Functions

% main functions
function tform = calibrateBoard(image, manual)
tform = [];
rot90 = false;
rot180 = false;
gridPoints = generateCheckerboardPoints(Consts.BOARDSIZE, Consts.SQUARESIZE) + Consts.SQUARESIZE;

% find transformation
[detectedPoints,boardSize] = detectCheckerboardPoints(image,'MinCornerMetric',0.1);
if  ~isequal(boardSize,Consts.BOARDSIZE)
    disp('Failed to detect board')
    return
else
    tform = fitgeotrans(detectedPoints,gridPoints,'projective');
end

% change major axes of squares counting in case low-left corner isn't black
transImage = imwarp(image,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
BW = binBoard(transImage);
boardMeans = calcSqaureMeans(BW);

if manual
    figure
    imshow(transImage)
    button = questdlg('Rotate by 90 deg counter clock wise?', 'Manual calibration', 'Yes','No','Yes');
    if strcmp(button, 'Yes')
        rot90 = true;
    end
    close
end

if (boardMeans(8,1) > boardMeans(1,1) && ~manual) || (manual && rot90)
    detectedPoints = reshape(detectedPoints, [7 7 2]);
    detectedPoints = permute(fliplr(detectedPoints), [2 1 3]);
    detectedPoints = reshape(detectedPoints, [49 2]);
    tform = fitgeotrans(detectedPoints,gridPoints,'projective');
    transImage = imwarp(image,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
    BW = binBoard(transImage);
    boardMeans = calcSqaureMeans(BW);
end

% rotate transform by 180 deg in case calibration pawn located in up-right corner
if manual
    figure
    imshow(transImage)
    button = questdlg('Rotate by 180 deg?', 'Manual calibration', 'Yes','No','No');
    if strcmp(button, 'Yes')
        rot180 = true;
    end
    close
end

if (boardMeans(1,8) > boardMeans(8,1) && ~manual) || (manual && rot180)
    detectedPoints = flip(detectedPoints);
    tform = fitgeotrans(detectedPoints,gridPoints,'projective');
end

if ~manual
    transImage = imwarp(image,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
    gridPoints = generateCheckerboardPoints(Consts.BOARDSIZE, Consts.SQUARESIZE) + Consts.SQUARESIZE;
    figure(1)
    sgtitle('Board Calibration')
    subplot(1,2,1)
    imshow(image)
    hold on
    plot(detectedPoints(:,1),detectedPoints(:,2),'ro')
    subplot(1,2,2)
    imshow(transImage)
    hold on
    plot(gridPoints(:,1),gridPoints(:,2),'ro')
    pause(1.5)
end
end

function [pieceNames] = classifyPieces(image, tform, isPlot)
transImage = imwarp(image,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));

% get prediction matricies of the board from all classifiers
classes = cell(1,64);

for s = 0:63
    ratioGroup = getRatioGroupByRow(floor(s/8));
    filename = fullfile('data',num2str(ratioGroup),'classifier.mat');
    load(filename,'classifier');
    square = cropSquare(image, tform, s);
    classes{s+1} = predict(classifier,square);
end

labels = "";
for l = 1:numel(classes)
    labels(l) = Consts.CLASSNAMES{classes{l}};
end
labels(labels == 'E') = '';

[~, occupiedSqaures] = find(labels ~= '');
squareColor = ~(checkerboard(1) > 0.5);
squareColor = squareColor(:);
BW = binBoard(transImage);
boardMeans = calcSqaureMeans(BW);

pieceNames = labels;
pieceColors = labels;
for s = 1:length(occupiedSqaures)
    pos = occupiedSqaures(s);
    if squareColor(pos) % white square
        if boardMeans(pos) > 0.7 % white on white
            pieceColors(pos) = 'W';
        else % black on white
            pieceColors(pos) = 'b';
            pieceNames(pos) = lower(pieceNames(pos));
        end
    else % black square
        if boardMeans(pos) > 0.3 % white on black
            pieceColors(pos) = 'W';
        else % black on black
            pieceColors(pos) = 'b';
            pieceNames(pos) = lower(pieceNames(pos));
        end
    end
end

if isPlot
    squareSize = Consts.SQUARESIZE;
    [x,y] = meshgrid((1:squareSize:8*squareSize) + round(squareSize/2), ...
        (1:squareSize:8*squareSize) + round(squareSize/2));
    figure(3)
    sgtitle("Piece Classification")
    subplot(1,2,1)
    imshow(transImage);
    pause(0.7)
    text(x(:),y(:),labels,'HorizontalAlignment','center','FontSize', 15, 'Color','r' )
    title('Detected Pieces')
    
    subplot(1,2,2)
    sgtitle("Color Recognition")
    imshow(transImage);
    pause(0.7)
    text(x(:),y(:),pieceColors,'HorizontalAlignment','center','FontSize', 15, 'Color','r' )
    title('Color Recognition')
end

end

function reset = trackMoves(cam, tform, pieceNames, whiteTurn, isPlot)

I = cam.snapshot;
transImage = imwarp(I,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
BW = threshBoard(transImage);
numPixels = numel(find(BW));
refImage = im2single(rgb2gray(I));

thresh = 0.03;

state = 0;
reset = false;

figure(4)
sgtitle('Game Tracking')
buttonHandle = uicontrol('Style', 'PushButton', 'String', 'Stop game', 'Callback', 'delete(gcbf)');

while true
    if ~ishandle(buttonHandle)
        break
    end
    I = cam.snapshot;
    transImage = imwarp(I,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
    BW = threshBoard(transImage);
    newNumPixels = numel(find(BW));
    metric = abs(newNumPixels - numPixels);
    currentImage = im2single(rgb2gray(I));
    diffImages = abs(refImage - currentImage);
    diffImages = medfilt2(diffImages);
    transImage = imwarp(diffImages,tform,'OutputView',imref2d(Consts.BOARDSIZE * Consts.SQUARESIZE));
    diffMeans = calcSqaureMeans(transImage);
    
    
    
    switch state
        case 0 % track for hand appering
            if metric > 1500
                disp(' ')
                disp('Hand detected')
                state = 1;
                if isPlot
                    figure(4)
                    subplot(1,3,1)
                    imshow(BW)
                    title('Detected Hand')
                    drawnow
                end
            end
            
        case 1 % wait until hand removed
            if metric < 1500
                disp(' ')
                disp('Hand removed')
                state = 2;
                
                if isPlot
                    figure(4)
                    subplot(1,3,2)
                    imshow(BW)
                    title('No Hand Detected')
                    drawnow
                end
            end
            
        case 2 % check for changes in board squares means
            [~, suspectedPos] = sort(diffMeans(:));
            suspectedPos = suspectedPos(end-1:end);
            if sum(diffMeans(suspectedPos) > thresh,'all') ~= 2
                state = 3;
            else
                state = 4;
            end
            
            if isPlot
                figure(4)
                subplot(1,3,3)
                imshow(diffImages,[])
                title('Diffrences Between Images')
                drawnow
            end
            
        case 3 % no move occours
            disp(' ')
            disp('Mo move was played')
            
            % update references
            numPixels = newNumPixels;
            refImage = currentImage;
            state = 0;
            
        case 4 % move occours
            [pos1, pos2] = deal(suspectedPos(1), suspectedPos(2));
            disp(' ')
            disp('Detecting move')
            if pieceNames(pos1) == ""
                [pos1, pos2] = deal(pos2, pos1);
            end
            
            % find the color of the piece in position 1
            if upper(pieceNames(pos1)) == pieceNames(pos1)
                whitePiece = true;
            else
                whitePiece = false;
            end
            
            % decied starting and ending points based on whos turn
            if ~xor(whiteTurn, whitePiece)
                startPos = pos1;
                endPos = pos2;
            else
                startPos = pos2;
                endPos = pos1;
            end
            
            newMove = pos2move(startPos, endPos);
            iligalMove = checkIfValid(newMove);
            
            if iligalMove
                disp('Wrong move! Please return the pieces to last position')
                state = 5;
            else
                % update references
                numPixels = newNumPixels;
                refImage = currentImage;
                
                [pieceNames(startPos), pieceNames(endPos)] = deal("",pieceNames(startPos));
                
                % TO DO - present colored squares%%%%%%%%%%%%%%%%%%%%%%%%%5
                
                whiteTurn = ~whiteTurn;
                disp(' ')
                disp(["Detected Move: " newMove])
                state = 0;
            end
            
        case 5 % handle ilegal moves
            [~, suspectedPos] = sort(diffMeans(:));
            suspectedPos = suspectedPos(end-1:end);
            if sum(diffMeans(suspectedPos) > thresh,'all') ~= 2
                numPixels = newNumPixels;
                refImage = currentImage;
                disp(' ')
                disp('please continue')
                state = 0;
            end
    end
end

button = questdlg('Do you wish for a new game?','Game stopped','Yes','No','Yes');

inputFile = fopen('GUI/input.txt','a');
if strcmp(button, 'Yes')
    fprintf(inputFile, '%s\n', 'reset');
    reset = true;
else
    fprintf(inputFile, '%s\n', 'stop');    
end
fclose(inputFile);

end


% auxiliary functions
function BW = binBoard(image)
image = rgb2gray(image);
image = im2single(image);
image = medfilt2(image,[7,7]);
image = imgaussfilt(image,2,'FilterSize',7);
level = graythresh(image);
BW = imbinarize(image,level);
end

function means = calcSqaureMeans(image)
% This function calculates the mean value of each sqaure in a board
% image and returns 8x8 matrix means

% radius = floor(Consts.SQUARESIZE/5);
% pad = floor((Consts.SQUARESIZE - (2 * radius + 1))/2);
% kernel = fspecial('disk',radius);
% kernel = padarray(kernel,[pad,pad],0);
kernel = fspecial('average',Consts.SQUARESIZE);
means = blockproc(image, [Consts.SQUARESIZE,Consts.SQUARESIZE], ...
    @(block) conv2(block.data,kernel,'valid'),'BorderSize', [Consts.SQUARESIZE, Consts.SQUARESIZE],'TrimBorder', true);
end

function [croppedImage] = cropSquare(image, tform, squareNum)
% This function crops square from real world board image
% param squareNum: number between 0 - 63 represent board squares by row
% major counting

squareSize = Consts.SQUARESIZE;
col = floor(squareNum/8);
row = mod(squareNum,8);
ratioGroup = getRatioGroupByRow(row);
ratio = Consts.RATIO(ratioGroup);

southwest = transformPointsInverse(tform,[col, row + 1] * squareSize);
southeast = transformPointsInverse(tform,[col + 1, row + 1] * squareSize);

x = southwest(1);
dx = southeast(1) - x ;
dy = ratio * dx;
y =  max(southwest(2), southeast(2)) - dy;

croppedImage = imcrop(image,[x,y,dx,dy]);
croppedImage = imresize(croppedImage,[squareSize*ratio,squareSize],'bilinear');
end

function ratioGroup = getRatioGroupByRow(rowNum)
% This functions returns the ratio group of the sqaure located in a given
% col

if 0 <= rowNum && rowNum <= 1
    ratioGroup = 3;
elseif 2 <= rowNum && rowNum <= 4
    ratioGroup = 2;
elseif 5 <= rowNum && rowNum <= 7
    ratioGroup = 1;
else
    ratioGroup = 0;
end
end

function BW = threshBoard(image)
lab = rgb2lab(image);
BW = lab(:,:,1) > 50 & lab(:,:,3) < 10;
end

function board = defaultBoard()
% Board intialize example - starting position - default position
board(1:8,1:8) = "";

%White
board(1,1) = "r";
board(1,8) = "r";
board(1,2) = "n";
board(1,7) = "n";
board(1,6) = "b";
board(1,3) = "b";
board(1,4) = "q";
board(1,5) = "k";
board(2,:) = "p";

%Black
board(8,1) = "R";
board(8,8) = "R";
board(8,2) = "N";
board(8,7) = "N";
board(8,6) = "B";
board(8,3) = "B";
board(8,4) = "Q";
board(8,5) = "K";
board(7,:) = "P";
end

function fen = board2fen(board)
% This function gets an 8x8 string matrix represent the current board, and
% return a FEN string.
% We can use this function to start a game from any board position
% detected. An assumption is that the position is white to move.

fen = "";
freeSquares = 0;
rankFEN = "";
for i = 1:8
    for j = 1:8
        if (board(i,j)=="")
            freeSquares = freeSquares + 1;
        elseif (freeSquares ~= 0)
            rankFEN = rankFEN + int2str(freeSquares) + board(i,j);
            freeSquares = 0;
        else
            rankFEN = rankFEN + board(i,j);
        end
    end
    if (freeSquares ~= 0)
        rankFEN = rankFEN + int2str(freeSquares);
    end
    fen = fen + "/" + rankFEN;
    rankFEN = "";
    freeSquares = 0;
end

% delete the first '/' and add the missing FEN format parts
if (extractAfter(fen, "/") == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    fen = extractAfter(fen,"/") + " w KQkq - 0 1"; 
    disp('Default position');
else
    fen = extractAfter(fen,"/") + " w - - 0 1"; 
    disp('Not default position');
end
end

function newMove = pos2move(startPos, endPos)
% This function will return the currect way to represent moves for the
% engine communiction (via the python code).
% input - 2 int numbers, represent the from & to squares as numbers from
% 1-64.
% output - UCI move, e.g. e2e4

% e.g. the detected move was from the 20 square to the 22 square
% rank 'a' in the chess board conatin square 1 to 8, rank b 9 to 16 etc. 
% the black side is the top one, and we start counting from there.
%      1     9    17    25    33    41    49    57
%      2    10    18    26    34    42    50    58
%      3    11    19    27    35    43    51    59
%      4    12    20    28    36    44    52    60
%      5    13    21    29    37    45    53    61
%      6    14    22    30    38    46    54    62
%      7    15    23    31    39    47    55    63
%      8    16    24    32    40    48    56    64
rank = ["a","b","c","d","e","f","g","h"];

column_from = ceil(startPos/8);
row_from = 9 - mod(startPos,8);
if (row_from == 9)
    row_from = 1;
end

column_to = ceil(endPos/8);
row_to = 9 - mod(endPos,8);
if (row_to == 9)
    row_to = 1;
end

newMove = rank(column_from) + num2str(row_from) + rank(column_to) + num2str(row_to);
end


% communication with python functions
function [illegalBoard] = displayBoard(fen)
% This function write the initial position detected to the python code
% and gets a legallity feedback of the inital position (legallity loop
% until the position is legal)
% Input :  fen -  Formal string represention of a position in chess
% Output : ilegalBoard - bolean flag   

% interpeter = "/Library/Frameworks/Python.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python";
% pythonFile = "'/Users/royschneider/Documents/Studies/Year D/Semester A/Digital Image Processing/Final Project/code/GUI/main.py'";
% commandStr =  interpeter + " " + pythonFile;
% system(commandStr);

inputFile = fopen('GUI/input.txt','a');
fprintf(inputFile, '%s\n', fen);
fclose(inputFile);

while true
    inputFile = fopen('GUI/output.txt','r');
    tline = fgetl(inputFile);
    if ischar(tline)
        illegalBoard = str2double(tline);
        inputFile = fopen('GUI/output.txt','w');
        fclose(inputFile);
        break
    end
    fclose(inputFile);
end
end

function [ilegalMove] = checkIfValid(newMove)
% This function write the initial position detected to the python code
% and gets a legallity feedback of the inital position (legallity loop
% until the position is legal)
% Input :  fen -  Formal string represention of a position in chess
% Output : ilegalBoard - bolean flag   
 
inputFile = fopen('GUI/input.txt','a');
fprintf(inputFile, '%s\n', newMove);
fclose(inputFile);
 
while true
    inputFile = fopen('GUI/output.txt','r');
    tline = fgetl(inputFile);
    if ischar(tline)
        ilegalMove = str2double(tline);
        inputFile = fopen('GUI/output.txt','w');
        fclose(inputFile);
        break
    end
    fclose(inputFile);
end
end


% functions for database creation
function labelImage(image, tform)
path = fullfile('data');
if ~exist(path, 'dir')
    mkdir(path)
end

colors = {'w','b'};
listString = Consts.CLASSNAMES;
listString{end+1} = 'Next row';
listString{end+1} = 'Exit image';

s = 0;
while s < 64
    ratioGroup = getRatioGroupByRow(floor(s/8));
    
    groupPath = fullfile(path,num2str(ratioGroup));
    if ~exist(groupPath, 'dir')
        mkdir(groupPath)
    end
    
    pieceImage = cropSquare(image, tform, s);
    figure(10)
    imshow(pieceImage)
    title(['square num: ' num2str(s) ' ratio group: ' num2str(ratioGroup)])
    
    classNum = listdlg('ListString',listString,'PromptString','Select piece');
    if ~isempty(classNum)
        if classNum == 8
            s = (floor(s/8) + 1) * 8;
            continue
        elseif classNum == 9
            break
        end
        color = listdlg('ListString',colors,'PromptString','Select a Color');
        
        classPath = fullfile(groupPath,Consts.CLASSNAMES{classNum});
        if ~exist(classPath, 'dir')
            mkdir(classPath)
        end
        
        imgName = [Consts.CLASSNAMES{classNum}];
        if colors{color} == 'b'
            imgName = lower(imgName);
        end
        imgNum = dir([classPath '/' imgName '*.jpg']);
        imgNum = numel(imgNum);
        imwrite(pieceImage, fullfile(classPath, [imgName num2str(imgNum) '.jpg']));
    end
    
    s = s + 1;
end
close
end

function createDataset(class)

piece = Consts.CLASSNAMES{class};

savePath = fullfile('data');
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

savePath = fullfile(savePath,piece);
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

positivePath = fullfile(savePath,'positive');
if ~exist(positivePath, 'dir')
    mkdir(positivePath)
end

negativePath = fullfile(savePath,'negative');
if ~exist(negativePath, 'dir')
    mkdir(negativePath)
end

imagePath = fullfile('images',piece);
images = imageDatastore(imagePath,'IncludeSubfolders', true, 'LabelSource', 'foldernames');

r = Consts.RATIO(class);
s = Consts.SQUARESIZE;
x = 0;
dx = s;
dy = dx * r;
y = 2 * s - dy;

for i = 1:length(images.Files)
    I = imread(images.Files{i});
    I = imcrop(I,[x,y,dx,dy]);
    imwrite(I, fullfile(positivePath,[num2str(i) '.jpg']));
end

for c = 1:Consts.NUMCLASSES
    if c ~= class
        piece = Consts.CLASSNAMES(c);
        imagePath = fullfile('images',piece);
        images = imageDatastore(imagePath,'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        for i = 1:length(images.Files)
            I = imread(images.Files{i});
            I = imcrop(I,[x,y,dx,dy]);
            imgNum = dir([negativePath '/*.jpg']);
            imgNum = numel(imgNum);
            imwrite(I, fullfile(negativePath,[num2str(imgNum) '.jpg']));
        end
    end
end
end

function createClassifier(rootFolder)
% This function create piece classifer from a root folder containg the 
% dataset.
% param rootFolder: string

imds = imageDatastore(rootFolder,'IncludeSubfolders', true, 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp(tbl)

% Separate the sets into training and validation data
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');

% Create a Visual Vocabulary and Train an Image Category Classifier
bag = bagOfFeatures(trainingSet,'CustomExtractor',@BagOfHOGFeaturesExtractor,'StrongestFeatures',0.8,'verbose',false);
% bag = bagOfFeatures(trainingSet,'PointSelection','Detector','StrongestFeatures',1,'VocabularySize',800,'verbose',true);

% Training an image category classifier for the categories.
classifier = trainImageCategoryClassifier(trainingSet, bag,'verbose',false);
save(fullfile(rootFolder,'classifier'),'classifier');

% Test with the trainingSet and validationSet
disp('evaluate training set')
evaluate(classifier, trainingSet);
disp('evaluate validation set')
evaluate(classifier, validationSet);
end

function [features, featureMetrics, varargout] = BagOfHOGFeaturesExtractor(I)

[height,width,numChannels] = size(I);
if numChannels > 1
    grayImage = rgb2gray(I);
else
    grayImage = I;
end

[features, visual] = extractHOGFeatures(grayImage, 'BlockSize',[16,16],'CellSize',[4 4],'BlockOverlap',[4 4]);

featureMetrics = var(features,[],2);


% figure(1);
% imshow(grayImage);
% hold on;
% plot(visual); 

if nargout > 2
    varargout{1} = validPoints;
end
end