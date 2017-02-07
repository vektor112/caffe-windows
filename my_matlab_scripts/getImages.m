% This script making image set from cifar-10 image data set. 
% Creating a .txt file in <image> <label> \n format,
% and save images into \imageSET\ folder.

% param 
% @data imagees as matrix
% @labels labels for images
% @directory destination directory path
% @extension extension of files
function [] = getImages(data, labels, directory, extension)
    newFolderName = '\imageSET\';
    
    if exist([directory newFolderName],'dir') == 0
        mkdir(directory, newFolderName);
    else
        error('The imageSET folder already exist.')
    end    
         
    imageFolder = [directory, newFolderName];
    
    fileID = fopen([directory '\imageSET.txt'],'wt');
    
     for n = 1:size(data, 1)

        row = data(n, :); 
        I = reshape(row, 32, 32, 3);
        strLabel = int2str(labels(n));
        fileName = [int2str(n) '_' strLabel '.' extension];
        imwrite(I,[imageFolder fileName]);
        fprintf(fileID, [fileName ' ' strLabel ' ' '\n']);

     end
    
    fclose(fileID);
end