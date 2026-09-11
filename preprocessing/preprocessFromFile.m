function img = preprocessFromFile(filename)

    img = imread(filename);
    img = preprocessImage(img);
    

end