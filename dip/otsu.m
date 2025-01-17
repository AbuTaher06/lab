function otsu_edge_detection(image_path)
    % Read the image
    img = imread(image_path);
    
    % Convert to grayscale if the image is colored
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Convert the image to double for processing
    img = double(img);
    
    % Step 1: Compute Histogram of the image
    [counts, bin] = imhist(uint8(img));
    total_pixels = numel(img);
    
    % Step 2: Compute the Otsu Threshold
    % Calculate the cumulative sum and cumulative mean
    cumulative_sum = cumsum(counts);
    cumulative_mean = cumsum(counts .* (0:255));
    
    % Total mean of the image
    total_mean = cumulative_mean(end) / total_pixels;
    
    % Compute between-class variance for all possible thresholds
    class_variance = zeros(1, 256);
    for t = 1:255
        weight_background = cumulative_sum(t) / total_pixels;
        weight_foreground = 1 - weight_background;
        
        if weight_background == 0 || weight_foreground == 0
            continue;
        end
        
        mean_background = cumulative_mean(t) / cumulative_sum(t);
        mean_foreground = (cumulative_mean(end) - cumulative_mean(t)) / (cumulative_sum(end) - cumulative_sum(t));
        
        class_variance(t) = weight_background * weight_foreground * (mean_background - mean_foreground)^2;
    end
    
    % Find the optimal threshold that maximizes the between-class variance
    [~, otsu_threshold] = max(class_variance);
    
    % Step 3: Apply the threshold to the image to detect edges
    otsu_edge_image = img > otsu_threshold;
    
    % Step 4: Display the original and thresholded images
    figure;
    subplot(1, 2, 1);
    imshow(uint8(img));
    title('Original Image');
    
    subplot(1, 2, 2);
    imshow(otsu_edge_image);
    title('Otsu Edge Detection Result');
end
