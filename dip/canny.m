function canny_edge_detection(image_path, low_threshold, high_threshold)
    % Read the image
    img = imread(image_path);
    
    % Convert to grayscale if the image is colored
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Convert the image to double for processing
    img = double(img);

    % Step 1: Gaussian Smoothing (Noise Reduction)
    kernel = fspecial('gaussian', [5 5], 1);
    smoothed_img = conv2(img, kernel, 'same');

    % Step 2: Compute the gradients using Sobel operator
    % Sobel kernel for detecting edges
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = sobel_x';
    
    % Apply Sobel operator to the image
    gradient_x = conv2(smoothed_img, sobel_x, 'same');
    gradient_y = conv2(smoothed_img, sobel_y, 'same');
    
    % Calculate the magnitude of the gradient and direction
    magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
    direction = atan2(gradient_y, gradient_x);
    
    % Normalize the direction to be in the range [0, 180]
    direction = mod(direction, pi);
    direction = direction * 180 / pi;

    % Step 3: Non-Maximum Suppression
    suppressed_img = non_maximum_suppression(magnitude, direction);

    % Step 4: Apply Double Thresholding (Edge Tracking by Hysteresis)
    edge_img = edge_tracking_by_hysteresis(suppressed_img, low_threshold, high_threshold);

    % Display the results
    figure;
    subplot(1, 2, 1);
    imshow(uint8(img));
    title('Original Image');

    subplot(1, 2, 2);
    imshow(edge_img);
    title('Canny Edge Detection Result');
end

% Non-Maximum Suppression
function suppressed_img = non_maximum_suppression(magnitude, direction)
    [rows, cols] = size(magnitude);
    suppressed_img = zeros(rows, cols);
    
    for i = 2:rows-1
        for j = 2:cols-1
            angle = direction(i, j);
            % Check the gradient direction and suppress non-maximal pixels
            if (angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)
                neighbor1 = magnitude(i, j-1);
                neighbor2 = magnitude(i, j+1);
            elseif (angle >= 22.5 && angle < 67.5)
                neighbor1 = magnitude(i-1, j+1);
                neighbor2 = magnitude(i+1, j-1);
            elseif (angle >= 67.5 && angle < 112.5)
                neighbor1 = magnitude(i-1, j);
                neighbor2 = magnitude(i+1, j);
            else
                neighbor1 = magnitude(i-1, j-1);
                neighbor2 = magnitude(i+1, j+1);
            end
            
            % Keep the pixel if it is the local maximum in the gradient direction
            if (magnitude(i, j) >= neighbor1) && (magnitude(i, j) >= neighbor2)
                suppressed_img(i, j) = magnitude(i, j);
            else
                suppressed_img(i, j) = 0;
            end
        end
    end
end

% Edge Tracking by Hysteresis
function edge_img = edge_tracking_by_hysteresis(suppressed_img, low_threshold, high_threshold)
    [rows, cols] = size(suppressed_img);
    edge_img = zeros(rows, cols);

    % Apply high and low thresholds
    strong_edge = suppressed_img > high_threshold;
    weak_edge = suppressed_img >= low_threshold & suppressed_img <= high_threshold;

    % Edge tracking: weak edges connected to strong edges are retained
    edge_img(strong_edge) = 1;

    for i = 2:rows-1
        for j = 2:cols-1
            if weak_edge(i, j)
                % Check for connectivity to strong edges
                if any(any(strong_edge(i-1:i+1, j-1:j+1)))
                    edge_img(i, j) = 1;
                end
            end
        end
    end
end
