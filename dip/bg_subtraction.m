function background_subtraction(video_path)
    % Create a video reader object to read the video
    video = VideoReader(video_path);
    
    % Read the first frame as the background model
    frame = readFrame(video);
    background = rgb2gray(frame);  % Convert the frame to grayscale
    
    % Create a figure to display the results
    figure;
    
    % Process each frame in the video
    while hasFrame(video)
        % Read the next frame
        frame = readFrame(video);
        
        % Convert the current frame to grayscale
        gray_frame = rgb2gray(frame);
        
        % Step 1: Background Subtraction (Compute absolute difference)
        foreground = abs(double(gray_frame) - double(background));
        
        % Step 2: Thresholding to identify foreground (moving objects)
        threshold_value = 30;  % You can adjust this threshold value
        foreground_binary = foreground > threshold_value;
        
        % Step 3: Display the current frame and the foreground mask
        subplot(1, 2, 1);
        imshow(frame);
        title('Original Frame');
        
        subplot(1, 2, 2);
        imshow(foreground_binary);
        title('Foreground Detected');
        
        % Pause to display the frame for a short period
        pause(0.05);
        
        % Optionally update the background model (here using the current frame)
        background = 0.95 * double(background) + 0.05 * double(gray_frame); % Update background with a weighted average
    end
end
