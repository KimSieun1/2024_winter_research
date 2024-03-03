function [mean_apen, sd_apen] = approximate_entropy3(eeg_data, m, r)
    num_channels = size(eeg_data, 1);
    apen_values = zeros(num_channels, 1);
    
    % Calculate ApEn for each channel
    for i = 1:num_channels
        channel_data = eeg_data(i, :);
        apen_values(i) = calculate_apen(channel_data, m, r);
    end
    
    % Calculate mean and standard deviation of ApEn values
    mean_apen = mean(apen_values);
    sd_apen = std(apen_values);
end

function apen = calculate_apen(data, m, r)
    N = 2000;
    apen_values = zeros(N - m + 1, 1);
    
    % Calculate ApEn values for each pattern
    for i = 1:(N - m + 1)
        C_m_count = 0;
        
        for j = 1:(N - m + 1)
            if max(abs(data(i:i+m-1) - data(j:j+m-1))) <= r
                C_m_count = C_m_count + 1;
            end
        end
        
        apen_values(i) = -log(C_m_count / (N - m + 1));
    end
    
    % Calculate ApEn value
    apen = mean(apen_values);
end