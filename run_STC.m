function results=run_STC(seq, res_path, bSaveImage)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% fprintf('asasasasasasasasasasas')
% close all
s_frames = seq.s_frames;
%%
fftw('planner','patient');
%% set path
%addpath('./data');
%img_dir = dir('./data/*.jpg');
%% initialization
%initstate = [161,65,75,95];%initial rectangle [x,y,width, height]
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];%initial size of the target
pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);%center of the target
%% parameters according to the paper
padding = 1;					%extra area surrounding the target
rho = 0.075;			        %the learning parameter \rho in Eq.(12)
sz = floor(target_sz * (1 + padding));% size of context region
%% parameters of scale update. See Eq.(15)
scale = 1;%initial scale ratio
lambda = 0.25;% \lambda in Eq.(15)
num = 5; % number of average frames
%% store pre-computed confidence map
alapha = 2.25;                    %parmeter \alpha in Eq.(6)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
dist = rs.^2 + cs.^2;
conf = exp(-0.5 / (alapha) * sqrt(dist));%confidence map function Eq.(6)
conf = conf/sum(sum(conf));% normalization
conff = fft2(conf); %transform conf to frequencey domain
%% store pre-computed weight window
hamming_window = hamming(sz(1)) * hann(sz(2))';
sigma = mean(target_sz);% initial \sigma_1 for the weight function w_{\sigma} in Eq.(11)
window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));% use Hamming window to reduce frequency effect of image boundary
window = window/sum(sum(window));%normalization


time = 0;  %to calculate FPS
%positions = zeros(numel(s_frames), 2);  %to calculate precision
rect_position = zeros(numel(s_frames), 4);


%%
for frame = 1:numel(s_frames),
    sigma = sigma*scale;% update scale in Eq.(15)
    window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));%update weight function w_{sigma} in Eq.(11)
    window = window/sum(sum(window));%normalization
 	%load image
    im = imread(s_frames{frame});	    
	if size(im,3) > 1,
		im = rgb2gray(im);
    end
    tic()
   	contextprior = get_context(im, pos, sz, window);% the context prior model Eq.(4)
    %%
    if frame > 1,
		%calculate response of the confidence map at all locations
	    confmap = real(ifft2(Hstcf.*fft2(contextprior))); %Eq.(11) 
       	%target location is at the maximum response
%         if frame>2
%             row_old=row;
%             col_old=col;
%         end
        max_response=max(confmap(:));
		[row, col] = find(confmap == max(confmap(:)), 1);
        if isempty(row)
           row= 100;
           col=100;
        end
        pos = pos - sz/2 + [row, col]; 
        contextprior = get_context(im, pos, sz, window);
        conftmp = real(ifft2(Hstcf.*fft2(contextprior))); 
        maxconf(frame-1)=max(conftmp(:));
        %% update scale by Eq.(15)
        if (mod(frame,num+2)==0)
            scale_curr = 0;
            for kk=1:num
               scale_curr = scale_curr + sqrt(maxconf(frame-kk)/maxconf(frame-kk-1));
            end            
            scale = (1-lambda)*scale+lambda*(scale_curr/num);%update scale
        end  
        %%
    end	
	%% update the spatial context model h^{sc} in Eq.(9)
   	contextprior = get_context(im, pos, sz, window); 
    hscf = conff./(fft2(contextprior)+eps);% Note the hscf is the FFT of hsc in Eq.(9)
    %% update the spatio-temporal context model by Eq.(12)
    if frame == 1,  %first frame, initialize the spatio-temporal context model as the spatial context model
		Hstcf = hscf;
    else
%         if max_response>0.0021
		%update the spatio-temporal context model H^{stc} by Eq.(12)
		Hstcf = (1 - rho) * Hstcf + rho * hscf;% Hstcf is the FFT of Hstc in Eq.(12)
%         end
    end
	%save position and calculate FPS
	%positions(frame,:) = pos;
	time = time + toc();
    
    %% visualization
	target_sz([2,1]) = target_sz([2,1])*scale;% update object size
    rect_position(frame,:) = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    
    if bSaveImage
        if frame == 1,  %first frame, create GUI
%             figure('Number','off', 'Name',['Tracker - ' video_path])
            im_handle = imshow(im, 'Border','tight', 'InitialMag',200);
            rect_handle = rectangle('Position',rect_position(frame,:), 'EdgeColor','g');
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position(frame,:))
            catch  %#ok, user has closed the window
                return
            end
        end
        imwrite(frame2im(getframe(gcf)),[res_path num2str(frame) '.jpg']); 
    end
    %drawnow; 

end
% if resize_image, rect_position = rect_position * 2; end

fps = numel(s_frames) / time;

disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = rect_position;%each row is a rectangle
results.fps = fps;

