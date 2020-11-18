%% clear all/ close figs
close all
clear
clc

%% default paramters
Participant = 32;
Video = 40;
Channel = 32;
Fs = 128;
Time = 63;
addpath('C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\deap\data_preprocessed_matlab')

%% set parameters
frameNum = 60;
totalScale = 64;
exScale = 32;
wname = 'db4';

%%

for participant = 1:Participant
    fprintf('\nworking on file number %d:\n', participant);
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    
    for video=1:Video
        
        fprintf('\ncreating file participant %d,video %d:\n',participant,video);
        op1 = 'participant';
        op2 = 'video';
        filename = [op1 int2str(participant) op2  int2str(video) '.txt'];filename;
        fid = fopen( filename, 'wt' );
        %fprintf(filename);
        output = zeros(Channel, totalScale, frameNum);                
        datastart=128*3;
        datalength=8064-128*3;
      
        
        for channel = 1:32
            data1=zeros(1,8064-datastart);
            for ii =1:datalength
                data1(1,ii)=data(video,channel,ii+datastart);
            end
   
            % decompose into wavelets
            % set scales
            f = 1:totalScale;
            f = Fs/totalScale * f;
            wcf = centfrq(wname);
            scal =  Fs * wcf./ f;           
            
            coefs = cwt(data1, 1:totalScale, wname);
            
            % Figures
            % 3d Figure
            %surf(coefs);shading interp;
            sc = wscalogram('no',coefs);
            
                       
            frameSize = 60/frameNum;
            start = 1;
            % split frames
            data2 = zeros(totalScale,frameNum);
            for k = 1: frameNum
                output(channel,:,k) = sum(sc(:,start:frameSize*k), 2);
                start = start + frameSize;
            end
             fprintf('.');
            
        end
        exoutput = output(:,8:39,:);
        output2d = reshape(exoutput,[Channel*exScale,frameNum]);
        for k = 1:Channel*exScale
            fprintf(fid, '%g,',output2d(k,:));
            fprintf(fid, '\n');
        end
       
        fclose(fid);
    end %the testcase loop
end %the file loop

