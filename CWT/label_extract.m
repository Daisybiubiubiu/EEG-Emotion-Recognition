%% default paramters
Participant = 32;
Video = 40;
Channel = 32;
addpath('C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\deap\data_preprocessed_matlab')


%% label process
op = 'label';
filename = [op '.txt'];filename;
fid = fopen( filename, 'wt' );
%%
output = zeros(32,40,4);
for participant = 1:Participant
    fprintf('\nworking on file number %d:\n', participant);
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    output(participant,:,:) = labels;
end
%%
output2d = reshape(output,[32*40,4]);
%%
for i = 1:32*40
    fprintf(fid, '%g,',output2d(i,:));
    fprintf(fid, '\n');
end
