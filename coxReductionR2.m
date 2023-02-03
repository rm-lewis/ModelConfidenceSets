function [setSelectedSquare, numSelectedSquare,L2] = coxReductionR2(Y,X,L2Start,L2Jump, setSelectedCube, sizeCap)

% This function performs the second round of Cox Reduction in a square
% based on a linear response and a design matrix. Only those covariates 
% indexed by "setSelectedCube" are arranged in the square. The reduction 
% is performed by retaining all variables significant at level "L2Start"
% in at least 1/2 regressions. If more than "sizeCap" variables are 
% retained, the significance level is reduced by "L2Jump" and the
% process is repeated until at most "sizeCap" variables are retained.

% Input:
% - Y (the response vector): a vector
% - X (the design matrix without an intercept column): a matrix with the 
%   same number of rows as Y
% - L2Start: a number between 0 and 1
% - L2Jump: a number between 0 and L2Start
% - setSelected (indexes the covariates that are retained after the first 
%   round of Cox reduction): a vector where each entry indexes a column of X

% Output:
% - setSelectedSquare (indexes the covariates that are retained after the
%   second round of Cox reduction): a vector indexing columns of X
% - numSelectedSquare (number of variables retained after second round of
%   Cox reduction): an integer
% - L2 (final significance level): a real number between 0 and L2Start.


[n,d]=size(X);

%Randomly arrange covariates in a square of suitable dimension
numSelectedTwice=size(setSelectedCube,1);
dimSquare2=ceil(sqrt(numSelectedTwice)); 
nearestSquare2=dimSquare2^2;
remainderSquare2=nearestSquare2-numSelectedTwice;
squareVars=[setSelectedCube;zeros(remainderSquare2,1)];
perm=randperm(size(squareVars,1)); 
square2=reshape(squareVars(perm),dimSquare2,dimSquare2);
square2(:,all(square2==0,1))=[]; %removes columns that are all zeros


%Perform regressions along fibres of square
L2=L2Start;                                     %Initial significance level used for selection
numSelectedSquare=size(setSelectedCube,1); 
while numSelectedSquare>sizeCap && L2-L2Jump>0  %iterate until at most sizeCap variables are selected
    L2=L2-L2Jump;
    squareSelect2=zeros(size(square2));         %stores the number of times a variable is selected

    % first fibre
    for indR=1:size(square2,1)
        if size(nonzeros(square2(indR,:)),1)>0
            subsetX=X(:,nonzeros(square2(indR,:)));
            nonZeroIdx=find(square2(indR,:)); 
            regressX=[ones(n,1),subsetX];
            [~,tStat]=regress_hsb(Y,regressX);    
            tStat(1)=[];
            k=size(tStat,1);
            tCrit=tinv(1-L2/2, n-(k+1));
            idxSelected=find((abs(tStat))>tCrit); %select all variables significant at required level
            if size(idxSelected,1)>0
                squareSelect2(indR,nonZeroIdx(idxSelected))=squareSelect2(indR,nonZeroIdx(idxSelected))+1; 
            end
        end
    end % indR

    %second fibre
    for indC=1:size(square2,2)
        if size(nonzeros(square2(:,indC)),1)>0
            subsetX=X(:,nonzeros(square2(:,indC)));
            nonZeroIdx=find(square2(:,indC));
            regressX=[ones(n,1),subsetX];
            [~,tStat]=regress_hsb(Y,regressX);
            tStat(1)=[];
            k=size(tStat,1);
            tCrit=tinv(1-L2/2, n-(k+1));
            idxSelected=find((abs(tStat))>tCrit); %select all variables significant at required level
            if size(idxSelected,1)>0
                squareSelect2(nonZeroIdx(idxSelected),indC)=squareSelect2(nonZeroIdx(idxSelected),indC)+1;
            end
        end
    end % indC

    if any(squareSelect2(:)>1)
        setSelectedSquare2=square2(find(squareSelect2>1)); %extract the indices of variables selected 2 times
    else
        setSelectedSquare2=[];
    end
    if any(squareSelect2(:)>0)
        setSelectedSquare=square2(find(squareSelect2>0)); % extract the indices of variables selected at least once
        numSelectedSquare=length(setSelectedSquare);
    else
        setSelected21=[];
        setSelectedSquare=[];
        numSelectedSquare=0;
    end
end

end

