function [setSelectedCube, numSelectedCube] = coxReductionR1(Y,X)

% This function performs the first round of Cox Reduction in a cube based 
% on a linear response and design matrix. It retains variables that 
% are among the two most significant in at least 2/3 regressions. 

% Input:
% - Y (the response vector): a vector
% - X (the design matrix without an intercept column): a matrix with the 
%   same number of columns as Y

% Output:
% - setSelectedCube (the variable indices retained after the first round of
%   Cox Reduction): a vector consisting of numbers indexing columns of X
% - numSelectedCube (the number of variables retained after the first round
%   of Cox Reduction): an integer


[n,p]=size(X);

%Randomly arrange variable indices in a cube of suitable dimension
dimCube=ceil(nthroot(p,3));
nearestCube=dimCube^3;
remainderCube=nearestCube-p;
cubeVars=[1:p,zeros(1,remainderCube)];
permuteVec=randperm(nearestCube);
cube=reshape(cubeVars(permuteVec),dimCube,dimCube,dimCube);

%%%%   Perform regressions along cube fibres to select variables  %%%%
cubeSelect=zeros(size(cube)); %hypercube that counts the number of times each variable is selected

%%% First fibre %%%%%
    for indL=1:size(cube,3)
        for indR=1:size(cube,1)
            if size(nonzeros(cube(indR,:,indL)),1)>0
                subsetX=X(:,nonzeros(cube(indR,:,indL)));
                nonZeroIdx=find(cube(indR,:,indL)); 
                regressX=[ones(n,1),subsetX];
                [~,tStat]=regress_hsb(Y,regressX);
                tStat(1)=[];
                [~,sortIndex]=sort(abs(tStat),'descend');
                if length(sortIndex)>1
                    idxSelected=sortIndex(1:2); % only select the two most significant variables 
                else
                    idxSelected=sortIndex;
                end
                if size(idxSelected,1)>0
                    cubeSelect(indR,nonZeroIdx(idxSelected),indL)=cubeSelect(indR,nonZeroIdx(idxSelected),indL)+1;
                end
            end
        end %indR

        %%% Second fibre %%%%%
        for indC=1:size(cube,2)
            if size(nonzeros(cube(:,indC,indL)),1)>0
                subsetX=X(:,nonzeros(cube(:,indC,indL)));
                nonZeroIdx=find(cube(:,indC,indL));
                regressX=[ones(n,1),subsetX];
                [~,tStat]=regress_hsb(Y,regressX);
                tStat(1)=[];
                [~,sortIndex]=sort(abs(tStat),'descend');
                if length(sortIndex)>1
                    idxSelected=sortIndex(1:2); % only select the two most significant variables
                else
                    idxSelected=sortIndex;
                end
                if size(idxSelected,1)>0
                    cubeSelect(nonZeroIdx(idxSelected),indC,indL)=cubeSelect(nonZeroIdx(idxSelected),indC,indL)+1; 
                end
            end
        end % indC
    end %indL

    %%% Third fibre %%%%%
    for indR=1:size(cube,1)
        for indC=1:size(cube,2)
            if size(nonzeros(cube(indR,indC,:)),1)>0
                subsetX=X(:,nonzeros(cube(indR,indC,:)));
                nonZeroIdx=find(cube(indR,indC,:));
                regressX=[ones(n,1),subsetX];
                [~,tStat]=regress_hsb(Y,regressX);
                tStat(1)=[];
                [~,sortIndex]=sort(abs(tStat),'descend');
                if length(sortIndex)>1
                    idxSelected=sortIndex(1:2); % only select the two most significant variables
                else
                    idxSelected=sortIndex;
                end
                if size(idxSelected,1)>0
                    cubeSelect(indR,indC,nonZeroIdx(idxSelected))=cubeSelect(indR,indC,nonZeroIdx(idxSelected))+1; 
                end
            end
        end % indC
    end %indR

    %Identify the variables selected in at least 2/3 regressions
    setSelectedCube=cube(find(cubeSelect>1)); 
    numSelectedCube=length(setSelectedCube);
end

