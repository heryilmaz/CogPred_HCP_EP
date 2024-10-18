function [new_mat, NRemEdges] = VirtualLesion(old_mat,network)

% Created by Hamdi Eryilmaz on 4/12/24

% This function virtually lesions (i.e., removes) a network from the
% correlation matrix and generates the new smaller matrix (networks are
% defined based on the Gordon 333 parcellation)
%
% new_mat: The new matrix after virtual lesioning
% NRemEdges: number of removed edges as a result of virtual lesioning

new_mat = old_mat;

switch network

    case 'DMN'

        myrange = [48:88];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'VIS'

        myrange = [89:127];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'FPN'

        myrange = [128:151];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'DAN'

        myrange = [152:183];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'VAN'

        myrange = [184:206];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'SAL'

        myrange = [207:210];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'CON'

        myrange = [211:250];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'SM'

        myrange = [251:288];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'SML'

        myrange = [289:296];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'AUD'

        myrange = [297:320];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'CP'

        myrange = [321:325];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

    case 'RSP'

        myrange = [326:333];
        new_mat(myrange,:) = [];
        new_mat(:,myrange) = [];
        NRemEdges = nnz(tril(old_mat,-1)) - nnz(tril(new_mat,-1)); 

end




