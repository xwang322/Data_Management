clear all;
clc;
% for first restaurant, good restaurant
A1 = zeros(5,5,5);
A11 = [5,6,1,7,11];
A12 = [1,1,5,47,661];
A13 = [16,27,142,411,417];
A11_sum = sum(A11);
A12_sum = sum(A12);
A13_sum = sum(A13);
% last axis is for iterating prob in first user, which is tough
A1(:,:,1) = A11(1)/A11_sum;
A1(:,:,2) = A11(2)/A11_sum;
A1(:,:,3) = A11(3)/A11_sum;
A1(:,:,4) = A11(4)/A11_sum;
A1(:,:,5) = A11(5)/A11_sum;
% second last axis is for iterating prob in second user, which is nice
A1(:,1,:) = A1(:,1,:) * A12(1)/A12_sum;
A1(:,2,:) = A1(:,2,:) * A12(2)/A12_sum;
A1(:,3,:) = A1(:,3,:) * A12(3)/A12_sum;
A1(:,4,:) = A1(:,4,:) * A12(4)/A12_sum;
A1(:,5,:) = A1(:,5,:) * A12(5)/A12_sum;
% first axis is for iterating prob in third user, which is average
A1(1,:,:) = A1(1,:,:) * A13(1)/A13_sum;
A1(2,:,:) = A1(2,:,:) * A13(2)/A13_sum;
A1(3,:,:) = A1(3,:,:) * A13(3)/A13_sum;
A1(4,:,:) = A1(4,:,:) * A13(4)/A13_sum;
A1(5,:,:) = A1(5,:,:) * A13(5)/A13_sum;
A1 = A1*100;

% for the second restaurant, bad restaurant
A2 = zeros(5,5,5);
A21 = [141,39,5,4,1];
A22 = [2,1,4,14,10];
A23 = [205,126,89,33,11];
A21_sum = sum(A21);
A22_sum = sum(A22);
A23_sum = sum(A23);
% last axis is for iterating prob in first user, which is tough
A2(:,:,1) = A21(1)/A21_sum;
A2(:,:,2) = A21(2)/A21_sum;
A2(:,:,3) = A21(3)/A21_sum;
A2(:,:,4) = A21(4)/A21_sum;
A2(:,:,5) = A21(5)/A21_sum;
% second last axis is for iterating prob in second user, which is nice
A2(:,1,:) = A2(:,1,:) * A22(1)/A22_sum;
A2(:,2,:) = A2(:,2,:) * A22(2)/A22_sum;
A2(:,3,:) = A2(:,3,:) * A22(3)/A22_sum;
A2(:,4,:) = A2(:,4,:) * A22(4)/A22_sum;
A2(:,5,:) = A2(:,5,:) * A22(5)/A22_sum;
% first axis is for iterating prob in third user, which is average
A2(1,:,:) = A2(1,:,:) * A23(1)/A23_sum;
A2(2,:,:) = A2(2,:,:) * A23(2)/A23_sum;
A2(3,:,:) = A2(3,:,:) * A23(3)/A23_sum;
A2(4,:,:) = A2(4,:,:) * A23(4)/A23_sum;
A2(5,:,:) = A2(5,:,:) * A23(5)/A23_sum;
A2 = A2 * 100;

% for the third restaurant, average restaurant
A3 = zeros(5,5,5);
A31 = [219,262,97,55,9];
A32 = [5,12,56,437,853];
A33 = [517,997,1770,1370,307];
A31_sum = sum(A31);
A32_sum = sum(A32);
A33_sum = sum(A33);
% last axis is for iterating prob in first user, which is tough
A3(:,:,1) = A31(1)/A31_sum;
A3(:,:,2) = A31(2)/A31_sum;
A3(:,:,3) = A31(3)/A31_sum;
A3(:,:,4) = A31(4)/A31_sum;
A3(:,:,5) = A31(5)/A31_sum;
% second last axis is for iterating prob in second user, which is nice
A3(:,1,:) = A3(:,1,:) * A32(1)/A32_sum;
A3(:,2,:) = A3(:,2,:) * A32(2)/A32_sum;
A3(:,3,:) = A3(:,3,:) * A32(3)/A32_sum;
A3(:,4,:) = A3(:,4,:) * A32(4)/A32_sum;
A3(:,5,:) = A3(:,5,:) * A32(5)/A32_sum;
% first axis is for iterating prob in third user, which is average
A3(1,:,:) = A3(1,:,:) * A33(1)/A33_sum;
A3(2,:,:) = A3(2,:,:) * A33(2)/A33_sum;
A3(3,:,:) = A3(3,:,:) * A33(3)/A33_sum;
A3(4,:,:) = A3(4,:,:) * A33(4)/A33_sum;
A3(5,:,:) = A3(5,:,:) * A33(5)/A33_sum;
A3 = A3 * 100;

% print in python format
for i = 5:5
    for j = 1:5
        tmp1 = num2str(A3(j,1,i));
        tmp2 = num2str(A3(j,2,i));
        tmp3 = num2str(A3(j,3,i));
        tmp4 = num2str(A3(j,4,i));
        tmp5 = num2str(A3(j,5,i));
        M = ['[',tmp1,',',tmp2,',',tmp3,',',tmp4,',',tmp5,']']
    end
end
% X1 = ['[',num2str(A3(1,1,5)),',',num2str(A3(1,2,5)),',',num2str(A3(1,3,5)),',',num2str(A3(1,4,5)),',',num2str(A3(1,5,5)),']'];
% X2 = ['[',num2str(A3(2,1,5)),',',num2str(A3(2,2,5)),',',num2str(A3(2,3,5)),',',num2str(A3(2,4,5)),',',num2str(A3(2,5,5)),']'];
% X3 = ['[',num2str(A3(3,1,5)),',',num2str(A3(3,2,5)),',',num2str(A3(3,3,5)),',',num2str(A3(3,4,5)),',',num2str(A3(3,5,5)),']'];
% X4 = ['[',num2str(A3(4,1,5)),',',num2str(A3(4,2,5)),',',num2str(A3(4,3,5)),',',num2str(A3(4,4,5)),',',num2str(A3(4,5,5)),']'];
% X5 = ['[',num2str(A3(5,1,5)),',',num2str(A3(5,2,5)),',',num2str(A3(5,3,5)),',',num2str(A3(5,4,5)),',',num2str(A3(5,5,5)),']'];
% D = ['[',X1,',',X2,',',X3,',',X4,',',X5,']'];
% disp(D);          

% devTitle = char('Thomas R. Lee','Sr. Developer','SFTware Corp.','')
% devTitle(2,:) = 'this is shit';
            
            
            
            
