

 clc
 clear all
                         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------Initial Parameter Setup---------------------------------%
Img_Folder = 'C:\ImageHandling\20210105_Daniel_LettuceFungiTimeLaps\20210105_Daniel_LettuceFungiTimeLaps\';%F:\2020.03.23_HAG_tomato_30min\raw
Save_Path  = 'C:\ImageHandling\20210105_Daniel_LettuceFungiTimeLaps\20210105_Daniel_LettuceFungiTimeLaps_Final\';

Img_Loop = 45;
Z_windows = 10 ;
Img_Depth_Number = 200;
Translate_Number = 6.4; %6.4 for 5X
%overlap_size = 170;
overlap_size = 242; %242 for 5X when the step is -1.35

%-----------------------------------------------------
    %weight_rings_0 = correct_matrix_ring  ('a', 128, Img_Folder, 1, Img_Depth_Number, Z_windows);
    
    weight_matrix_b = correct_matrix_histogram('b', Img_Folder, 1, Img_Depth_Number, Z_windows);
    weight_matrix_b_1 = rot90(weight_matrix_b, 1); 
    
    %weight_matrix_d = correct_matrix_histogram('d', Img_Folder, 1, Img_Depth_Number, Z_windows);
    %weight_matrix_d_1 = rot90(weight_matrix_d, -1);
    %correct_mean = ones(512,512, 5);  
    %weight_matrix_b_1 = correct_mean;
%--------------------------------------------------------------------------
 tform_45_degree = affine3d([ 1   0   0   0
                              0   1   0   0
               Translate_Number   0   1   0
                              0   0   0   1 ]);
mould_1 = ones(512,512, Img_Depth_Number);
mould_trans = imwarp(mould_1, tform_45_degree, 'interp', 'nearest', 'SmoothEdges', true);
mould_trans(mould_trans>0)=1;
mould_project_D1 = squeeze(sum(mould_trans, 3));
mould_project_D1 = mould_project_D1./mean2(mould_project_D1);
%mould_project_D2 = squeeze(sum(mould_trans, 2));
%--------------------------------------------------------------------------
y = 1/overlap_size : 1/overlap_size : 1;
x = 1:1:size(mould_trans,2);
[X,Y] = meshgrid(x,y);
Y2 = rot90(Y, 2);

cell_data_info = {Img_Folder, Save_Path, Z_windows, Img_Depth_Number, overlap_size, tform_45_degree,  Translate_Number, Y, Y2}; 
% read folder; save_place; Z_windows; Img_Depth_Number, overlap_size, tform_45_degree 

%----------------------------------------
for i_Loop = 1:1:Img_Loop
        
   
   PureStich_front_and_side_projection ('b', weight_matrix_b_1, 1,  mould_project_D1, i_Loop, cell_data_info);
   PureStich_front_and_side_projection('a', weight_matrix_b_1, 4,  mould_project_D1, i_Loop, cell_data_info);
   %PureStich_front_and_side_projection('c', weight_matrix_b_1, 4,  mould_project_D1, i_Loop, cell_data_info);
   PureStich_front_and_side_projection('d', weight_matrix_b_1, 4,  mould_project_D1, i_Loop, cell_data_info);
 
   disp(i_Loop);
end



function PureStich_front_and_side_projection( laser_colour, correct_mean, method, mould_project_D1, i_Loop, cell_data_info)

 
Img_Folder = cell_data_info{1};
Save_Path  = cell_data_info{2};
Z_windows  = cell_data_info{3}; 
Img_Depth_Number = cell_data_info{4}; 
overlap_size = cell_data_info{5};
tform_45_degree= cell_data_info{6};
Translate_Number = cell_data_info{7};
Y  = cell_data_info{8};
Y2 = cell_data_info{9};

Loop_Number_str = num2str(i_Loop - 1);
Img_Loop_Folder = [Img_Folder 'LOOP_' Loop_Number_str '\'];

for i_z = 1:Z_windows
    w_mask_1 = squeeze(correct_mean(:,:, i_z));    
    w_mask_2 = imresize(w_mask_1, [256, 256],  'nearest');
    w_mask_3 = imresize(w_mask_1, [128, 128],  'nearest');
    w_mask_4 = imresize(w_mask_1, [64, 64],    'nearest');
    w_mask_5 = imresize(w_mask_1, [32, 32],    'nearest');
    w_mask_6 = (imresize(w_mask_1, [16, 16],   'nearest'));
    w_mask_7 = (imresize(w_mask_1, [8, 8],     'nearest'));
    w_mask_8 = (imresize(w_mask_1, [4, 4],     'nearest'));
       
    w_mask_1_N(:,:, i_z) = w_mask_1./ sum(sum(w_mask_1)).*512.*512;
    w_mask_2_N(:,:, i_z) = w_mask_2./(sum(sum(w_mask_2))).*256.*256;
    w_mask_3_N(:,:, i_z) = w_mask_3./(sum(sum(w_mask_3))).*128.*128;
    w_mask_4_N(:,:, i_z) = w_mask_4./(sum(sum(w_mask_4))) .*64.*64;
    w_mask_5_N(:,:, i_z) = w_mask_5./(sum(sum(w_mask_5))) .*32.*32; 
    w_mask_6_N(:,:, i_z) = w_mask_6./(sum(sum(w_mask_6))) .*16.*16;
    w_mask_7_N(:,:, i_z) = w_mask_7./(sum(sum(w_mask_7))) .*8.*8;
    w_mask_8_N(:,:, i_z) = w_mask_8./(sum(sum(w_mask_8))) .*4.*4;

end
                         
        
         for zt = 1:Z_windows
             Z_wndows_Number = num2str(zt);
             
             for i_depth = 1:1:Img_Depth_Number
                 str_i_depth = num2str(i_depth-1);
                 READ_NAME = [Img_Loop_Folder 'Z_Window_z'  Z_wndows_Number '\' laser_colour '_' str_i_depth '.tif'];
                 img_orin = imread(READ_NAME);                 
                 mrp_img = multiresolutionPyramid((im2double(img_orin)), 8);
                 lapp_origin = laplacianPyramid(mrp_img);
                 lapp_new{1} = lapp_origin{1} ./ (squeeze(w_mask_1_N(:,:, zt)));
                 lapp_new{2} = lapp_origin{2} ./ (squeeze(w_mask_2_N(:,:, zt)));
                 lapp_new{3} = lapp_origin{3} ./ (squeeze(w_mask_3_N(:,:, zt)));
                 lapp_new{4} = lapp_origin{4} ./ (squeeze(w_mask_4_N(:,:, zt)));
                 lapp_new{5} = lapp_origin{5} ./ (squeeze(w_mask_5_N(:,:, zt)));
                 lapp_new{6} = lapp_origin{6} ./ (squeeze(w_mask_6_N(:,:, zt)));
                 lapp_new{7} = lapp_origin{7} ./ (squeeze(w_mask_7_N(:,:, zt)));
                 lapp_new{8} = lapp_origin{8};% ./ (squeeze(w_mask_8_N(:,:, zt)));
                 %lapp_new{9} = lapp_origin{9};
                 img_new = reconstructFromLaplacianPyramid(lapp_new);                
                 img_stack(:,:, i_depth) = img_new;
                 %img_stack(:,:, i_depth) = img_orin;

             end

             
             transformed_stack = imwarp(img_stack, tform_45_degree, 'interp', 'nearest', 'SmoothEdges', true);
             %transformed_stack = im2uin16(transformed_stack);
             
                       if    method == 1
                             f  = max(transformed_stack, [], 3);
                             s  = max(transformed_stack, [], 2);
                             %front_projected_zstack(:,:, zt) = im2uint16(imresize(f, [482 1168]));
                             front_projected_zstack(:,:, zt) = im2uint16(f);
                             side_projected_zstack (:,:, zt) = im2uint16(s);
                             
                       elseif method == 2
                              f  =  squeeze(mean(transformed_stack, 3))./mould_project_D1;
                              s  =  squeeze(mean(transformed_stack, 2));
                              front_projected_zstack(:,:, zt) = im2uint16(f);
                              side_projected_zstack (:,:, zt) = im2uint16(s);
                       
                       elseif method == 3            
                              f  = im2uint16(squeeze(std(transformed_stack,1, 3))./mould_project_D1);
                              s  = im2uint16(squeeze(std(transformed_stack,1, 2)));
                              front_projected_zstack(:,:, zt) = im2uint16(f);
                              side_projected_zstack (:,:, zt) = im2uint16(s);
                              
                       elseif method == 4  
                           f  = maxk(transformed_stack, 10, 3);
                           p_size_x = size(f, 1);
                           p_size_y = size(f, 2);
                           
                           for img_y = 1:p_size_x
                               for img_x = 1:p_size_y
                                   z_img= squeeze( f(img_y, img_x, :));
                                   sum_z = sum(z_img);
                                   
                                   if  sum_z >= 32768
                                       z_xy(img_y, img_x) = mean(z_img );
                                   else
                                       z_xy(img_y, img_x) = max(z_img);
                                   end
                                
                               end
                           end
               
                           f_mean = z_xy;
                           %f_mean = mean(f, 3);
                           front_projected_zstack(:,:, zt) = im2uint16(f_mean);
                           
                           s  = maxk(transformed_stack, 7, 2);
                           s_mean = mean(s, 2);
                           side_projected_zstack(:,:, zt) = im2uint16(s_mean);
                           
                           
                       else
                           disp('Wrong Method input');
                       end
             
                    
         end
         
         
         
         
         if     laser_colour == 'a'
                folder_save_name = '633';
                img_save_name    = 'A';
              
         elseif laser_colour == 'b'
                folder_save_name = '561';
                img_save_name    = 'B';
              
         elseif laser_colour == 'c'
                folder_save_name = '514';
                img_save_name    = 'C';
                
         elseif laser_colour == 'd'
                folder_save_name = '488';
                img_save_name    = 'D';
         end
         
         front_Save_Folder = [Save_Path  'Front_MIP_' folder_save_name '\'];
         side_Save_Folder  = [Save_Path  'Side_MIP_'  folder_save_name '\'];
         if ~exist(front_Save_Folder, 'dir')
            mkdir (front_Save_Folder);
            mkdir (side_Save_Folder);
         end
         
         %modifed_image = front_projected_zstack(:, 271:1110,:);
         front_SAVE_NAME = [front_Save_Folder Loop_Number_str '_' img_save_name '.tif'];
         side_SAVE_NAME  = [side_Save_Folder  Loop_Number_str '_' img_save_name '.tif'];
         
         front_stiched_img = New_Pyramid_Fuse(front_projected_zstack, Z_windows, overlap_size);
         side_stiched_img =  New_Pyramid_Fuse(side_projected_zstack,  Z_windows, overlap_size);
         
         %side_stiched_img  = Stich_ExposureTime(side_projected_zstack,  Z_windows, overlap_size, Y ,Y2);
         %front_stiched_img = HalfBlend_PureStichV2(front_max_projected_zstack, Z_windows, overlap_size);
         %side_stiched_img  = HalfBlend_PureStichV2(side_max_projected_zstack, Z_windows, overlap_size);
         img_heigth = size(side_stiched_img, 1);
         img_width  = size(side_stiched_img, 2);
         side_img_2 = imresize(side_stiched_img,[img_heigth (round(abs(img_width*Translate_Number)))]);
         imwrite(front_stiched_img, front_SAVE_NAME, 'Compression', 'packbits');
         imwrite(side_img_2, side_SAVE_NAME,   'Compression', 'packbits');
             
end




function  Full_Img = New_Pyramid_Fuse(image_array, Z_Direction_Steps, overlap_size)

y_img = size(image_array,1);
rest_new_bottom = y_img - overlap_size;
mid_overlap = round(overlap_size/2);
x_num_1 =  size(image_array, 2);
maskfuse_Zeros = zeros(overlap_size, x_num_1); 
mask_A = maskfuse_Zeros;
% alpha = (1:1/100:0).^2;
alpha = 0:1/100:1;
mask_A(mid_overlap-50:mid_overlap+50,:) = (repmat(alpha, x_num_1,1))';
mask_A(mid_overlap+51:overlap_size,   :) =1;
mask_A(1:mid_overlap-51,   :) =0;
%%%%%%%%        First IMG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure; imagesc(weight_matrix_11);
        
        img_1_zeros_Bottom = uint16(zeros((rest_new_bottom*(Z_Direction_Steps - 1) + overlap_size),  x_num_1));
        img_1_Top    = image_array(1:overlap_size, :, 1);
        img_1_Middle = image_array(overlap_size+1 :rest_new_bottom, :,1);
        img_1_Bottom = image_array(rest_new_bottom + 1 :y_img, :, 1) ;
        img_fuse_1 = img_1_Bottom;
        Fusion_img_Part(:,:,1) = cat(1, img_1_Top, img_1_Middle, img_1_zeros_Bottom);
        
%%%%%%%%%%% Second IMG and Rest %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  for Window_Z = 2:1:(Z_Direction_Steps-1)                 
                      img_2_Top    = image_array(1:overlap_size, :, Window_Z);
                      B = im2double(img_fuse_1);
                      A = im2double(img_2_Top);                           
                      mrp_A = multiresolutionPyramid(A, 4);
                      mrp_B = multiresolutionPyramid(B, 4); 
                      mrp_mask_A = multiresolutionPyramid(mask_A, 4);
                      lap_A = laplacianPyramid(mrp_A);
                      lap_B = laplacianPyramid(mrp_B); 
                         for k = 1:length(lap_A)
                             lap_blend{k} = (lap_A{k} .* mrp_mask_A{k}) + (lap_B{k} .* (1 - mrp_mask_A{k}));
                         end
                      C_blended = reconstructFromLaplacianPyramid(lap_blend);

                     overlap_blend = im2uint16(C_blended);
                     %overlap_blend = blendexposure(img_2_Top, img_fuse_1);
                      img_2_Bottom = image_array(rest_new_bottom +1:y_img, :,Window_Z);                         
                      img_2_Middle = image_array(overlap_size+1 :rest_new_bottom, :,Window_Z);
                      img_fuse_1 = img_2_Bottom;
                         
                      img_2_ZerosTop    = uint16(zeros((rest_new_bottom * (Window_Z-1) ), x_num_1)); %(makeup_zeros*2 + x_num_B) ));
                      img_2_ZerosBottom = uint16(zeros((rest_new_bottom * (Z_Direction_Steps - Window_Z) + overlap_size),x_num_1)); %(makeup_zeros*2 + x_num_B) ));
  
                      
                      Fusion_img_Part(:,:, Window_Z) = cat(1, img_2_ZerosTop, overlap_blend, img_2_Middle, img_2_ZerosBottom);
                  end
%              zero_Middle = uint16(zeros(size(img_2_Middle)));             
%              Fusion_img_Part(:,:, Window_Z) = cat(1, img_2_ZerosTop, overlap_blend, zero_Middle, img_2_ZerosBottom);
%              
             Window_Z = Z_Direction_Steps;      
             img_3_Top  = image_array(1:overlap_size,:,Z_Direction_Steps);
             %img_fuse_3 = blendexposure(img_3_Top, img_fuse_1);
             B = im2double(image_array(rest_new_bottom +1:y_img, :,Z_Direction_Steps-1));
             A = im2double(img_3_Top);                           
             mrp_A = multiresolutionPyramid(A, 4);
             mrp_B = multiresolutionPyramid(B, 4); 
             mrp_mask_A = multiresolutionPyramid(mask_A, 4);
             lap_A = laplacianPyramid(mrp_A);
             lap_B = laplacianPyramid(mrp_B); 
             for k = 1:length(lap_A)
                 lap_blend{k} = (lap_A{k} .* mrp_mask_A{k}) + (lap_B{k} .* (1 - mrp_mask_A{k}));
             end
                 C_blended = reconstructFromLaplacianPyramid(lap_blend);
                 overlap_blend = im2uint16(C_blended);

             img_fuse_3 = overlap_blend;
             %img_3_Top    = uint16(round(img_3_Top .* 0.5));
             img_3_Bottom = image_array(overlap_size +1:y_img,:,Window_Z);
             img_3_ZerosTop = uint16(zeros((rest_new_bottom * (Window_Z-1) ), x_num_1));
             Fusion_img_Part(:,:, Z_Direction_Steps) = cat(1, img_3_ZerosTop, img_fuse_3, img_3_Bottom);
             
             %Fusion_img_Part(:,:, Z_Direction_Steps) = uint16(zeros(size(cat(1, img_3_ZerosTop, img_fuse_3, img_3_Bottom))));
             
              part_i = Fusion_img_Part(:,:, 1);
              for i_part = 2:1:Z_Direction_Steps
                  part_i = imlincomb(1, part_i, 1, squeeze(Fusion_img_Part(:,:,i_part)), 1);
%                   motion_top = (i_part-1)* rest_new_bottom -5;
%                   motion_bottom = (i_part-1)* rest_new_bottom +1;
%                   montion_part = imfilter(part_i,H_motion,'replicate');
%                   part_i(motion_top:motion_bottom, :, :) = montion_part(motion_top:motion_bottom, :, :);
              end
              

Full_Img = (part_i);

end





function  Full_Img = HalfBlend_ExposureTime(image_array, Z_Direction_Steps, overlap_size)

y_img = size(image_array,1);
rest_new_bottom = y_img - overlap_size;

x_num_1 =  size(image_array, 2);

mask_overlay = zeros( overlap_size, x_num_1);
mask_A = mask_overlay;
mask_A(1:round(overlap_size/2),:) =1;
%%%%%%%%        First IMG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        img_1_zeros_Bottom = uint16(zeros((rest_new_bottom*(Z_Direction_Steps - 1) + overlap_size),  x_num_1));
        img_1_Top    = image_array(1:(y_img-overlap_size),:,1);
        img_1_Bottom = image_array(rest_new_bottom + 1 :y_img, :, 1) ;
        img_fuse_1 = img_1_Bottom;
        %img_1_Bottom = uint16(round(img_1_Bottom .* 0.5));
        
        Fusion_img_Part(:,:,1) = cat(1, img_1_Top, img_1_zeros_Bottom);
        
%%%%%%%%%%% Second IMG and Rest %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  for Window_Z = 2:1:(Z_Direction_Steps-1)                 
                      
                         img_2_Top    = image_array(1:overlap_size, :, Window_Z);
                         C_blended = blendexposure(img_fuse_1, img_2_Top);
                        
                         img_2_Middle = image_array(overlap_size+1 :rest_new_bottom, :,Window_Z);
                         img_2_Bottom = image_array(rest_new_bottom +1:y_img, :,Window_Z);
                         img_fuse_1 = img_2_Bottom;
                         
                         img_2_ZerosTop = uint16(zeros((rest_new_bottom * (Window_Z-1) ), x_num_1)); %(makeup_zeros*2 + x_num_B) ));
                         img_2_ZerosBottom = uint16(zeros((rest_new_bottom * (Z_Direction_Steps - Window_Z) + overlap_size),x_num_1)); %(makeup_zeros*2 + x_num_B) ));
  
                         Fusion_img_Part(:,:, Window_Z) = cat(1, img_2_ZerosTop, C_blended, img_2_Middle, img_2_ZerosBottom);
                  end

                  
                  Window_Z = Z_Direction_Steps;      
                  img_3_Top    = image_array(1:overlap_size,:,Window_Z);
                                          
                  C_blended = blendexposure(img_fuse_1, img_3_Top);

                  
                  
                  img_fuse_3       = C_blended;    
                  img_3_Bottom = image_array(overlap_size +1:y_img,:,Window_Z);
                  img_3_ZerosTop = uint16(zeros((rest_new_bottom * (Window_Z-1) ), x_num_1));
                  Fusion_img_Part(:,:, Window_Z) = cat(1, img_3_ZerosTop, img_fuse_3, img_3_Bottom);
             
             
             if     Z_Direction_Steps == 3
                 Full_Img = imlincomb(1, Fusion_img_Part(:,:,1),  1, Fusion_img_Part(:,:,2), 1, Fusion_img_Part(:,:,3), 1); 
             elseif Z_Direction_Steps == 4
                 Full_Img = imlincomb(1, Fusion_img_Part(:,:,1),  1, Fusion_img_Part(:,:,2), 1, Fusion_img_Part(:,:,3), 1, Fusion_img_Part(:,:,4), 1);              
             elseif Z_Direction_Steps == 5
                 Full_Img = imlincomb(1, Fusion_img_Part(:,:,1),  1, Fusion_img_Part(:,:,2), 1, Fusion_img_Part(:,:,3), 1, Fusion_img_Part(:,:,4), 1, Fusion_img_Part(:,:,5), 1); 
             elseif Z_Direction_Steps == 10    
                 Full_Img = imlincomb(1, Fusion_img_Part(:,:,1),  1, Fusion_img_Part(:,:,2), 1, Fusion_img_Part(:,:,3), 1, Fusion_img_Part(:,:,4), 1, Fusion_img_Part(:,:,5), 1, Fusion_img_Part(:,:,6), 1, Fusion_img_Part(:,:,7), 1, Fusion_img_Part(:,:,8), 1, Fusion_img_Part(:,:,9), 1, Fusion_img_Part(:,:,10), 1); 
             elseif Z_Direction_Steps == 15
                 Full_Img = imlincomb(1, Fusion_img_Part(:,:,1),  1, Fusion_img_Part(:,:,2), 1, Fusion_img_Part(:,:,3), 1, Fusion_img_Part(:,:,4), 1, Fusion_img_Part(:,:,5), 1, Fusion_img_Part(:,:,6), 1, Fusion_img_Part(:,:,7), 1, Fusion_img_Part(:,:,8), 1, Fusion_img_Part(:,:,9), 1, Fusion_img_Part(:,:,10), 1, Fusion_img_Part(:,:,11), 1, Fusion_img_Part(:,:,12), 1, Fusion_img_Part(:,:,13), 1, Fusion_img_Part(:,:,14), 1, Fusion_img_Part(:,:,15),1); 
             end
             
             % Fusion_img_Part(:,:,6), 1, Fusion_img_Part(:,:,7), 1, Fusion_img_Part(:,:,8), 1, Fusion_img_Part(:,:,9), 1, Fusion_img_Part(:,:,10), 1, Fusion_img_Part(:,:,11));%,  1, Fusion_img_Part(:,:,12)); %, 1, Fusion_img_Part(:,:,13), 1, Fusion_img_Part(:,:,14), 1, Fusion_img_Part(:,:,15));     
             %Full_Img =  blendexposure( Fusion_img_Part(:,:,1),  Fusion_img_Part(:,:,2), Fusion_img_Part(:,:,3), Fusion_img_Part(:,:,4));% Fusion_img_Part(:,:,5),Fusion_img_Part(:,:,6),Fusion_img_Part(:,:,7),Fusion_img_Part(:,:,8),Fusion_img_Part(:,:,9),Fusion_img_Part(:,:,10));
     
%      se1 = strel('line', 256, 0); 
%      Img = imtophat(Img, se1);
%Fusion_img_Part(:,:,3), Fusion_img_Part(:,:,4), Fusion_img_Part(:,:,5),Fusion_img_Part(:,:,6),Fusion_img_Part(:,:,7),Fusion_img_Part(:,:,8),Fusion_img_Part(:,:,9),Fusion_img_Part(:,:,10) 
end



 



function  correct_mean = correct_matrix_mdian_mode(laser_colour, Img_Folder, the_LOOP, Img_Depth_Number, Z_windows)

bins_ranges =  -0.5 : 1 : 65536.5;

for j_y = 1:1:49
   for  i_x= 1:1:49
       mask = zeros(512,512);
       mask( (10*(i_x-1)+1):(10*(i_x-1)+32), (10*(j_y-1)+1):(10*(j_y-1)+32) ) = 1;
       mask_stack(:,:, i_x+ 49*(j_y-1) ) = mask;       
   end 
end

%%%%%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       for j_y1 = 1:1:4
%           for i_x1 = 1:1:63              
%               z0 = mask_stack(:,:, (i_x1 + 63*(j_y1-1)));%i_x+ 29*(j_y-1)
%               z = uint8(z0.*255);
%               imwrite(z, 'E:\Projection_2020_3_22_WT\T1.tif', 'writemode', 'append');
%           end
%       end




number_squares = size(mask_stack, 3);
Loop_Number_str = num2str(the_LOOP - 1);
%Img_Loop_Folder = [Img_Folder laser_colour '_LOOP_' Loop_Number_str '_'];%A_LOOP_0_Z_Window_z1
Img_Loop_Folder = [Img_Folder 'LOOP_' Loop_Number_str '\'];         
      for zt = 1:Z_windows
          Z_wndows_Number = num2str(zt);
             
             for i_depth = 1:1:Img_Depth_Number
%                READ_NAME = [Img_Loop_Folder 'Z_Window_z'  Z_wndows_Number '.tif']; 
%                img_stack(:,:, i_depth) = imread(READ_NAME, i_depth); 
                 str_i_depth = num2str(i_depth-1);
                 READ_NAME = [Img_Loop_Folder 'Z_Window_z'  Z_wndows_Number '\' laser_colour '_' str_i_depth '.tif'];
                 img_stack(:,:, i_depth) = imread(READ_NAME);                                  
             end
             
             %all_mean_value(zt) = double(mean(img_stack(:)));
             %all_mode_value(zt) = double(mode(img_stack(:)));
             %all_median_value(zt) = double(median(img_stack(:)));
             %rate_mean_mode(zt) = abs(all_mean_value(zt)/all_mode_value(zt));
             %all_min_value(zt) = double(min(img_stack(:)));
             %[histogram_a_stack,~]  =  histcounts(img_stack(:), bins_ranges);
             
             for i_square = 1:number_squares
                 i_mask = squeeze(mask_stack(:,:, i_square)); 
                 img_stack_squares = double(img_stack).* (i_mask);
                 [~,~, v_nonezero_square] = find(img_stack_squares); 
                 v_nonezero_square = double(v_nonezero_square);
                 v_square_median = median(v_nonezero_square(:));
                 v_square_mean   = mean(v_nonezero_square(:));
                 v_square_std   = std(v_nonezero_square(:));
                 v_square_mode  = mode(v_nonezero_square(:));
                 v_square_3m   = moment(v_nonezero_square,3,'all');
                 [N, ~] = histcounts(v_nonezero_square, bins_ranges);
%                  %number_size(i_square,zt) = max(size(v_nonezero_square));
                  N = imgaussfilt(N, [1 3]);
                  [~, index_mostcounts] = max(N);
                  hisogram_squares  =  double(index_mostcounts -1);
                  %delta_median_mode(i_square) =   v_square_mean - hisogram_squares;
                  %delta_median_mode(i_square) =  v_square_std;
                  v_skewness = v_square_3m/(v_square_std^3);
                  delta_median_mode(i_square) =  hisogram_squares;
                  %delta_median_mode(i_square) = v_square_mean - hisogram_squares(i_square); % hisogram_squares(i_square); %v_square_median(i_square) - hisogram_squares(i_square);
                  %delta_median_mode(i_square) =  v_square_mean - v_square_median(i_square);% - hisogram_squares(i_square);
             end 

          for j_y1 = 1:1:49
              for i_x1 = 1:1:49              
               %ij_squares_median = squeeze(v_square_median( i_x1 + 63*(j_y1-1)));
               %matrix_median(j_y1, i_x1) = ij_squares_median;
              
               %ij_squares_mean = squeeze(v_square_mean( i_x1 + 63*(j_y1-1)));
               %matrix_mean(j_y1, i_x1) = ij_squares_mean;
               
               ij_squares_delta = squeeze(delta_median_mode( i_x1 + 49*(j_y1-1)));
               matrix_delta(j_y1, i_x1) = ij_squares_delta;
              end
          end
          
      %weight_matrix_0  = double(matrix_mode); %.* squeeze(rate_mean_mode(zt));
      weight_matrix_0  = (matrix_delta);% - squeeze(all_min_value(zt)); 
      %surf(double(weight_matrix_0)./double(mode_value(zt)), 'EdgeColor', 'interp' );      
      %weight_matrix_01 = weight_matrix_0;

      weight_matrix_03 = imgaussfilt(weight_matrix_0, [2 2]);
      
      x_0 = 16:10:496;
      y_0 = 16:10:496;
      [X_0, Y_0] = meshgrid(x_0, y_0);
      
%       xq = min(x_0):1:max(x_0);
%       yq = min(y_0):1:max(y_0);

      xq = min(x_0):((max(x_0) - min(x_0))/512):max(x_0);
      yq = min(y_0):((max(y_0) - min(y_0))/512):max(y_0);
      [Xq, Yq] = meshgrid(xq, yq);
      weight_matrix_1  = interp2(X_0, Y_0, weight_matrix_03, Xq, Yq, 'linear');
      weight_matrix_11 = imresize(weight_matrix_1, [512 512], 'nearest');
      weight_matrix_2  = imgaussfilt(weight_matrix_11, [15 15]);
      figure; imagesc((weight_matrix_11./mean(weight_matrix_11(:))));
      %vector_weightmatrix = weight_matrix_2(:);
      %mean_value = mean(mean((weight_matrix_2));
      %mean_value = mean(vector_weightmatrix);
      correct_mean(:,:, zt)  = weight_matrix_2./mean(weight_matrix_2(:));
      figure; surf(correct_mean(:,:, zt),'EdgeColor', 'interp' );
      
      end
         
      
end

function  correct_mean = correct_matrix_histogram(laser_colour, Img_Folder, the_LOOP, Img_Depth_Number, Z_windows)

bins_ranges =  -0.5 : 1 : 65536.5;

for j_y = 1:1:4
   for  i_x= 1:1:63
       mask = zeros(512,512);
       mask( (8*(i_x-1)+1):(8*(i_x-1)+16), (32*(j_y-1)+1):(32*(j_y-1)+416) ) = 1;
       mask_stack(:,:, i_x+ 63*(j_y-1) ) = mask; 
       
   end 
end

%%%%%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       for j_y1 = 1:1:4
%           for i_x1 = 1:1:63              
%               z0 = mask_stack(:,:, (i_x1 + 63*(j_y1-1)));%i_x+ 29*(j_y-1)
%               z = uint8(z0.*255);
%               imwrite(z, 'E:\Projection_2020_3_22_WT\T1.tif', 'writemode', 'append');
%           end
%       end




number_squares = size(mask_stack, 3);
Loop_Number_str = num2str(the_LOOP - 1);
%Img_Loop_Folder = [Img_Folder laser_colour '_LOOP_' Loop_Number_str '_'];%A_LOOP_0_Z_Window_z1
Img_Loop_Folder = [Img_Folder 'LOOP_' Loop_Number_str '\'];         
      for zt = 1:Z_windows
          Z_wndows_Number = num2str(zt);
             
             for i_depth = 1:1:Img_Depth_Number
%                 READ_NAME = [Img_Loop_Folder 'Z_Window_z'  Z_wndows_Number '.tif']; 
%                 img_stack(:,:, i_depth) = imread(READ_NAME, i_depth); 
                 str_i_depth = num2str(i_depth-1);
                 READ_NAME = [Img_Loop_Folder 'Z_Window_z'  Z_wndows_Number '\' laser_colour '_' str_i_depth '.tif'];
                 img_stack(:,:, i_depth) = imread(READ_NAME);                                  
             end
             
             %all_mean_value(zt) = double(mean(img_stack(:)));
             %all_mode_value(zt) = double(mode(img_stack(:)));
             %all_median_value(zt) = double(median(img_stack(:)));
             %rate_mean_mode(zt) = abs(all_mean_value(zt)/all_mode_value(zt));
             %all_min_value(zt) = double(min(img_stack(:)));
             %[histogram_a_stack,~]  =  histcounts(img_stack(:), bins_ranges);
             
             for i_square = 1:number_squares
                 i_mask = squeeze(mask_stack(:,:, i_square)); 
                 img_stack_squares = img_stack.* uint16(i_mask);
                 [~,~, v_nonezero_square] = find(img_stack_squares);              
                 %v_square_median(i_square) = double(median(v_nonezero_square));
                 %v_square_mean(i_square)   = mean(v_nonezero_square);
                 %v_square_mode(i_square)   = mode(v_nonezero_square);
                 [N, ~] = histcounts(v_nonezero_square, bins_ranges);
%                  %number_size(i_square,zt) = max(size(v_nonezero_square));
                  N = imgaussfilt(N, [1 7]);
                  [~, index_mostcounts] = max(N);
                  hisogram_squares =  double(index_mostcounts -1);
                  delta_median_mode(i_square) = hisogram_squares;
                  %delta_median_mode(i_square) =  v_square_mean(i_square) - v_square_median(i_square);% - hisogram_squares(i_square);
             end 

          for j_y1 = 1:1:4
              for i_x1 = 1:1:63              
               %ij_squares_median = squeeze(v_square_median( i_x1 + 63*(j_y1-1)));
               %matrix_median(j_y1, i_x1) = ij_squares_median;
              
               %ij_squares_mean = squeeze(v_square_mean( i_x1 + 63*(j_y1-1)));
               %matrix_mean(j_y1, i_x1) = ij_squares_mean;
               
               ij_squares_delta = squeeze(delta_median_mode( i_x1 + 63*(j_y1-1)));
               matrix_delta(j_y1, i_x1) = ij_squares_delta;
              end
          end
          
      %weight_matrix_0  = double(matrix_mode); %.* squeeze(rate_mean_mode(zt));
      weight_matrix_0  = double(matrix_delta);% - squeeze(all_min_value(zt)); 
      %surf(double(weight_matrix_0)./double(mode_value(zt)), 'EdgeColor', 'interp' );
      column_left  = weight_matrix_0(1, :);
      column_right = weight_matrix_0(end, :);
      weight_matrix_01 = cat(1,  column_left, weight_matrix_0, column_right);
      
      %weight_matrix_01 = weight_matrix_0;
      
      row_top = weight_matrix_01(:, 1); 
      row_bottom = weight_matrix_01(:, 63);
      weight_matrix_02 = cat(2, row_top, weight_matrix_01, row_bottom);
      weight_matrix_03 = imgaussfilt(weight_matrix_02, [1 2]);
      
      x_0 = 1:8:513;
      y_0 =176:32:336;
      [X_0, Y_0] = meshgrid(x_0, y_0);
      
      xq = min(x_0):1:max(x_0);
      yq = min(y_0):1:max(y_0);
      [Xq, Yq] = meshgrid(xq, yq);
      weight_matrix_1  = interp2(X_0, Y_0, weight_matrix_03, Xq, Yq, 'linear');
      weight_matrix_11 = imresize(weight_matrix_1, [512 512], 'nearest');
      weight_matrix_2  = imgaussfilt(weight_matrix_11, [9 9]);
      
      %vector_weightmatrix = weight_matrix_2(:);
      %mean_value = mean(mean((weight_matrix_2));
      %mean_value = mean(vector_weightmatrix);
      correct_mean(:,:, zt)  = weight_matrix_2./mean2(weight_matrix_2(zt));
      %figure; surf(correct_mean(:,:, zt),'EdgeColor', 'interp' );
      
      end
         
      
end



function mrp = multiresolutionPyramid(A,num_levels)
%multiresolutionPyramid(A,numlevels)
%   mrp = multiresolutionPyramid(A,numlevels) returns a multiresolution
%   pyramd from the input image, A. The output, mrp, is a 1-by-numlevels
%   cell array. The first element of mrp, mrp{1}, is the input image.
%
%   If numlevels is not specified, then it is automatically computed to
%   keep the smallest level in the pyramid at least 32-by-32.

%   Steve Eddins
%   Copyright The MathWorks, Inc. 2019

%A = im2double(A);

M = size(A,1);
N = size(A,2);

if nargin < 2
    lower_limit = 32;
    num_levels = min(floor(log2([M N]) - log2(lower_limit))) + 1;
else
    num_levels = min(num_levels, min(floor(log2([M N]))) + 2);
end

mrp = cell(1,num_levels);

smallest_size = [M N] / 2^(num_levels - 1);
smallest_size = ceil(smallest_size);
padded_size = smallest_size * 2^(num_levels - 1);

Ap = padarray(A,padded_size - [M N],'replicate','post');

mrp{1} = Ap;
for k = 2:num_levels
    mrp{k} = imresize(mrp{k-1},0.5,'lanczos3');
end

mrp{1} = A;
end

function lapp = laplacianPyramid(mrp)

% Steve Eddins
% MathWorks

lapp = cell(size(mrp));
num_levels = numel(mrp);
lapp{num_levels} = mrp{num_levels};
for k = 1:(num_levels - 1)
   A = mrp{k};
   B = imresize(mrp{k+1},2,'lanczos3');
   [M,N,~] = size(A);
   lapp{k} = A - B(1:M,1:N,:);
end
lapp{end} = mrp{end};
end

function out = reconstructFromLaplacianPyramid(lapp)

% Steve Eddins
% MathWorks

num_levels = numel(lapp);
out = lapp{end};
for k = (num_levels - 1) : -1 : 1
   out = imresize(out,2,'lanczos3');
   g = lapp{k};
   [M,N,~] = size(g);
   out = out(1:M,1:N,:) + g;
end
end